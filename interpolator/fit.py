import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import lmfit
import pyphot
import emcee
import corner
import warnings

from dust_extinction.parameter_averages import F19
from dust_extinction.parameter_averages import F99
import astropy.units as u

from typing import Tuple
from . import interpolator

#physical constants in SI units
speed_light = 299792458 #m/s
radius_sun = 6.957e8 #m
mass_sun = 1.9884e30 #kg
newton_G = 6.674e-11 #N m^2/kg^2
pc_to_m = 3.086775e16

lib = pyphot.get_library()
def get_photo_bands(synthmag : pd.DataFrame):
    bands = [col for col in synthmag.columns if col in sorted(lib.content)]
    e_bands = [f"e_{col}" for col in bands]
    return bands, e_bands

def mag_to_flux(mag : np.array, e_mag : np.array, filters : np.array) -> Tuple[np.array, np.array]:
    """convert vega magnitudes to fluxes in flam units
    """
    flux = np.array([10**(-0.4*(mag[i] + lib[band].Vega_zero_mag)) for i, band in enumerate(filters)])
    #e_flux = np.array([np.abs((np.log(10)*(-0.4)*10**(-0.4 * (mag[i] + lib[band].Vega_zero_mag)) * e_mag[i])) for i, band in enumerate(filters)])
    e_flux = 1.09 * flux * e_mag
    return flux, e_flux

def get_model_flux(theta : np.array, factors : np.array, interpolator : interpolator.WarwickPhotometry, logg_function = None) -> np.array:
    """get model photometric flux for a WD with a given radius, located a given distance away
    """     
    mass_sun, radius_sun, newton_G, speed_light = 1.9884e30, 6.957e8, 6.674e-11, 299792458
    if logg_function == None:
        # if no logg function is provided, assume mass is provided and calculate
        teff, radius, distance, av, mass = theta
        logg = np.log10(100*(newton_G * mass_sun * mass) / (radius * radius_sun)**2)
    else:
        # if logg function is provided, use it
        teff, logg, distance, av = theta
        #radius = logg_function(teff, radius)
        radius = logg_function(teff, logg)
    fl = 4 * np.pi * interpolator(teff, logg) # flux in physical units
    #convert to SI units
    pc_to_m, radius_sun = 3.086775e16, 6.957e8
    radius *= radius_sun # Rsun to meter
    distance *= pc_to_m # Parsec to meter
    extinction = np.array([0.835*av, 1.139*av, 0.650*av])
    return (radius / distance)**2 * fl * F99(Rv=3.1).extinguish(factors, Av=av)#np.power(10.0, -0.4 * extinction) ### scale down flux by distance

def loss(params, fl, e_fl, factors, interp, logg_function = None):
    # ugly cases code. need to kill this with reason
    if logg_function == None:
        teff, radius, distance, av, mass = params.valuesdict().values()
        theta = np.array([teff, radius, distance, av, mass])
    else:
        teff, radius, distance, av = params.valuesdict().values()
        theta = np.array([teff, radius, distance, av])
    flux_model = get_model_flux(theta, factors=factors, interpolator=interp, logg_function=logg_function)
    return (fl - flux_model) / e_fl

def coarse_fit(flux : np.array, e_flux : np.array, interp : interpolator.WarwickPhotometry, distance : np.float64, av = np.float64,
                logg_function = None, vary_mass : bool = False, p0 : list = [10000, 8, 0.6], 
                coarse_kws : dict = {'nan_policy':'omit'}):
    # make parameters
    params = lmfit.Parameters()
    params.add('teff', value=p0[0], min=2000, max=120000, vary=True)
    params.add('logg', value=p0[1], min=7.1, max=9.4, vary=True)
    params.add('distance', value=distance, min=1, max=10000, vary=False)
    params.add('av', value=av, min=0.000001, max=2, vary=False)
    if logg_function is None:
        params.add('mass', value=0.6, min=0.1, max=1.4, vary=vary_mass)
    # perform fit
    factors = 0.0001*np.array([lib[band].lpivot.to('angstrom').value for band in interp.bands])*u.micron
    res = lmfit.minimize(loss, params, args = (flux, e_flux, factors, interp, logg_function), **coarse_kws)
    return res

class Likelihood:
    def __init__(self, flux : np.array, e_flux : np.array, interp = interpolator.WarwickPhotometry):
        # fluxes
        self.flux, self.e_flux = flux.astype(np.float64), e_flux.astype(np.float64)
        # interpolation
        self.interp = interp
        self.factors = 0.0001*np.array([lib[band].lpivot.to('angstrom').value for band in interp.bands])*u.micron

    def ll(self, theta, logg_function = None):
        flux_model = get_model_flux(theta, factors=self.factors, interpolator=self.interp, logg_function=logg_function)
        return -0.5 * np.sum((self.flux - flux_model)**2 / self.e_flux**2 + np.log(2 * np.pi * self.e_flux**2))
    
    def gaussian_prior(self, val : np.float64, true : np.float64, e_true : np.float64):
        return -0.5 * ((val - true)**2 / e_true**2 + np.log(2*np.pi*e_true**2))
    
    def uniform_prior(self, theta, bounds):
        log_prior = np.zeros(theta.shape[0])
        within_bounds = (theta >= bounds[:,0]) & (theta <= bounds[:,1])
        log_prior[~within_bounds] = -np.inf
        return log_prior.sum()

def mcmc_fit(loss_function, loss_args : dict, initial_guess : np.array = np.array([10000, 8, 100, 0.01, 0.6]), 
             discard : int = 0, printprogress : bool = True):
    # first, run 2500 steps of MCMC to understand how much we actually need to run
    nsteps = 1000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            init_pos = initial_guess + 1e-2*np.random.randn(50,initial_guess.shape[0])*initial_guess # intialize postions of walkers
            nwalkers, ndim = init_pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, loss_function, kwargs = loss_args)
            sampler.run_mcmc(init_pos, nsteps, progress = printprogress) # run x steps of mcmc
        except:
            init_pos = initial_guess + 1e-2*np.random.randn(50,initial_guess.shape[0])*initial_guess # intialize postions of walkers
            nwalkers, ndim = init_pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, loss_function, kwargs = loss_args)#, moves = [emcee.moves.StretchMove(a=1.75)])
            sampler.run_mcmc(init_pos, nsteps, progress = printprogress) # run x steps of mcmc
        auto_corr_time = np.max(sampler.get_autocorr_time(quiet = True)) # get amount of steps for burn-in so we know how many steps we should run
        try:
            #print(f"Auto-Correlation Time = {auto_corr_time}, additional steps = {int(52*auto_corr_time) - nsteps}")
            if np.isfinite(auto_corr_time):
                if nsteps <= int(52*auto_corr_time):
                    sampler.run_mcmc(None, int(52*auto_corr_time) - nsteps, progress = printprogress)
                flat_samples = sampler.get_chain(discard = discard, thin = int(0.5*auto_corr_time), flat = True)
            else:
                flat_samples = sampler.get_chain(discard = discard, flat = True)  
        except:
            flat_samples = sampler.get_chain(discard = discard, flat = True) 
    return flat_samples

def plot_chain(chain : np.ndarray, labels : list = [r'$T_\text{eff}$', r'Radius', r'd [pc]', r'$A_v$ [mag]', r'M $[M_\odot]$'],
                corner_kwargs = {'quantiles' : [0.16, 0.5, 0.84], 'show_titles' : True, 'title_fmt' : '.3f', 
                'title_kwargs':{"fontsize": 12}}):
    best_est = np.zeros((len(labels)))  
    for j in range(len(labels)):
        mcmc = np.percentile(chain[:,j],[16,50,84])
        best_est[j] = mcmc[1]
    discard = int(0.1*chain.shape[0])
    corner_fig = corner.corner(chain[discard:], labels=labels, truths = best_est, **corner_kwargs)
    return corner_fig