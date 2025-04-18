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
        teff, radius, distance, av = theta
        logg = logg_function(teff, radius)
    fl = 4 * np.pi * interpolator(teff, logg) # flux in physical units
    #convert to SI units
    pc_to_m, radius_sun = 3.086775e16, 6.957e8
    radius *= radius_sun # Rsun to meter
    distance *= pc_to_m # Parsec to meter
    extinction = np.array([0.835*av, 1.139*av, 0.650*av])
    return (radius / distance)**2 * fl * np.power(10.0, -0.4 * extinction)#* F99(Rv=3.1).extinguish(factors, Av=av)# scale down flux by distance

def loss(params, fl, e_fl, factors, interp, logprob = False, logg_function = None):
    # ugly cases code. need to kill this with reason
    if not logprob:
        if logg_function == None:
            teff, radius, distance, av, mass = params.valuesdict().values()
            theta = np.array([teff, radius, distance, av, mass])
        else:
            teff, radius, distance, av = params.valuesdict().values()
            theta = np.array([teff, radius, distance, av])
    else:
        if logg_function == None:
            teff, radius, distance, av, mass = params
            theta = np.array([teff, radius, distance, av, mass])
        else:
            teff, radius, distance, av = params
            theta = np.array([teff, radius, distance, av])
    flux_model = get_model_flux(theta, factors=factors, interpolator=interp, logg_function=logg_function)
    if logprob:
        return -0.5 * np.sum((fl - flux_model)**2 / e_fl**2 + np.log(2 * np.pi * e_fl**2))
    else:
        return (fl - flux_model) / e_fl

def gaussian_prior(val : np.float64, true : np.float64, e_true : np.float64):
    return -0.5 * ((val - true)**2 / e_true**2 + np.log(2*np.pi*e_true**2))

def log_prob(theta : np.array, flux : np.array, e_flux : np.array, factors : np.array, teff_obs : np.float64, e_teff_obs : np.float64, 
                radius_obs : np.float64, e_radius_obs : np.float64, plx : np.float64, e_plx : np.float64, av : np.float64, 
                e_av : np.float64, interpolator : interpolator.WarwickPhotometry):
    def log_prior(theta : np.array, teff_obs : np.float64, e_teff_obs : np.float64, radius_obs : np.float64, e_radius_obs : np.float64,
                    plx : np.float64, e_plx : np.float64, av_obs : np.float64, e_av_obs : np.float64):
        teff, radius, distance, av, mass = theta
        av_prior = gaussian_prior(av, av_obs, e_av_obs)
        distance_prior = gaussian_prior(1000/distance, plx, e_plx) + 2*np.log10(distance)
        teff_prior = gaussian_prior(teff, teff_obs, e_teff_obs)
        radius_prior = gaussian_prior(radius, radius_obs, e_radius_obs)
        #uniform prior
        log_prior = np.zeros(theta.shape[0])
        bounds = np.array([[1000, 100000], [0.002, 0.05], [10, 2000], [1e-4, 1], [0.1, 1.4]])
        within_bounds = (theta >= bounds[:,0]) & (theta <= bounds[:,1])
        log_prior[~within_bounds] = -np.inf
        return log_prior.sum() + av_prior + distance_prior + teff_prior + radius_prior
    lp = log_prior(theta = theta, teff_obs = teff_obs, e_teff_obs = e_teff_obs, radius_obs = radius_obs, e_radius_obs = e_radius_obs,
                    plx = plx, e_plx = e_plx, av_obs = av, e_av_obs = e_av)
    ll = loss(params = theta, fl = flux, e_fl = e_flux, factors = factors, interp = interpolator, logprob=True)
    if np.isnan(ll):
        ll = -np.inf
    if np.isnan(lp):
        lp = -np.inf
    return ll + lp

def coarse_fit(flux : np.array, e_flux : np.array, interp : interpolator.WarwickPhotometry, distance : np.float64, av = np.float64,
                logg_function = None, vary_mass : bool = False, p0 : list = [10000, 0.012, 0.6], 
                coarse_kws : dict = {'nan_policy':'omit'}):
    # make parameters
    params = lmfit.Parameters()
    params.add('teff', value=p0[0], min=2000, max=120000, vary=True)
    params.add('radius', value=p0[1], min=0.003, max=0.02, vary=True)
    params.add('distance', value=distance, min=10, max=3000, vary=False)
    params.add('av', value=av, min=0.000001, max=2, vary=False)
    if logg_function is None:
        params.add('mass', value=0.6, min=0.1, max=1.4, vary=vary_mass)
    # perform fit
    factors = 0.0001*np.array([lib[band].lpivot.to('angstrom').value for band in interp.bands])*u.micron
    res = lmfit.minimize(loss, params, args = (flux, e_flux, factors, interp, False, logg_function), **coarse_kws)
    return res

def mcmc_fit(flux : np.array, e_flux : np.array, plx : np.float64, e_plx : np.float64, av : np.float64, e_av : np.float64, 
                interp : interpolator.WarwickPhotometry, p0 : np.array = np.array([10000,0.01,0.6]), coarse_kws : dict = {}, 
                base_pct_err = 0.2, discard : int = 0, printprogress : bool = True):
    factors = 0.0001*np.array([lib[band].lpivot.to('angstrom').value for band in interp.bands])*u.micron
    res = coarse_fit(flux = flux, e_flux = e_flux, interp = interp, distance = 1000/plx, av = av, p0 = p0, coarse_kws = coarse_kws)
    names, theta_best = list(res.params.valuesdict().keys()), np.array(list(res.params.valuesdict().values()))
    try:
        theta_err = np.array([res.params[name].stderr for name in names])
    except:
        theta_err = np.zeros(len(names))
    theta_err = np.sqrt(theta_err**2 + (theta_best*base_pct_err)**2)
    args_dict = {'flux' : flux, 'e_flux' : e_flux, 'factors' : factors, 'teff_obs' : theta_best[0], 'e_teff_obs' : theta_err[0], 
                'radius_obs' : theta_best[1], 'e_radius_obs' : theta_err[1], 'plx' : plx, 'e_plx' : e_plx, 'av' : av, 
                'e_av' : e_av, 'interpolator' : interp}
    # first, run 2500 steps of MCMC to understand how much we actually need to run
    initial_guess = theta_best
    nsteps = 1000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            init_pos = initial_guess + 1e-4*np.random.randn(50,initial_guess.shape[0])*initial_guess # intialize postions of walkers
            nwalkers, ndim = init_pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, kwargs = args_dict)
            sampler.run_mcmc(init_pos, nsteps, progress = printprogress) # run x steps of mcmc
        except:
            init_pos = initial_guess + 1e-2*np.random.randn(50,initial_guess.shape[0])*initial_guess # intialize postions of walkers
            nwalkers, ndim = init_pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, kwargs = args_dict)#, moves = [emcee.moves.StretchMove(a=1.75)])
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
    return res, flat_samples

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