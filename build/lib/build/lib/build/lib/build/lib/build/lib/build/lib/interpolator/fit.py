import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import lmfit
import pyphot
import emcee
import warnings

from dust_extinction.parameter_averages import F19
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
def mag_to_flux(mag : np.array, e_mag : np.array, filters : np.array) -> Tuple[np.array, np.array]:
    """convert vega magnitudes to fluxes in flam units
    """
    flux = np.array([10**(-0.4*(mag[i] + lib[band].Vega_zero_mag)) for i, band in enumerate(filters)])
    #e_flux = np.array([np.abs((np.log(10)*(-0.4)*10**(-0.4 * (mag[i] + lib[band].Vega_zero_mag)) * e_mag[i])) for i, band in enumerate(filters)])
    e_flux = 1.09 * flux * e_mag
    return flux, e_flux

def get_model_flux(theta : np.array, factors : np.array, interpolator : interpolator.WarwickDAInterpolator) -> np.array:
    """get model photometric flux for a WD with a given radius, located a given distance away
    """     
    mass_sun, radius_sun, newton_G, speed_light = 1.9884e30, 6.957e8, 6.674e-11, 299792458
    teff, radius, distance, av, mass = theta
    logg = np.log10(100*(newton_G * mass_sun * mass) / (radius * radius_sun)**2)
    fl = 4 * np.pi * interpolator(teff, logg) # flux in physical units
    #convert to SI units
    pc_to_m, radius_sun = 3.086775e16, 6.957e8
    radius *= radius_sun # Rsun to meter
    distance *= pc_to_m # Parsec to meter
    return (radius / distance)**2 * fl * F19(Rv=3.1).extinguish(factors, Av=av)# scale down flux by distance

def loss(params, fl, e_fl, factors, interp, logprob = False):
    teff, radius, distance, av, mass = params.valuesdict().values()
    theta = np.array([teff, radius, distance, av, mass])
    flux_model = radcode.get_model_flux(theta, factors=factors, interpolator=interp)
    if np.any(np.isnan(flux_model)):
        print(theta)
    if logprob:
        return -0.5 * np.sum((fl - flux_model)**2 / e_fl**2 + np.log(2 * np.pi * e_fl**2))
    else:
        return (fl - flux_model)**2 / e_fl**2

def coarse_fit(flux : np.array, e_flux : np.array, interp : interpolator.WarwickPhotometry, p0 : np.array = np.array([10000,0.01,100,0.004,0.6]),
                p0_min : np.array = np.array([1000, 0.005, 10, 0, 0.1]), p0_max : np.array = np.array([100000, 0.025, 2000, 0.2, 1.4]),
                vary : np.array = np.array([True, True, False, False, True]), coarse_kws : dict = {}):
    # unpack inputs into parameter list
    teff, radius, distance, av, mass = p0
    teff_min, radius_min, distance_min, av_min, mass_min = p0_min
    teff_max, radius_max, distance_max, av_max, mass_max = p0_min
    teff_vary, radius_vary, distance_vary, av_vary, mass_vary = vary
    # make parameters
    params = lmfit.Parameters()
    params.add('teff', value=teff, min=teff_min, max=teff_max, vary = teff_vary)
    params.add('radius', value=radius, min=radius_min, max=radius_max, vary = radius_vary)
    params.add('distance', value=distance, min=distance_min, max=distance_max, vary=distance_vary)
    params.add('av', value=av, min=av_min, max=av_max, vary=av_vary)
    params.add('mass', value=mass, min=mass_min, max=mass_max, vary=mass_vary)
    # perform fit
    factors = 0.0001*np.array([lib[band].lpivot.to('angstrom').value for band in interp.bands])*u.micron
    res = lmfit.minimize(loss, params, args = (flux, e_flux, factors, interp), **coarse_kws)
    return res

def mcmc_fit(flux : np.array, e_flux : np.array, interp : interpolator.WarwickPhotometry, p0 : np.array = np.array([10000,0.01,100,0.004,0.6]),
                p0_min : np.array = np.array([1000, 0.005, 10, 0, 0.1]), p0_max : np.array = np.array([100000, 0.025, 2000, 0.2, 1.4]),
                vary : np.array = np.array([True, True, False, False, True]), coarse_kws : dict = {}):
    res = coarse_fit(flux = flux, e_flux = e_flux, interp = interp, p0 = p0, p0_min = p0_min, p0_max = p0_max, vary  = vary, coarse_kws = coarse_kws)
    best_theta = res.params.valuesdict()
    return best_theta