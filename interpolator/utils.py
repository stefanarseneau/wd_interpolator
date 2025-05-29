import matplotlib.pyplot as plt
import numpy as np
import pyphot

def plot(obs_mag, e_obs_mag, distance, radius, teff, logg, Engine):
    model_flux = 4 * np.pi * Engine.interpolator(teff, logg)
    #convert to SI units
    radius = radius * 6.957e8 # Rsun to meter
    distance = distance * 3.086775e16 # Parsec to meter
    model_flux = (radius / distance)**2 * model_flux

    lib = pyphot.get_library()
    model_wavl = [lib[band].lpivot.to('angstrom').value for band in Engine.bands]

    obs_flux, e_obs_flux = Engine.mag_to_flux(obs_mag,  e_obs_mag)

    f = plt.figure(figsize = (8,7))
    plt.scatter(model_wavl, model_flux, c = 'blue', label='Model Photometry')
    plt.errorbar(model_wavl, obs_flux, yerr=e_obs_flux, linestyle = 'none', marker = 'None', capsize = 5, color = 'k', label=f'Teff={teff:6.0f}\nlogg={logg:1.1f}')
    plt.xlim(2500,15000)
    plt.xlabel(r'Wavelength $[\AA]$')
    plt.ylabel(r'Flux $[erg/s/cm^2/\AA]$')
    return f

def correct_gband(bp, rp, astrometric_params_solved, phot_g_mean_mag):
    """Correct the G-band fluxes and magnitudes for the input list of Gaia EDR3 data.
    """
    bp_rp = bp - rp

    if np.isscalar(bp_rp) or np.isscalar(astrometric_params_solved) or np.isscalar(phot_g_mean_mag):
        bp_rp = np.float64(bp_rp)
        astrometric_params_solved = np.int64(astrometric_params_solved)
        phot_g_mean_mag = np.float64(phot_g_mean_mag)
    
    if not (bp_rp.shape == astrometric_params_solved.shape == phot_g_mean_mag.shape):
        raise ValueError('Function parameters must be of the same shape!')
    
    do_not_correct = np.isnan(bp_rp) | (phot_g_mean_mag<13) | (astrometric_params_solved == 31)
    bright_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>=13) & (phot_g_mean_mag<=16)
    faint_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>16)
    bp_rp_c = np.clip(bp_rp, 0.25, 3.0)

    correction_factor = np.ones_like(phot_g_mean_mag)
    correction_factor[faint_correct] = 1.00525 - 0.02323*bp_rp_c[faint_correct] + \
        0.01740*np.power(bp_rp_c[faint_correct],2) - 0.00253*np.power(bp_rp_c[faint_correct],3)
    correction_factor[bright_correct] = 1.00876 - 0.02540*bp_rp_c[bright_correct] + \
        0.01747*np.power(bp_rp_c[bright_correct],2) - 0.00277*np.power(bp_rp_c[bright_correct],3)
    
    gmag_corrected = phot_g_mean_mag - 2.5*np.log10(correction_factor)
    return gmag_corrected

# convert air wavelengths to vacuum
def air2vac(wv):
    _tl=1.e4/np.array(wv)
    return (np.array(wv)*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))
