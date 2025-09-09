from astropy.io import fits
import numpy as np
import os

class Filter:
    def __init__(self, wavl : np.array, transm : np.array, zerofile : str = 'alpha_lyr_mod_004'):
        """define a filter
        """
        self.wavl = wavl
        self.transm = transm
        self.type = type
        # determine the zeropoints
        dirname = os.path.dirname(os.path.abspath(__file__)) 
        spec = fits.open(os.path.join(dirname, 'reference', f'{zerofile}.fits'))
        wavl, flux = spec[1].data['WAVELENGTH'], spec[1].data['FLUX']
        self.zeropoint = -2.5*np.log10(self(flux, wavl))

    def __call__(self, flux, wavl = None):
        """integrate a spectrum through a filter
        """
        transm = np.interp(wavl, self.wavl, self.transm, left=0., right=0.) if wavl is not None else self.transm
        wavl = self.wavl if wavl is None else wavl
        num = np.trapezoid(flux*transm*wavl, wavl)
        den = np.trapezoid(transm*wavl, wavl)
        return num/den

class SED:
    def __init__(self, filters, wavl = None):
        """generate a synthetic SED from a list of observations
        filters     :   iterable of Filter() objects
        wavl        :   common wavelength grid to iterate onto
        """
        self.wavl = wavl
        self.transms = np.array([np.interp(wavl, filt.wavl, filt.transm)
                                  for filt in filters])
        self.zeropoints = np.array([filt.zeropoint for filt in filters])
        
    def __call__(self, flux, wavl = None, axis = 1):
        """integrate a spectrum through a filter
        """
        if wavl is None:
            return np.trapezoid(flux*self.transms*self.wavl, self.wavl, axis=axis) / np.trapezoid(self.transms*self.wavl, self.wavl, axis=axis)
        else:
            transms = np.array([np.interp(wavl, self.wavl, tr, left=0., right=0.) for tr in self.transms])
            return np.trapezoid(flux*transms*wavl, wavl, axis=axis) / np.trapezoid(transms*wavl, wavl, axis=axis)
        
def load_from_path(path, filterkws):
    data = np.load(path)
    return Filter(data[0], data[1], **filterkws)

def get_default_filters(filterkws = {}):
    dirname = os.path.dirname(os.path.abspath(__file__)) 
    bands = {
        # Gaia Photometry
        'Gaia_G' : load_from_path(os.path.join(dirname, 'bandpasses/Gaia_G.npy'), filterkws),
        'Gaia_BP' : load_from_path(os.path.join(dirname, 'bandpasses/Gaia_BP.npy'), filterkws),
        'Gaia_RP' : load_from_path(os.path.join(dirname, 'bandpasses/Gaia_RP.npy'), filterkws),
        # SDSS Photometry
        'SDSS_u' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_u.npy'), filterkws),
        'SDSS_g' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_g.npy'), filterkws),
        'SDSS_r' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_r.npy'), filterkws),
        'SDSS_i' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_i.npy'), filterkws),
        'SDSS_z' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_z.npy'), filterkws),
        # PanSTARRS DR1
        'PS1_g' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_g.npy'), filterkws),
        'PS1_r' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_r.npy'), filterkws),
        'PS1_i' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_i.npy'), filterkws),
        'PS1_z' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_z.npy'), filterkws),
        'PS1_y' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_y.npy'), filterkws),
        # J-PLUS F Filters
        'JPLUS_F0378' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_F0378.npy'), filterkws),
        'JPLUS_F0395' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_F0395.npy'), filterkws),
        'JPLUS_F0410' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_F0410.npy'), filterkws),
        'JPLUS_F0430' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_F0430.npy'), filterkws),   
        'JPLUS_F0515' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_F0515.npy'), filterkws),
        'JPLUS_F0660' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_F0660.npy'), filterkws),
        'JPLUS_F0861' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_F0861.npy'), filterkws),
        # J-PLUS J Filters
        'JPLUS_J0378' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_J0378.npy'), filterkws),
        'JPLUS_J0395' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_J0395.npy'), filterkws),
        'JPLUS_J0410' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_J0410.npy'), filterkws),
        'JPLUS_J0430' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_J0430.npy'), filterkws),
        'JPLUS_J0515' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_J0515.npy'), filterkws),
        'JPLUS_J0660' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_J0660.npy'), filterkws),
        'JPLUS_J0861' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_J0861.npy'), filterkws),
        # J-PLUS SDSS
        'JPLUS_uJava' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_uJava.npy'), filterkws),
        'JPLUS_gSDSS' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_gSDSS.npy'), filterkws),
        'JPLUS_rSDSS' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_rSDSS.npy'), filterkws),
        'JPLUS_iSDSS' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_iSDSS.npy'), filterkws),
        'JPLUS_zSDSS' : load_from_path(os.path.join(dirname, 'bandpasses/JPLUS_zSDSS.npy'), filterkws),
    }
    ## define some synthetic filters for comparison
    lam_min, lam_max = 3600, 9600
    n_boxes = 24 ; width = 300.0
    centers = np.linspace(lam_min + width/2, lam_max - width/2, n_boxes)
    lam = np.linspace(1000, 20000, 10000)
    for ii, c in enumerate(centers):
        bands[f"SYNTH_{ii+1}"] = Filter(lam, ((lam >= c - width/2) & (lam <= c + width/2)).astype(float))
    return bands
