import numpy as np
import os

class Filter:
    def __init__(self, wavl : np.array, transm : np.array):
        """define a filter
        """
        self.wavl = wavl
        self.transm = transm

    def __call__(self, flux, wavl = None):
        """integrate a spectrum through a filter
        """
        transm = np.interp(wavl, self.wavl, self.transm) if wavl is not None else self.transm
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
        
    def __call__(self, flux, wavl = None, axis = 1):
        """integrate a spectrum through a filter
        """
        if wavl == None:
            return np.trapezoid(flux*self.transms*self.wavl, self.wavl, axis=axis) / np.trapezoid(self.transms*self.wavl, self.wavl, axis=axis)
        else:
            transms = np.array([np.interp(wavl, self.wavl, tr) for tr in self.transms])
            return np.trapezoid(flux*transms*wavl, wavl, axis=axis) / np.trapezoid(transms*wavl, wavl, axis=axis)
        
def load_from_path(path):
    data = np.load(path)
    return Filter(data[0], data[1])

def get_default_filters():
    dirname = os.path.dirname(os.path.abspath(__file__)) 
    return {
        'Gaia_G' : load_from_path(os.path.join(dirname, 'bandpasses/Gaia_G.npy')),
        'Gaia_BP' : load_from_path(os.path.join(dirname, 'bandpasses/Gaia_BP.npy')),
        'Gaia_RP' : load_from_path(os.path.join(dirname, 'bandpasses/Gaia_RP.npy')),
        'SDSS_u' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_u.npy')),
        'SDSS_g' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_g.npy')),
        'SDSS_r' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_r.npy')),
        'SDSS_i' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_i.npy')),
        'SDSS_z' : load_from_path(os.path.join(dirname, 'bandpasses/SDSS_z.npy')),
        'PS1_g' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_g.npy')),
        'PS1_r' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_r.npy')),
        'PS1_i' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_i.npy')),
        'PS1_z' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_z.npy')),
        'PS1_y' : load_from_path(os.path.join(dirname, 'bandpasses/PS1_y.npy')),
    }