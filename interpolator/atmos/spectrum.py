from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import glob, os, re

from dust_extinction.parameter_averages import G23
import astropy.units as u

from .sed import SED

dirname = os.path.dirname(os.path.abspath(__file__)) 
supported_models = {'1d_da_nlte': ('../data/1d_da_nlte/', 2, 'air'),
                            '1d_elm_da_lte': ('../data/1d_elm_da_lte/', 2, 'air'),
                            '3d_da_lte_noh2': ('../data/3d_da_lte_noh2/', 2, 'vac'),
                            '3d_da_lte_h2': ('../data/3d_da_lte_h2/', 2, 'vac'),
                            '3d_da_lte_old': ('../data/3d_da_lte_old/', 2, 'air')}

class WarwickPhotometry:
    def __init__(self, model_name, filters, units = 'flam', speckws = {'wavl_range' : (1000,30000)}):
        self.model_name = model_name
        self.units = units
        self.spectrum = WarwickSpectrum(self.model_name, self.units, **speckws)
        self.SED = SED(filters, self.spectrum.wavl)

    def make_cache(self, Rv = 3.1, minAV = 0.0001, maxAV = 0.5, nAV = 60):
        def cached_interp(teff, logg, av = None):
            if av is None:
                return interp_sansav((teff, logg))
            else:
                return interp((teff, logg, av))
        # make the info we need
        avs = np.linspace(minAV, maxAV, nAV)
        T, L, A = np.meshgrid(10**self.spectrum.unique_logteff, self.spectrum.unique_logg, avs, indexing='ij')
        # compute the grid without redenning
        grid_sansav = self.SED(self.spectrum(T[...,0], L[...,0])[...,None,:], axis=-1)
        interp_sansav =  RegularGridInterpolator(
            (T[:,0,0], L[0,:,0]), grid_sansav, method='linear',
            bounds_error=False, fill_value=None)
        # compute the grid with redenning
        ext_grid = np.array([G23(Rv=Rv).extinguish(1e4/(self.spectrum.wavl*u.micron), Av=av) for av in avs])
        grid = self.SED((self.spectrum(T, L) * ext_grid[None,None,:,:])[...,None,:], axis=-1)
        interp =  RegularGridInterpolator(
            (T[:,0,0], L[0,:,0], A[0,0,:]), grid, method='linear',
            bounds_error=False, fill_value=None)
        return cached_interp, (T[:,0,0], L[0,:,0], A[0,0,:], grid_sansav, grid)

    def __call__(self, teff, logg, av = None, Rv = 3.1):
        if av is None:
            return self.SED(self.spectrum(teff, logg))
        else:
            return self.SED(self.spectrum(teff, logg)*G23(Rv=Rv).extinguish(1e4/(self.spectrum.wavl*u.micron), Av=av))

class WarwickSpectrum:
    def __init__(self, model, units = 'flam', wavl_range = (3600, 9000)):
        # (path_to_files, n_free_parameters, wavlength_frame)
        assert model in list(supported_models.keys()), 'requested model not supported'
        # load in the model files
        self.path = os.path.join(dirname, supported_models[model][0])
        self.files = list(set(glob.glob(f"{self.path}/*")) - set(glob.glob(f"{self.path}/*.csv")))
        self.units = units
        self.modelname = model
        self.wavl_range = wavl_range
        # fetch the wavelength and flux grids
        self.nparams = supported_models[model][1]
        wavls, values, fluxes = [], [], []
        for file in self.files:
            wl, vals, fls = self.filehandler(file)
            wavls += wl
            values += vals
            fluxes += fls        
        self.values = np.array(values, dtype=float)
        wl_grid_length = list(set([len(wl) for wl in wavls]))
        # Handle multiple wavelength grids in the same model
        try:
            fluxes_np = np.array(fluxes, dtype=float)
            wavls = np.array(wavls, dtype=float)
        except ValueError:
            wavls, fluxes_np = self.interpolate(wavls, fluxes, max(wl_grid_length))
        mask = (self.wavl_range[0] < wavls[0]) & (wavls[0] < self.wavl_range[1])
        self.wavl, self.fluxes = wavls[0][mask], fluxes_np[:,mask]
        # convert to flam if that option is specified
        if self.units == 'flam':
            self.fnu_to_flam()

        if supported_models[model][2] == 'air':
            self.air2vac()
        self.build_interpolator()

    def fnu_to_flam(self):
        self.fluxes = 2.99792458e18 * self.fluxes / self.wavl**2

    def air2vac(self):
        _tl=1.e4/self.wavl
        self.wavl = (self.wavl*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))

    def interpolate(self, wavls, fluxes, length_to_interpolate):
        for i in range(len(wavls)):
            if len(wavls[i]) == length_to_interpolate:
                reference_grid = wavls[i]
                break
        for i in range(len(wavls)):
            if len(wavls[i]) != length_to_interpolate:
                fluxes[i] = np.interp(reference_grid, wavls[i], fluxes[i])
                wavls[i] = reference_grid
        return np.array(wavls), np.array(fluxes)

    def build_interpolator(self):
        self.unique_logteff = np.array(sorted(list(set(self.values[:,0]))))
        self.unique_logg = np.array(sorted(list(set(self.values[:,1]))))
        self.flux_grid = np.zeros((len(self.unique_logteff), 
                                len(self.unique_logg), 
                                len(self.wavl)))

        for i in range(len(self.unique_logteff)):
            for j in range(len(self.unique_logg)):
                target = [self.unique_logteff[i], self.unique_logg[j]]
                try:
                    indx = np.where((self.values == target).all(axis=1))[0][0]
                    self.flux_grid[i,j] = self.fluxes[indx]
                except IndexError:
                    self.flux_grid[i,j] += -999

        self.model_spec = RegularGridInterpolator((10**self.unique_logteff, self.unique_logg), self.flux_grid) 

    def filehandler(self, file):
        with open(file, 'r') as f:
            fdata = f.read()

        wavl = self.fetch_wavl(fdata)
        values, fluxes = self.fetch_spectra(fdata)
        dim_wavl = []
        for i in range(len(fluxes)):
            dim_wavl.append(wavl)
        return dim_wavl, values, fluxes

    def fetch_wavl(self, fdata):
        def get_linenum(npoints):
            first = 1
            last = (npoints // 10 + 1)
            if (npoints % 10) != 0:
                last += 1
            return first, last
        # figure out how many points are in the file
        lines = fdata.split('\n')
        npoints = int(lines[0])
        first, last = get_linenum(npoints)

        wavl = []
        for line in lines[first:last]:
            line = line.strip('\n').split()
            for num in line:
                wavl.append(float(num))
        return wavl

    def fetch_spectra(self, fdata):
        def idx_to_params(indx, first_n):
            string = lines[indx]
            regex = "[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
            params = re.findall(regex, string)[:first_n]
            params = [np.log10(float(num)) for num in params]
            return params
        lines = fdata.split('\n')
        npoints = int(lines[0])
        indx = [i for i, line in enumerate(lines) if 'Effective' in line]
        
        values, fluxes = [], []
        for n in range(len(indx)):
            first = indx[n]+1
            try:
                last = indx[n+1]
            except IndexError:
                last = len(lines)
            params = idx_to_params(indx[n], self.nparams)
            values.append(params)

            flux = []
            for line in lines[first:last]:
                line = line.strip('\n').split()
                for num in line:
                    flux.append(float(num))

            assert len(flux) == npoints, "Error reading spectrum: wrong number of points!"
            fluxes.append(flux)
        return values, fluxes
    
    def __call__(self, teff, logg):
        return self.model_spec((teff, logg))