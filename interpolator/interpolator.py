from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RegularGridInterpolator
from scipy.interpolate import griddata, interp1d
from dataclasses import dataclass
import re
import pandas as pd
import numpy as np
import tqdm
import os

from astropy.table import Table, vstack
import glob
import pyphot

if __name__ != "__main__":
    from . import utils

lib = pyphot.get_library()

limits = {
    'CO_Hrich': ((4800, 79000), (8.84, 9.26)),
    'CO_Hdef': ((4500, 79000), (8.85, 9.28)),
    'ONe_Hrich': ((3750, 78800), (8.86, 9.30)),
    'ONe_Hdef': ((4250, 78800), (8.86, 9.31)),
}

dirname = os.path.dirname(os.path.abspath(__file__)) 
supported_models = {'1d_da_nlte': ('data/1d_da_nlte/', 2, 'air'),
                            '1d_elm_da_lte': ('data/1d_elm_da_lte/', 2, 'air'),
                            '3d_da_lte_noh2': ('data/3d_da_lte_noh2/', 2, 'vac'),
                            '3d_da_lte_h2': ('data/3d_da_lte_h2/', 2, 'vac'),
                            '3d_da_lte_old': ('data/3d_da_lte_old/', 2, 'air')}

class Interpolator:
    def __init__(self, interp_obj, teff_lims, logg_lims):
        self.interp_obj = interp_obj
        self.teff_lims = teff_lims
        self.logg_lims = logg_lims

def purge_cachetables(names : list = None):
    names = list(supported_models.keys()) if names == None else names
    assert np.all([name in supported_models.keys() for name in names])
    for name in names:
        cachetable = os.path.join(dirname, supported_models[name][0], 'cache_table.csv')
        if os.path.isfile(cachetable):
            os.remove(cachetable)

class WarwickPhotometry:
    """
    """
    def __init__(self, model, bands, precache=True, speckws = {}):
        self.model = model
        self.bands = bands # pyphot library objects
        self.precache = precache # use precaching?
        self.spectrum = WarwickSpectrum(model = self.model, with_cachetable = self.precache, **speckws)

        if not self.precache:
            self.teff_lims = (1500, 140000.0)
            self.logg_lims = (6.5, 9.49)
            # generate the interpolator 
            self.interp = lambda teff, logg: np.array([lib[band].get_flux(self.spectrum.wavl * pyphot.unit['angstrom'], self.spectrum.model_spec((teff, logg)) * pyphot.unit['erg/s/cm**2/angstrom'], axis = 1).to('erg/s/cm**2/angstrom').value for band in self.bands])
        else:
            self.teff_lims = (1500, 140000.0)
            self.logg_lims = (6.5, 9.49)
            table = self.spectrum.cachetable
            self.interp = MultiBandInterpolator(table, self.bands, self.teff_lims, self.logg_lims)

    def __call__(self, teff, logg):
        return self.interp(teff, logg)
    
class WarwickSpectrum:
    def __init__(self, model, units = 'flam', wavl_range = (3600, 9000), with_cachetable = False):
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

        if os.path.isfile(os.path.join(self.path, 'cache_table.csv')) and with_cachetable:
            self.cachetable = pd.read_csv(os.path.join(self.path, 'cache_table.csv'))
        elif with_cachetable:
            self.build_cachetable()

    def build_cachetable(self):
        print('Cachetable not found! Building...')
        filters = lib.content
        rowvals = np.nan*np.zeros((self.unique_logg.shape[0]*self.unique_logteff.shape[0], 2 + len(filters)))
        for i, logg in tqdm.tqdm(enumerate(self.unique_logg), total=self.unique_logg.shape[0]):
            for j, teff in enumerate(10**self.unique_logteff):
                idx = i*self.unique_logteff.shape[0] + j
                fluxes = np.array([lib[filt].get_flux(self.wavl*pyphot.unit['AA'], self(teff, logg)*pyphot.unit['erg/s/cm**2/AA']).to('erg/s/cm**2/AA').value for filt in filters])
                rowvals[idx,0] = teff
                rowvals[idx,1] = logg
                rowvals[idx,2:] = fluxes
        columns = ['teff', 'logg'] + filters
        self.cachetable = pd.DataFrame(rowvals, columns = columns)
        self.cachetable.to_csv(os.path.join(self.path, 'cache_table.csv'), index=False)

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
    
class LaPlataBase:
    def __init__(self, bands, layer =  None):        
        self.bands = bands
        self.layer = layer

        dirpath = os.path.dirname(os.path.realpath(__file__)) # identify the current directory
        path = ''
        
        if self.layer == 'Hrich':
            path = f'{dirpath}/data/laplata/allwd_Hrich.csv'
        elif self.layer == "Hdef":
            path = f'{dirpath}/data/laplata/allwd_Hdef.csv'

        self.table = Table.read(path)
        self.teff_lims = (5000, 79000)
        self.logg_lims = (7, 9.5)
        self.interp = MultiBandInterpolator(self.table, self.bands, self.teff_lims, self.logg_lims)

        self.mass_array, self.logg_array, self.age_cool_array, self.teff_array, self.Mbol_array = self.read_cooling_tracks()

    def read_cooling_tracks(self):
        dirpath = os.path.dirname(os.path.realpath(__file__))

        # initialize the array
        mass_array  = np.zeros(0)
        logg        = np.zeros(0)
        age_cool    = np.zeros(0)
        teff        = np.zeros(0)
        Mbol        = np.zeros(0)

        # load in values from each track
        Cool = self.table
        #Cool = Cool[::10] # (Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)
        mass_array  = np.concatenate(( mass_array, Cool['mWD'] ))
        logg        = np.concatenate(( logg, Cool['logg'] ))
        age_cool    = np.concatenate(( age_cool, (10**Cool['TpreWD(gyr)'])))
        teff     = np.concatenate(( teff, Cool['teff']))
        Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['log(L)'] ))
        del Cool

        select = ~np.isnan(mass_array + logg + age_cool + teff + Mbol)# * (age_cool > 1)
        return mass_array[select], logg[select], age_cool[select], teff[select], Mbol[select]
    
    def masstoradius(self, massarray, teffarray):
        radius_sun = 6.957e8
        mass_sun = 1.9884e30
        newton_G = 6.674e-11

        g_acc = (10**self.logg_array) / 100
        rsun = np.sqrt(self.mass_array * mass_sun * newton_G / g_acc) / radius_sun
        
        selected    = ~np.isnan(self.mass_array + self.teff_array + rsun)
        rsun_teff_to_m = LinearNDInterpolator((self.mass_array[selected], self.teff_array[selected]), rsun[selected])
        radius = rsun_teff_to_m(massarray, teffarray)
        return radius
    
    def radiustomass(self, radiusarray, teffarray):
        radius_sun = 6.957e8
        mass_sun = 1.9884e30
        newton_G = 6.674e-11

        g_acc = (10**self.logg_array) / 100
        rsun = np.sqrt(self.mass_array * mass_sun * newton_G / g_acc) / radius_sun
        
        selected    = ~np.isnan(self.mass_array + self.teff_array + rsun)
        rsun_teff_to_m = LinearNDInterpolator((rsun[selected], self.teff_array[selected]), self.mass_array[selected])
        mass = rsun_teff_to_m(radiusarray, teffarray)
        return mass
    
    def measurabletoradius(self, teff, logg):
        radius_sun = 6.957e8
        mass_sun = 1.9884e30
        newton_G = 6.674e-11

        g_acc = (10**self.logg_array) / 100
        rsun = np.sqrt(self.mass_array * mass_sun * newton_G / g_acc) / radius_sun
        
        selected    = ~np.isnan(self.mass_array + self.teff_array + rsun)
        rsun_teff_to_m = LinearNDInterpolator((self.teff_array[selected], self.logg_array[selected]), rsun[selected])
        radius = rsun_teff_to_m(teff, logg)
        return radius    

    def __call__(self, teff, logg):
        return self.interp(teff, logg)  
    
class LaPlataLowMass:
    def __init__(self, bands, core =  None):        
        self.bands = bands
        self.core = core

        dirpath = os.path.dirname(os.path.realpath(__file__)) # identify the current directory
        if self.core == 'He':
            path = f'{dirpath}/data/laplata/He_LowMass.csv'
            self.teff_lims = (3000, 20000)
            self.logg_lims = (3, 7.5)
        elif self.core == 'CO':
            path = f'{dirpath}/data/laplata/CO_LowMass.csv'
            self.teff_lims = (3000, 20000)
            self.logg_lims = (3, 7.5)

        self.table = Table.read(path)
        self.interp = MultiBandInterpolator(self.table, self.bands, self.teff_lims, self.logg_lims)
        self.mass_array, self.logg_array, self.age_cool_array, self.teff_array, self.Mbol_array = self.read_cooling_tracks()

    def read_cooling_tracks(self):
        dirpath = os.path.dirname(os.path.realpath(__file__))

        # initialize the array
        mass_array  = np.zeros(0)
        logg        = np.zeros(0)
        age_cool    = np.zeros(0)
        teff        = np.zeros(0)
        Mbol        = np.zeros(0)

        # load in values from each track
        Cool = self.table
        #Cool = Cool[::10] # (Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)
        mass_array  = np.concatenate(( mass_array, Cool['mass'] ))
        logg        = np.concatenate(( logg, Cool['logg'] ))
        age_cool    = np.concatenate(( age_cool, (Cool['age'])))
        teff     = np.concatenate(( teff, Cool['teff']))
        Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['logL'] ))
        del Cool

        select = ~np.isnan(mass_array + logg + age_cool + teff + Mbol)# * (age_cool > 1)
        return mass_array[select], logg[select], age_cool[select], teff[select], Mbol[select]
    
    def massradius(self, massarray, teffarray):
        radius_sun = 6.957e8
        mass_sun = 1.9884e30
        newton_G = 6.674e-11

        g_acc = (10**self.logg_array) / 100
        rsun = np.sqrt(self.mass_array * mass_sun * newton_G / g_acc) / radius_sun
        
        selected    = ~np.isnan(self.mass_array + self.teff_array + rsun)
        rsun_teff_to_m = LinearNDInterpolator((self.mass_array[selected], self.teff_array[selected]), rsun[selected])
        radius = rsun_teff_to_m(massarray, teffarray)
        return radius
    
    def radiustomass(self, radiusarray, teffarray):
        radius_sun = 6.957e8
        mass_sun = 1.9884e30
        newton_G = 6.674e-11

        g_acc = (10**self.logg_array) / 100
        rsun = np.sqrt(self.mass_array * mass_sun * newton_G / g_acc) / radius_sun
        
        selected    = ~np.isnan(self.mass_array + self.teff_array + rsun)
        rsun_teff_to_m = LinearNDInterpolator((rsun[selected], self.teff_array[selected]), self.mass_array[selected])
        mass = rsun_teff_to_m(radiusarray, teffarray)
        return mass

    def measurabletoradius(self, teff, logg):
        radius_sun = 6.957e8
        mass_sun = 1.9884e30
        newton_G = 6.674e-11

        g_acc = (10**self.logg_array) / 100
        rsun = np.sqrt(self.mass_array * mass_sun * newton_G / g_acc) / radius_sun
        
        selected    = ~np.isnan(self.mass_array + self.teff_array + rsun)
        rsun_teff_to_m = LinearNDInterpolator((self.teff_array[selected], self.logg_array[selected]), rsun[selected])
        radius = rsun_teff_to_m(teff, logg)
        return radius    
    
    def __call__(self, teff, logg):
        return self.interp(teff, logg) 


class LaPlataUltramassive:
    def __init__(self, bands, core = None, layer =  None):        
        self.bands = bands
        self.core, self.layer = core, layer

        dirpath = os.path.dirname(os.path.realpath(__file__)) # identify the current directory
        model = f'{self.core}_{self.layer}'
        path = f'{dirpath}/data/laplata/{model}_Massive.csv' if (self.core is not None) else f'{dirpath}/data/laplata/allwd_Hrich.csv'
        self.table = Table.read(path)
    
        self.teff_lims = limits[model][0] if (self.core is not None) else (5000, 79000)
        self.logg_lims = limits[model][1] if (self.core is not None) else (7, 9.5)
        self.interp = MultiBandInterpolator(self.table, self.bands, self.teff_lims, self.logg_lims)

        self.mass_array, self.logg_array, self.age_cool_array, self.teff_array, self.Mbol_array = self.read_cooling_tracks()

    def read_cooling_tracks(self):
        dirpath = os.path.dirname(os.path.realpath(__file__))

        # initialize the array
        mass_array  = np.zeros(0)
        logg        = np.zeros(0)
        age_cool    = np.zeros(0)
        teff        = np.zeros(0)
        Mbol        = np.zeros(0)

        if self.core == 'ONe':
            masses = ['110','116','122','129']
        elif self.core == 'CO':
            masses = ['110','116','123','129']

        # load in values from each track
        for mass in masses:
            if self.core == 'CO':
                Cool = Table.read(dirpath+'/data/laplata/high_mass/'+self.core+'_'+mass+'_'+self.layer+'_0_02.dat', format='ascii') 
                Cool = Cool[::10] # (Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)
                mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
                logg        = np.concatenate(( logg, Cool['logg(CGS)'] ))
                age_cool    = np.concatenate(( age_cool, Cool['tcool(gyr)']))
                teff        = np.concatenate(( teff, Cool['Teff'] ))
                Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['log(L/Lsun)'] ))
                del Cool
            elif self.core == 'ONe':
                Cool = Table.read(dirpath+'/data/laplata/high_mass/'+self.core+'_'+mass+'_'+self.layer+'_0_02.dat', format='ascii') 
                Cool = Cool[::10] # (Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)
                mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
                logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
                age_cool    = np.concatenate(( age_cool, (10**Cool['Log(edad/Myr)'] -
                                                    10**Cool['Log(edad/Myr)'][0]) * 1e6 ))
                teff     = np.concatenate(( teff, 10**Cool['LOG(TEFF)']))
                Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['LOG(L)'] ))
                del Cool

        if self.core == "ONe":
            age_cool *= 1e-3

        select = ~np.isnan(mass_array + logg + age_cool + teff + Mbol) * (age_cool > 1)
        return mass_array[select], logg[select], age_cool[select], teff[select], Mbol[select]

    def masstoradius(self, massarray, teffarray):
        radius_sun = 6.957e8
        mass_sun = 1.9884e30
        newton_G = 6.674e-11

        g_acc = (10**self.logg_array) / 100
        rsun = np.sqrt(self.mass_array * mass_sun * newton_G / g_acc) / radius_sun
        print(rsun)
        
        selected    = ~np.isnan(self.mass_array + self.teff_array + rsun)
        msun_teff_to_r = LinearNDInterpolator((self.mass_array[selected], self.teff_array[selected]), rsun[selected])
        radius = msun_teff_to_r(massarray, teffarray)
        return radius
    
    def radiustomass(self, radiusarray, teffarray):
        radius_sun = 6.957e8
        mass_sun = 1.9884e30
        newton_G = 6.674e-11

        g_acc = (10**self.logg_array) / 100
        rsun = np.sqrt(self.mass_array * mass_sun * newton_G / g_acc) / radius_sun
        
        selected    = ~np.isnan(self.mass_array + self.teff_array + rsun)
        rsun_teff_to_m = LinearNDInterpolator((rsun[selected], self.teff_array[selected]), self.mass_array[selected])
        mass = rsun_teff_to_m(radiusarray, teffarray)
        return mass

    def measurabletoradius(self, teff, logg):
        radius_sun = 6.957e8
        mass_sun = 1.9884e30
        newton_G = 6.674e-11

        g_acc = (10**self.logg_array) / 100
        rsun = np.sqrt(self.mass_array * mass_sun * newton_G / g_acc) / radius_sun
        
        selected    = ~np.isnan(self.mass_array + self.teff_array + rsun)
        rsun_teff_to_m = LinearNDInterpolator((self.teff_array[selected], self.logg_array[selected]), rsun[selected])
        radius = rsun_teff_to_m(teff, logg)
        return radius    

    def __call__(self, teff, logg):
        return self.interp(teff, logg)    
    

class SingleBandInterpolator:
    def __init__(self, table, band, teff_lims, logg_lims):
        self.table = table
        self.band = band
        self.teff_lims = teff_lims
        self.logg_lims = logg_lims

        self.eval = self.build_interpolator()

    def __call__(self, teff, logg):
        return self.eval(teff, logg)

    def build_interpolator(self):
        def interpolate_2d(x, y, z, method):
            if method == 'linear':
                interpolator = LinearNDInterpolator
            elif method == 'cubic':
                interpolator = CloughTocher2DInterpolator
            return interpolator((x, y), z, rescale=True)
            #return interp2d(x, y, z, kind=method)

        def interp(x, y, z):
            grid_z      = griddata(np.array((x, y)).T, z, (grid_x, grid_y), method='linear')
            z_func      = interpolate_2d(x, y, z, 'linear')
            return z_func

        logteff_logg_grid=(self.teff_lims[0], self.teff_lims[1], 1000, self.logg_lims[0], self.logg_lims[1], 0.01)
        grid_x, grid_y = np.mgrid[logteff_logg_grid[0]:logteff_logg_grid[1]:logteff_logg_grid[2],
                                    logteff_logg_grid[3]:logteff_logg_grid[4]:logteff_logg_grid[5]]

        band_func = interp(self.table['teff'], self.table['logg'], self.table[self.band])

        photometry = lambda teff, logg: float(band_func(teff, logg))
        return photometry

class MultiBandInterpolator:
    def __init__(self, table, bands, teff_lims, logg_lims):
        self.table = table
        self.bands = bands
        self.teff_lims = teff_lims
        self.logg_lims = logg_lims

        self.interpolator = [SingleBandInterpolator(self.table, band, self.teff_lims, self.logg_lims) for band in self.bands]

    def __call__(self, teff, logg):
        return np.array([interp(teff, logg) for interp in self.interpolator])

    
