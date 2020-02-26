import piscola
from .filter_integration import *
from .gaussian_process import *
from .spline import *
from .extinction_correction import *
from .mangling import *
from .util import *

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from peakutils import peak
import pandas as pd
import numpy as np
import pickle
import math
import glob
import os

### Initialisation functions ###

def initialise(file_name):
    """Initialise the 'sn' object.

    The object is initialise with all the necessary information like filters, fluxes, etc.

    Parameters
    ----------
    file_name : str
        Name of the SN or SN file.

    Returns
    -------
    New 'sn' object.

    """

    name, z, ra, dec = pd.read_csv(file_name, delim_whitespace=True, nrows=1).iloc[0].values
    sn_file = pd.read_csv(file_name, delim_whitespace=True, skiprows=2)
    sn_file.columns = sn_file.columns.str.lower()

    # call sn object
    sn_obj = sn(name, z=z, ra=ra, dec=dec)
    sn_obj.bands = [band for band in list(sn_file['band'].unique()) if len(sn_file[sn_file['band']==band]['flux']) >= 3]
    sn_obj.call_filters()

    # order filters by wavelength
    eff_waves = [sn_obj.filters[band]['eff_wave'] for band in sn_obj.bands]
    sorted_idx = sorted(range(len(eff_waves)), key=lambda k: eff_waves[k])
    sn_obj.bands = [sn_obj.bands[x] for x in sorted_idx]

    for band in sn_obj.bands:
        A_lambda = extinction_filter(sn_obj.filters[band]['wave'], sn_obj.filters[band]['transmission'], ra, dec)
        sn_obj.filters[band]['A_lambda'] = A_lambda

    # add data of every band
    for band in sn_obj.bands:
        band_info = sn_file[sn_file['band']==band]
        sn_obj.data[band] = {'mjd':band_info['mjd'].values,
                             'flux':band_info['flux'].values,
                             'flux_err':band_info['flux_err'].values,
                             'zp':float(band_info['zp'].unique()[0]),
                             'mag_sys':band_info['mag_sys'].unique()[0],
                            }

    sn_obj.set_sed_template()

    return sn_obj


def sn_file(str, directory='data/'):
    """Loads a supernova from a file.

    Parameters
    ----------
    directory : str, default 'data/'
        Directory where to look for the SN data files.

    """

    if os.path.isfile(directory+str):
        return initialise(directory+str)

    elif os.path.isfile(directory+str+'.dat'):
        return initialise(directory+str+'.dat')

    elif os.path.isfile(str):
        return initialise(str)

    else:
        raise ValueError(f'{str} was not a valid SN name or file.')

def load_sn(name, path=None):
    """Loads a 'sn' oject that was previously saved as a pickle file.

    Parameters
    ----------
    name : str
        Name of the SN object.

    Returns
    -------
    'sn' object previously saved as a pickle file.

    """

    if path is None:
        with open(name + '.pisco', 'rb') as file:
            return pickle.load(file)
    else:
        with open(path + name + '.pisco', 'rb') as file:
            return pickle.load(file)

#################################

class sn(object):
    """Supernova class for representing a supernova."""

    def __init__(self, name, z=0, ra=None, dec=None):
        self.name = name
        self.z = z # redshift
        self.ra = ra # coordinates in degrees
        self.dec = dec

        if self.ra is None or self.dec is None:
            print('Warning, ra and/or dec not specified')
            self.ra , self.dec = 0, 0

        self.__dict__['data'] = {}  # data for each band
        self.__dict__['sed'] = {}  # sed info
        self.__dict__['sed_fits'] = {}  # sed 2D-GP fits
        self.__dict__['filters'] = {}  # filter info for each band
        self.__dict__['lc_fits'] = {}  # gp fitted data
        self.__dict__['lc_interp'] = {}  # interpolated data to be used for light curves correction
        self.__dict__['lc_correct'] = {}  # corrected light curves
        self.__dict__['lc_final_fits'] = {}  # interpolated corrected light curves
        self.__dict__['lc_parameters'] = {}  # final SN light-curves parameters
        self.__dict__['sed_results'] = {}  # final SED for every phase if successful
        self.__dict__['mangling_results'] = {}  # mangling results for every phase if successful
        self.bands = None
        self.pivot_band = None
        self.tmax = None
        self.phase = 0 # Initial value for approximate effefctive wavelength calculation
        self.normalization = None  # NOT USED ANYMORE(?)
        self.test = None  # to test stuff - not part of the release


    def __repr__(self):
        return f'name = {self.name}, z = {self.z:.5}, ra = {self.ra}, dec = {self.dec}'

    def __getattr__(self, attribute):
        if attribute=='name':
            return self.name
        if attribute=='z':
            return self.z
        if attribute=='ra':
            return self.ra
        if attribute=='dec':
            return self.dec
        if 'data' in self.__dict__:
            if attribute in self.data:
                return(self.data[attribute])
        else:
            return f'Attribute {attribute} is not defined.'

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def save(self, name=None, path=None):
        """Saves a SN object into a pickle file"""

        if name is None:
            name = self.name

        if path is None:
            with open(name + '.pisco', 'wb') as pfile:
                pickle.dump(self, pfile, pickle.HIGHEST_PROTOCOL)
        else:
            with open(path + name + '.pisco', 'wb') as pfile:
                pickle.dump(self, pfile, pickle.HIGHEST_PROTOCOL)


    ############################################################################
    ################################ Filters ###################################
    ############################################################################

    def call_filters(self):
        """Obtains the filters's transmission function for the observed bands and the Bessell bands."""

        path = piscola.__path__[0]
        vega_wave, vega_flux = np.loadtxt(path + '/templates/alpha_lyr_stis_005.dat').T

        # add filters of the observed bands
        for band in self.bands:
            file = f'{band}.dat'

            for root, dirs, files in os.walk(path + '/filters/'):
                if file in files:
                    wave0, transmission0 = np.loadtxt(os.path.join(root, file)).T
                    # linearly interpolate filters
                    wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
                    transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
                    # remove long tails of zero values on both edges
                    imin, imax = trim_filters(transmission)
                    wave, transmission = wave[imin:imax], transmission[imin:imax]

                    response_type = 'photon'
                    self.filters[band] = {'wave':wave,
                                          'transmission':transmission,
                                          'eff_wave':calc_eff_wave(vega_wave, vega_flux, wave,
                                                                   transmission, response_type=response_type),
                                          'pivot_wave':calc_pivot_wave(wave, transmission,
                                                                       response_type=response_type),
                                          'response_type':response_type}

        # add Bessell filters
        file_paths = [file for file in glob.glob(path + '/filters/Bessell/*.dat')]

        for file_path in file_paths:
            band = os.path.basename(file_path).split('.')[0]
            wave0, transmission0 = np.loadtxt(file_path).T
            # linearly interpolate filters
            wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
            transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
            # remove long tails of zero values on both edges
            imin, imax = trim_filters(transmission)
            wave, transmission = wave[imin:imax], transmission[imin:imax]

            response_type = 'energy'
            self.filters[band] = {'wave':wave,
                                  'transmission':transmission,
                                  'eff_wave':calc_eff_wave(vega_wave, vega_flux, wave, transmission,
                                                           response_type=response_type),
                                  'pivot_wave':calc_pivot_wave(wave, transmission, response_type=response_type),
                                  'response_type':response_type}


    def add_filters(self, filter_list, response_type='photon'):
        """Add choosen filters. You can add a complete directory with filters in it or add filters given in a list.

        Parameters
        ----------
        filter_list : list
            List of bands.
        response_type : str, default 'photon'
            Response type of the filter. The only options are: 'photon' and 'energy'.
            Only the Bessell filters use energy response type.

        """

        path = piscola.__path__[0]
        vega_wave, vega_flux = np.loadtxt(path + '/templates/alpha_lyr_stis_005.dat').T

        if isinstance(filter_list, str) and os.path.isdir(f'{path}/filters/{filter_list}'):
            # add directory
            path = piscola.__path__[0]
            path = f'{path}/filters/{filter_list}'
            for file in os.listdir(path):
                if file[-4:]=='.dat':
                    band = file.split('.')[0]
                    wave0, transmission0 = np.loadtxt(os.path.join(path, file)).T
                    # linearly interpolate filters
                    wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
                    transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
                    # remove long tails of zero values on both edges
                    imin, imax = trim_filters(transmission)
                    wave, transmission = wave[imin:imax], transmission[imin:imax]

                    self.filters[band] = {'wave':wave,
                                          'transmission':transmission,
                                          'eff_wave':calc_eff_wave(vega_wave, vega_flux, wave, transmission,
                                                                   response_type=response_type),
                                          'pivot_wave':calc_pivot_wave(wave, transmission,
                                                                       response_type=response_type),
                                          'response_type':response_type}

        else:
            # add filters in list
            for band in filter_list:
                file = f'{band}.dat'

                for root, dirs, files in os.walk(path + '/filters/'):
                    if file in files:
                        wave0, transmission0 = np.loadtxt(os.path.join(root, file)).T
                        # linearly interpolate filters
                        wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
                        transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
                        # remove long tails of zero values on both edges
                        imin, imax = trim_filters(transmission)
                        wave, transmission = wave[imin:imax], transmission[imin:imax]

                        self.filters[band] = {'wave':wave,
                                              'transmission':transmission,
                                              'eff_wave':calc_eff_wave(vega_wave, vega_flux, wave, transmission,
                                                                       response_type=response_type),
                                              'pivot_wave':calc_pivot_wave(wave, transmission,
                                                                           response_type=response_type),
                                              'response_type':response_type}


    def plot_filters(self, filter_list=None, save=False):
        """Plot the filters' transmission functions.

        Parameters
        ----------
        filter_list : list, default 'None'
            List of bands.
        save : bool, default 'False'
            If true, saves the plot into a file.

        """

        if filter_list is None:
            filter_list = self.bands

        f, ax = plt.subplots(figsize=(8,6))
        for band in filter_list:
            norm = self.filters[band]['transmission'].max()
            ax.plot(self.filters[band]['wave'], self.filters[band]['transmission']/norm, label=band)

        ax.set_xlabel(r'wavelength ($\AA$)', fontsize=18, family='serif')
        ax.set_ylabel('normalized response', fontsize=18, family='serif')
        ax.set_title(r'Filters response functions', fontsize=18, family='serif')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.minorticks_on()
        ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
        ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

        if save:
            f.tight_layout()
            plt.savefig('plots/filters.png')

        plt.show()


    def calc_pivot(self, band_list=None):
        """Calculates the observed band closest to Bessell-B band.

        The pivot band will be the band on which the SED phases will be
        based on during the first cycle of fits.

        Parameters
        ----------
        filter_list : list, default 'None'
            List of bands.

        """

        BessellB_eff_wave = self.filters['Bessell_B']['eff_wave']  # effective wavelength in Angstroms

        if band_list is None:
            band_list = self.bands

        bands_eff_wave =  np.asarray([self.filters[band]['eff_wave']/(1+self.z) for band in band_list])
        idx = (np.abs(BessellB_eff_wave - bands_eff_wave)).argmin()
        self.pivot_band = band_list[idx]


    def delete_bands(self, bands_list, verbose=False):
        """Delete chosen bands together with the data in it.

        Parameters
        ----------
        bands : list
            List of bands.
        verbose : bool, default 'False'
            If 'True', a warning is given when a band from 'bands' was found within the SN bands.

        """

        for band in bands_list:
            if band in self.bands:
                self.data.pop(band, None)
                self.filters.pop(band, None)
                self.bands.remove(band)
            else:
                if verbose:
                    print(f'Warning, {band} not found!')


    ############################################################################
    ############################### SED template ###############################
    ############################################################################

    def print_sed_templates(self):
        """Prints all the available SED templates in the 'templates' directory"""

        path = piscola.__path__[0]
        print('The list of available SED templates are:', [name for name in os.listdir(path + "/templates/")
                                                           if os.path.isdir(f"{path}/templates/{name}")])


    def set_sed_template(self, template='conley09f'):
        """Sets the SED templates that are going to be used for the mangling.

        Parameters
        ----------
        template : str, default 'conley09f'
            Template name.

        """

        path = piscola.__path__[0]
        file = f'{path}/templates/{template}/snflux_1av2.dat'  # v2 is the interpolated version to 0.5 day steps
        sed_df = pd.read_csv(file, delim_whitespace=True, names=['phase', 'wave', 'flux'])
        self.sed['info'] = sed_df
        self.sed['name'] = template
        '''
        self.sed_lcs = {band:[] for band in self.bands}
        sed_df = sed_df[(-20 <= sed_df.phase) & (sed_df.phase <= 40)]  # for faster computation

        for phase in sed_df.phase.unique():
            phase_df = sed_df[sed_df.phase==phase]
            for band in self.bands:
                band_flux = run_filter(phase_df.wave.values, phase_df.flux.values, self.filters[band]['wave'],
                                       self.filters[band]['transmission'], self.filters[band]['response_type'])
                self.sed_lcs[band].append(band_flux)

        for band in self.bands:
            self.sed_lcs[band] = np.array(self.sed_lcs[band])

        self.sed_lcs['phase'] = sed_df.phase.unique()
        '''


    def set_sed_epoch(self, set_eff_wave=True):
        """Sets the SED phase given the current value of 'self.phase'.

        The chosen template is used to do all the corrections. The SED is immediately moved to the SN frame.

        Parameters
        ----------
        set_eff_wave : bool, default 'True'
            If True set the effective wavelengths of the filters given the SED at the current phase.

        """

        sed_data = self.sed['info'][self.sed['info'].phase == self.phase]
        self.sed['wave'], self.sed['flux'] = sed_data.wave.values*(1+self.z), sed_data.flux.values/(1+self.z)

        # These fluxes are used in the mangling process to compared to mangled ones with these.
        for band in self.bands:
            flux = run_filter(self.sed['wave'], self.sed['flux'], self.filters[band]['wave'],
                              self.filters[band]['transmission'], self.filters[band]['response_type'])
            self.sed[band] = {'flux': flux}

        # add filter's effective wavelength, which depends on the SED and the phase.
        if set_eff_wave:
            self.set_eff_wave()


    def set_eff_wave(self):
        """Sets the effective wavelength of each band using the current state of the SED."""

        for band in self.filters.keys():
            self.filters[band]['eff_wave'] = calc_eff_wave(self.sed['wave'], self.sed['flux'],
                                                           self.filters[band]['wave'], self.filters[band]['transmission'],
                                                           self.filters[band]['response_type'])

    ############################################################################
    ########################### Light Curves Data ##############################
    ############################################################################

    def mask_data(self, band_list=None, mask_phase=True, min_phase=-20, max_phase=40, mask_snr=False, snr=3):
       """Mask the data with the given S/N ratio and/or within the given range of days respect to maximum in B band.

       NOTE: Bands with less than 3 data points after mask is applied will be deleted.

       Parameters
       ----------
       band_list : list, default 'None'
       mask_phase : bool, default 'False'
           If 'True', keeps the flux values within the given phase range set by 'min_phase' and 'max_phase'.
            An initial estimation of the peak is needed first (can be set manually).
       min_phase : float, default '-20'
           Minimum phase threshold applied to mask data.
       max_phase : float, default '40'
           Maximum phase threshold applied to mask data.
           List of bands to mask. If 'None', the mask is applied to all bands in self.bands
       mask_snr : bool, default 'True'
           If 'True', keeps the flux values with S/N ratio greater or equal to the threshold 'snr'.
       snr : float, default '3'
           S/N ratio threshold applied to mask data.

       """

       if band_list is None:
           band_list = self.bands

       bands2delete = []

       if mask_phase:
           assert self.tmax, 'An initial estimation of the peak is needed first!'

           for band in band_list:
               mask = np.where((self.data[band]['mjd'] - self.tmax >= min_phase*(1+self.z)) &
                               (self.data[band]['mjd'] - self.tmax <= max_phase*(1+self.z))
                              )
               self.data[band]['mjd'] = self.data[band]['mjd'][mask]
               self.data[band]['flux'] = self.data[band]['flux'][mask]
               self.data[band]['flux_err'] = self.data[band]['flux_err'][mask]

               if len(self.data[band]['flux']) <= 2:
                   bands2delete.append(band)

       if mask_snr:
           for band in band_list:
               mask = np.where(np.abs(self.data[band]['flux']/self.data[band]['flux_err']) >= snr)
               self.data[band]['mjd'] = self.data[band]['mjd'][mask]
               self.data[band]['flux'] = self.data[band]['flux'][mask]
               self.data[band]['flux_err'] = self.data[band]['flux_err'][mask]

               if len(self.data[band]['flux']) <= 2:
                   bands2delete.append(band)

       self.delete_bands(bands2delete)  # delete bands with less than 3 data points after applying mask


    def plot_data(self, band_list=None, plot_type='mag', save=False, fig_name=None):
       """Plot the SN light curves.

       Negative fluxes are masked out if magnitudes are plotted.

       Parameters
       ----------
       band_list : list, default 'None'
           List of filters to plot. If 'None', band list is set to 'self.bands'.
       plot_type : str, default 'mag'
           Type of value plotted: either 'mag' or 'flux'.
       save : bool, default 'False'
           If true, saves the plot into a file.
       fig_name : str, default 'None'
           Name of the saved plot. If 'None' is used the name of the file will be ''{self.name}_sed{self.phase}.png'.
           Only works if 'save' is set to 'True'.

       """

       assert (plot_type=='mag' or plot_type=='flux'), f'"{plot_type}" is not a valid plot type.'
       new_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]

       if band_list is None:
           band_list = self.bands

       exp = np.round(np.log10(self.data[band_list[0]]['flux'].max()), 0)
       y_norm = 10**exp

       # to set plot limits
       if plot_type=='flux':
           plot_lim_vals = [[self.data[band]['flux'].min()/y_norm, self.data[band]['flux'].max()/y_norm] for band in self.bands]
           plot_lim_vals = np.ndarray.flatten(np.asarray(plot_lim_vals))
           ymin_lim = np.r_[plot_lim_vals, 0.0].min()*0.9
           if ymin_lim < 0.0:
               ymin_lim *= 1.1/0.9
           ymax_lim = plot_lim_vals.max()*1.1
       elif plot_type=='mag':
           plot_lim_vals = [[np.nanmin(-2.5*np.log10(self.data[band]['flux']) + self.data[band]['zp']),
                             np.nanmax(-2.5*np.log10(self.data[band]['flux']) + self.data[band]['zp'])] for band in self.bands]
           plot_lim_vals = np.ndarray.flatten(np.asarray(plot_lim_vals))
           ymin_lim = np.nanmin(plot_lim_vals)*0.98
           ymax_lim = np.nanmax(plot_lim_vals)*1.02

       f, ax = plt.subplots(figsize=(8,6))
       for i, band in enumerate(band_list):
           if plot_type=='flux':
               time, flux, err = np.copy(self.data[band]['mjd']), np.copy(self.data[band]['flux']), np.copy(self.data[band]['flux_err'])
               flux, err = flux/y_norm, err/y_norm
               ax.errorbar(time, flux, err, fmt='o', capsize=3, label=band, color=new_palette[i])
               ylabel = r'Flux [10$^{%.0f}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'%exp
           elif plot_type=='mag':
               ylabel = 'Apparent Magnitude'
               mask = np.where(self.data[band]['flux'] > 0)
               mjd = self.data[band]['mjd'][mask]
               mag = -2.5*np.log10(self.data[band]['flux'][mask]) + self.data[band]['zp']
               err = np.abs(2.5*self.data[band]['flux_err'][mask]/(self.data[band]['flux'][mask]*np.log(10)))

               ax.errorbar(mjd, mag, err, fmt='o', capsize=3, label=band, color=new_palette[i])

       ax.set_ylabel(ylabel, fontsize=16, family='serif')
       ax.set_xlabel('Modified Julian Date', fontsize=16, family='serif')
       ax.set_title(f'{self.name} (z = {self.z:.5})', fontsize=18, family='serif')
       ax.minorticks_on()
       ax.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=16)
       ax.tick_params(which='minor', length=4, width=1, direction='in', top=True, right=True, labelsize=16)
       ax.legend(fontsize=13)
       ax.set_ylim(ymin_lim, ymax_lim)

       if plot_type=='mag':
           plt.gca().invert_yaxis()

       if save:
           if fig_name is None:
               fig_name = f'{self.name}_lcs.png'
           f.tight_layout()
           plt.savefig(f'plots/{fig_name}')

       plt.show()


    def normalize_data(self):
       """Normalize the fluxes and zero-points.

       Fluxes are converted to physical units and the magnitude system is changed to 'Vega'.
       """

       for band in self.bands:
           mag_sys = self.data[band]['mag_sys']

           zp = calc_zp(self.filters[band]['wave'], self.filters[band]['transmission'],
                        self.filters[band]['response_type'], mag_sys, band)
           zp_vega = calc_zp(self.filters[band]['wave'], self.filters[band]['transmission'],
                        self.filters[band]['response_type'], 'vega')

           self.data[band]['flux'] = self.data[band]['flux']*10**(-0.4*(self.data[band]['zp'] - zp))
           self.data[band]['flux_err'] = self.data[band]['flux_err']*10**(-0.4*(self.data[band]['zp'] - zp))
           self.data[band]['zp'] = zp_vega
           self.data[band]['mag_sys'] = 'VEGA'

    ############################################################################
    ############################ Light Curves Fits #############################
    ############################################################################

    def fit_lcs(self, kernel='matern52'):
        """Fits the data for each band using gaussian process

        The fits are done independently for each band. The initial B-band peak time is estimated with
        the pivot band as long as a peak can me calculated, having a derivative equal to zero.

        Parameters
        ----------
        kernel : str, default 'matern52'
            Kernel to be used with gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.

        """

        ########## gp fit ############
        flux_array = np.hstack(np.array([self.data[band]['flux'] for band in self.bands]))
        flux_err_array = np.hstack(np.array([self.data[band]['flux_err'] for band in self.bands]))

        time_array = np.hstack(np.array([self.data[band]['mjd'] for band in self.bands]))
        wave_array = np.hstack(np.array([[self.filters[band]['eff_wave']]*len(self.data[band]['mjd']) for band in self.bands]))

        bands_waves = np.hstack(np.array([self.filters[band]['wave'] for band in self.bands]))
        bands_edges = np.array([bands_waves.min(), bands_waves.max()])

        timeXwave, mu, std = fit_2dgp(time_array, wave_array, flux_array, flux_err_array, kernel=kernel, x2_edges=bands_edges)
        self.lc_fits['timeXwave'], self.lc_fits['mu'], self.lc_fits['std'] = timeXwave, mu, std

        ########## check peak ############
        wave_ind = np.argmin(np.abs(self.filters['Bessell_B']['eff_wave']*(1+self.z) - timeXwave.T[1]))
        eff_wave = timeXwave.T[1][wave_ind]  # closest wavelength from the gp grid to the effective_wavelength*(1+z) of Bessell_B
        inds = [i for i, txw_tuplet in enumerate(timeXwave) if txw_tuplet[1]==eff_wave]

        time, flux, err = timeXwave.T[0][inds], mu[inds], std[inds]

        try:
            peakidxs = peak.indexes(flux, thres=.3, min_dist=5//(time[1]-time[0]))
            # pick the index of the first peak in case some band has 2 peaks (like IR bands)
            idx_max = np.asarray([idx for idx in peakidxs if all(flux[:idx]<flux[idx])]).min()
            self.tmax = np.round(time[idx_max], 2)

            phaseXwave = np.copy(timeXwave)
            phaseXwave.T[0] = (phaseXwave.T[0] - self.tmax)/(1 + self.z)
            self.lc_fits['phaseXwave'] = phaseXwave
        except:
            raise ValueError(f'Unable to obtain an accurate B-band peak for {self.name}!')

        ########## LCs interpolation ##########
        for band in self.bands:

            wave_ind = np.argmin(np.abs(self.filters[band]['eff_wave'] - timeXwave.T[1]))
            eff_wave = timeXwave.T[1][wave_ind]  # closest wavelength from the gp grid to the effective wavelength of the band
            inds = [i for i, txw_tuplet in enumerate(timeXwave) if txw_tuplet[1]==eff_wave]

            time, phase, flux, err = timeXwave.T[0][inds], phaseXwave.T[0][inds], mu[inds], std[inds]
            self.lc_fits[band] = {'mjd':time, 'phase':phase, 'flux':flux, 'std':err}


    def plot_fits(self, plot_together=True, plot_type='mag', save=False, fig_name=None):
        """Plots the light-curves fits results.

        Plots the observed data for each band together with the gaussian process fits. The initial B-band
        peak estimation is plotted. The final B-band peak estimation after light-curves corrections is
        also potted if corrections have been applied.

        Parameters
        ----------
        plot_together : bool, default 'True'
            If 'True', plots the bands together in one plot. Otherwise, each band is plotted separately.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be ''{self.name}_sed{self.phase}.png'.
            Only works if 'save' is set to 'True'.

        """

        if plot_together:

            new_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]

            exp = np.round(np.log10(self.data[self.bands[0]]['flux'].max()), 0)
            y_norm = 10**exp

            # to set plot limits
            if plot_type=='flux':
                plot_lim_vals = [[self.data[band]['flux'].min(), self.data[band]['flux'].max()] for band in self.bands]
                plot_lim_vals = np.ndarray.flatten(np.asarray(plot_lim_vals))/y_norm
                ymin_lim = np.r_[plot_lim_vals, 0.0].min()*0.9
                if ymin_lim < 0.0:
                    ymin_lim *= 1.1/0.9
                ymax_lim = plot_lim_vals.max()*1.05
            elif plot_type=='mag':
                plot_lim_vals = [[-2.5*np.log10(self.data[band]['flux'].min()) + self.data[band]['zp'],
                                  -2.5*np.log10(self.data[band]['flux'].max()) + self.data[band]['zp']] for band in self.bands]
                plot_lim_vals = np.ndarray.flatten(np.asarray(plot_lim_vals))
                ymin_lim = np.nanmin(plot_lim_vals)*0.98
                ymax_lim = np.nanmax(plot_lim_vals)*1.02

            fig, ax = plt.subplots(figsize=(8, 6))
            for i, band in enumerate(self.bands):

                time, flux, std = np.copy(self.lc_fits[band]['mjd']), np.copy(self.lc_fits[band]['flux']), np.copy(self.lc_fits[band]['std'])
                data_time, data_flux, data_std = np.copy(self.data[band]['mjd']), np.copy(self.data[band]['flux']), np.copy(self.data[band]['flux_err'])

                if plot_type=='flux':
                    flux, std = flux/y_norm, std/y_norm
                    data_flux, data_std = data_flux/y_norm, data_std/y_norm

                    ax.errorbar(data_time, data_flux, data_std, fmt='o', capsize=3, color=new_palette[i],label=band)
                    ax.plot(time, flux,'-', color=new_palette[i])
                    ax.fill_between(time, flux-std, flux+std, alpha=0.5, color=new_palette[i])
                    ax.set_ylabel(r'Flux [10$^{%.0f}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'%exp, fontsize=16, family='serif')

                elif plot_type=='mag':
                    mag = -2.5*np.log10(flux) + self.data[band]['zp']
                    err = np.abs(2.5*std/(flux*np.log(10)))
                    data_mag = -2.5*np.log10(data_flux) + self.data[band]['zp']
                    data_err = np.abs(2.5*data_std/(data_flux*np.log(10)))

                    ax.errorbar(data_time, data_mag, data_err, fmt='o', capsize=3, color=new_palette[i],label=band)
                    ax.plot(time, mag,'-', color=new_palette[i])
                    ax.fill_between(time, mag-err, mag+err, alpha=0.5, color=new_palette[i])
                    ax.set_ylabel(r'Apparent Magnitude [mag]', fontsize=16, family='serif')

            ax.axvline(x=self.tmax, color='r', linestyle='--')
            #ax.axvline(x=self.lc_fits[self.pivot_band]['tmax'], color='k', linestyle='--')
            ax.minorticks_on()
            ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True, labelsize=16)
            ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True, labelsize=16)
            ax.set_xlabel('Modified Julian Date', fontsize=16, family='serif')

            ax.set_title(f'{self.name} (z = {self.z:.5})', fontsize=18, family='serif')
            ax.legend(fontsize=13)
            ax.set_ylim(ymin_lim, ymax_lim)

            if plot_type=='mag':
                plt.gca().invert_yaxis()
        else:
            h = 3
            v = math.ceil(len(self.bands) / h)

            fig = plt.figure(figsize=(15, 5*v))
            gs = gridspec.GridSpec(v , h)

            for i, band in enumerate(self.bands):
                j = math.ceil(i % h)
                k =i // h
                ax = plt.subplot(gs[k,j])

                time, flux, std = self.lc_fits[band]['mjd'], self.lc_fits[band]['flux'], self.lc_fits[band]['std']
                ax.errorbar(self.data[band]['mjd'], self.data[band]['flux'], self.data[band]['flux_err'], fmt='ok')
                ax.plot(time, flux,'-')
                ax.fill_between(time, flux-std, flux+std, alpha=0.5)

                ax.axvline(x=self.tmax, color='r', linestyle='--')
                #ax.axvline(x=self.lc_fits[self.pivot_band]['tmax'], color='k', linestyle='--')
                ax.set_title(f'{band}', fontsize=16, family='serif')
                ax.xaxis.set_tick_params(labelsize=15)
                ax.yaxis.set_tick_params(labelsize=15)
                ax.minorticks_on()
                ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
                ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)

                fig.text(0.5, 0.95, f'{self.name} (z = {self.z:.5})', ha='center', fontsize=20, family='serif')
                fig.text(0.5, 0.04, 'Modified Julian Date', ha='center', fontsize=18, family='serif')
                fig.text(0.04, 0.5, r'Flux [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', va='center', rotation='vertical', fontsize=18, family='serif')

        if save:
            if fig_name is None:
                fig_name = f'{self.name}_lcfits.png'
            fig.tight_layout()
            plt.savefig(f'plots/{fig_name}')

        plt.show()

    ############################################################################
    ######################### Light Curves Correction ##########################
    ############################################################################

    def mangle_sed(self, kernel='matern32', method='gp'):
        """Mangles the SED with the given method to match the SN magnitudes.

        Parameters
        ----------
        kernel : str, default 'squaredexp'
            Kernel to be used for the gaussian process fit.  Possible choices are: 'matern52',
            'matern32', 'squaredexp'.
        method: str, default 'gp'
            Method to mangle the SED: either 'gp' for gaussian process or 'spline' for spline.
            NOTE: 'spline' method does not return correct errors, this needs to be fixed in the future.

        """

        phases = np.arange(-10, 21, 1)
        lc_phases = self.lc_fits['phaseXwave'].T[0]

        ####################################
        ##### Calculate SED photometry #####
        ####################################
        self.sed_lcs = {band:{'flux':[], 'mjd':None, 'phase':None} for band in self.bands}
        sed_df = self.sed['info'].copy()
        sed_df = sed_df[(lc_phases.min() <= sed_df.phase) & (sed_df.phase <= lc_phases.max())]  # to match the available epochs from the lcs
        sed_df = sed_df[sed_df.phase.isin(phases)]  # to match the requested epochs

        # first redshift the SED ("move" it in z) and then apply extinction from MW only
        sed_df.wave, sed_df.flux = sed_df.wave.values*(1+self.z), sed_df.flux.values/(1+self.z)
        sed_df.flux = redden(sed_df.wave.values, sed_df.flux.values, self.ra, self.dec)

        for phase in sed_df.phase.unique():
            phase_df = sed_df[sed_df.phase==phase]
            epoch_wave, epoch_flux = phase_df.wave.values, phase_df.flux.values
            for band in self.bands:
                band_flux = run_filter(phase_df.wave.values, phase_df.flux.values, self.filters[band]['wave'],
                                       self.filters[band]['transmission'], self.filters[band]['response_type'])
                self.sed_lcs[band]['flux'].append(band_flux)

        sed_mjd = sed_df.phase.unique()*(1+self.z) + self.tmax
        sed_phases = sed_df.phase.unique()

        for band in self.bands:
            self.sed_lcs[band]['flux'] = np.array(self.sed_lcs[band]['flux'])
            self.sed_lcs[band]['phase'] = sed_phases
            self.sed_lcs[band]['mjd'] = sed_mjd

        ###################################
        ####### set-up for mangling #######
        ###################################
        # minor linear interpolation of observed light curves to match the exact SED phases
        interp_lc_fluxes = {band:np.interp(self.sed_lcs[band]['phase'], self.lc_fits[band]['phase'], self.lc_fits[band]['flux']) for band in self.bands}
        interp_lc_stds = {band:np.interp(self.sed_lcs[band]['phase'], self.lc_fits[band]['phase'], self.lc_fits[band]['std']) for band in self.bands}
        flux_ratios = {band:interp_lc_fluxes[band]/self.sed_lcs[band]['flux'] for band in self.bands}
        flux_ratios_err = {band:interp_lc_stds[band]/self.sed_lcs[band]['flux'] for band in self.bands}

        flux_ratios_array = np.hstack(np.array([flux_ratios[band] for band in self.bands]))
        flux_ratios_err_array = np.hstack(np.array([flux_ratios_err[band] for band in self.bands]))
        obs_flux_array = np.hstack(np.array([interp_lc_fluxes[band] for band in self.bands]))  # to compare the mangled SED with
        obs_err_array = np.hstack(np.array([interp_lc_stds[band] for band in self.bands]))  # NECESSARY?

        phase_array = np.hstack(np.array([sed_phases for band in self.bands]))
        wave_array = np.hstack(np.array([[self.filters[band]['eff_wave']]*len(self.sed_lcs[band]['phase']) for band in self.bands]))

        bands_waves = np.hstack(np.array([self.filters[band]['wave'] for band in self.bands]))
        bands_edges = np.array([bands_waves.min(), bands_waves.max()])  # to match the grid with the observed data
        epoch_edges = np.array([sed_phases.min(), sed_phases.max()])  # same here

        ################################
        ########## mangle SED ##########
        ################################
        #mangling_results = fit_2dgp(phase_array, wave_array, flux_ratios_array, flux_ratios_err_array,
        #                                            kernel=kernel, x1_edges=epoch_edges, x2_edges=bands_edges, normalize_x1=False)
        mangling_results = mangle(phase_array, wave_array,
                                    flux_ratios_array, flux_ratios_err_array,
                                        sed_df, obs_flux_array, obs_err_array,
                                            self.bands, self.filters, kernel=kernel,
                                                x1_edges=epoch_edges, x2_edges=bands_edges, normalize_x1=False)

        self.mangling_results = mangling_results
        self.sed_fits['phaseXwave'], self.sed_fits['ratio_mu'], self.sed_fits['ratio_std'] = mangling_results
        ####################################################
        phaseXwave, ratio_mu, ratio_std = mangling_results
        phaseXwave.T[0] = np.round(phaseXwave.T[0], 1)

        sed_flux_array = []
        sed_err_array = []
        corr_sed_df = pd.DataFrame(columns=sed_df.columns)
        for phase in np.unique(phaseXwave.T[0]):
            if phase in phase_array:
                mask = np.where(phaseXwave.T[0]==phase)
                wave = phaseXwave.T[1][mask]
                ratio = ratio_mu[mask]
                err = ratio_std[mask]

                phase_df = sed_df[sed_df.phase==phase]
                sed_flux = np.interp(wave, phase_df.wave.values, phase_df.flux.values)
                mod_sed_flux = sed_flux*ratio  # convolve SED flux with mangling function
                mod_sed_err = sed_flux*err

                # correct SED for extinction and then blueshift it ("move" it in z to restframe)
                mod_sed_flux = deredden(wave, mod_sed_flux, self.ra, self.dec)
                mod_sed_err = deredden(wave, mod_sed_err, self.ra, self.dec)
                wave, mod_sed_flux, mod_sed_err = wave/(1+self.z), mod_sed_flux*(1+self.z), mod_sed_err*(1+self.z)

                temp_sed_df = pd.DataFrame({'phase':[phase]*len(wave), 'wave':wave, 'flux':mod_sed_flux, 'err':mod_sed_err})
                corr_sed_df = corr_sed_df.append(temp_sed_df, ignore_index=True)

                bands_fluxes = [run_filter(wave, mod_sed_flux, self.filters[band]['wave'], self.filters[band]['transmission'],
                                                                self.filters[band]['response_type']) for band in self.bands]
                bands_errs = [run_filter(wave, mod_sed_flux, self.filters[band]['wave'], self.filters[band]['transmission'],
                                                                self.filters[band]['response_type']) for band in self.bands]
                sed_flux_array.append(bands_fluxes)
                sed_err_array.append(bands_errs)

        sed_flux_array = np.array(sed_flux_array).T
        self.sed['corrected_sed'] = corr_sed_df

        '''
        # estimate precision of the mangling function
        mag_diffs = {}
        for band, obs_flux in zip(valid_bands, obs_fluxes):
            band_wave, band_transmission = self.filters[band]['wave'], self.filters[band]['transmission']
            response_type = self.filters[band]['response_type']
            model_flux = run_filter(mangled_wave, mangled_flux, band_wave, band_transmission, response_type)

            mag_diffs[band] = -2.5*np.log10(obs_flux) - (-2.5*np.log10(model_flux))

        self.sed['wave'], self.sed['flux'], self.sed['flux_err'] = mangled_wave, mangled_flux, mangled_flux_err
        self.mangling_results.update({self.phase:mangling_results})
        self.mangling_results[self.phase].update({'mag_diff':mag_diffs})  # this line actually adds the 'mag_diff' key

        self.set_eff_wave()
        '''


    def plot_mangling_function(self, phase=None, mangle_only=True, verbose=True, save=False, fig_name=None):
        """Plot the mangling function for a given phase.

        Parameters
        ----------
        band_list : list, default 'None'
            List of filters to plot. If 'None', band list is set to 'self.bands'.
        mangle_only : bool, default 'True'
            If 'True', only plots the mangling function, else, plots the SEDs and filters as well (in a
            relative scale).
        verbose : bool, default 'True'
            If 'True', returns the difference between the magnitudes from the fits and the magnitudes from the
            modified SED after mangling, for each of the bands in 'band_list'.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be ''{self.name}_sed{self.phase}.png'.
            Only works if 'save' is set to 'True'.

        """

        if phase is None:
            phase = self.phase

        assert (phase in self.mangling_results.keys()), f'phase {phase} does not have a mangling result.'

        man = self.mangling_results[phase]
        eff_waves = np.copy(man['init_vals']['waves'])
        init_flux_ratios = np.copy(man['init_vals']['flux_ratios'])
        flux_ratios_err = np.copy(man['init_vals']['flux_ratios_err'])

        opt_flux_ratios = np.copy(man['opt_vals']['flux_ratios'])
        obs_fluxes = np.copy(man['obs_vals']['fluxes'])
        sed_fluxes = np.copy(man['sed_vals']['fluxes'])

        x, y, yerr = np.copy(man['opt_fit']['waves']), np.copy(man['opt_fit']['flux_ratios']), np.copy(man['opt_fit']['flux_ratios_err'])
        mang_sed_wave, mang_sed_flux = man['mangled_sed']['wave'], man['mangled_sed']['flux']
        init_sed_wave, init_sed_flux = man['init_sed']['wave'], man['init_sed']['flux']

        kernel = man['kernel']
        bands = man['mag_diff'].keys()

        if mangle_only:
            f, ax = plt.subplots(figsize=(8,6))
            ax2 = ax.twiny()

            exp = np.round(np.log10(init_flux_ratios.max()), 0)
            y_norm = 10**exp
            init_flux_ratios, flux_ratios_err = init_flux_ratios/y_norm, flux_ratios_err/y_norm
            y, yerr = y/y_norm, yerr/y_norm
            opt_flux_ratios = opt_flux_ratios/y_norm

            ax.errorbar(eff_waves, init_flux_ratios, flux_ratios_err, fmt='o', capsize=3, label='Initial values')
            ax.plot(x, y)
            ax.fill_between(x, y-yerr, y+yerr, alpha=0.5, color='orange')
            ax.errorbar(eff_waves, opt_flux_ratios, flux_ratios_err, fmt='*', capsize=3, color='red', label='Optimized values')

            ax.set_xlabel(r'Observer-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax.set_ylabel(r'Flux$_{\rm Obs}$ / Flux$_{\rm Temp} \times$ 10$^{%.0f}$'%exp, fontsize=16, family='serif')
            ax.set_title(f'Mangling Function', fontsize=18, family='serif')
            ax.minorticks_on()
            ax.tick_params(which='major', length=8, width=1, direction='in', right=True, labelsize=16)
            ax.tick_params(which='minor', length=4, width=1, direction='in', right=True, labelsize=16)

            xticks = x[::len(x)//(len(self.bands)+1)]
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(np.round(xticks/(1+self.z), 0))
            ax2.set_xlabel(r'Rest-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax2.tick_params(which='major', length=8, width=1, direction='in', labelsize=16)
            ax2.tick_params(which='minor', length=4, width=1, direction='in', labelsize=16)

            for i, band in enumerate(bands):
                x1, y1 = ax.transLimits.transform((eff_waves[i], init_flux_ratios[i]))
                ax.text(x1, y1+(-1)**i*0.12, band, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=14)
            ax.legend(loc='upper right', fontsize=12)

        else:
            f, ax = plt.subplots(figsize=(8,6))
            ax2 = ax.twiny()
            ax3 = ax.twinx()

            norm = 2  # for bands
            norm2 = 1  # for SEDs
            index = (len(bands)-1)//2  # index of the band to do relative comparison

            init_sed_flux2 = init_sed_flux/sed_fluxes[index]
            mang_sed_flux2 = mang_sed_flux/obs_fluxes[index]
            sed_fluxes2 =sed_fluxes/sed_fluxes[index]
            obs_fluxes2 = obs_fluxes/obs_fluxes[index]

            # filters
            for i, band in enumerate(bands):
                wave, trans = self.filters[band]['wave'], self.filters[band]['transmission']
                if i==index:
                    ax3.plot(wave, trans/trans.max()*norm, color='b', alpha=0.4)
                else:
                    ax3.plot(wave, trans/trans.max()*norm, color='k', alpha=0.4)

            # mangling function
            ax.plot(x, y/opt_flux_ratios[index], 'green')
            indexes = [np.argmin(np.abs(x-wave_val)) for wave_val in eff_waves]
            #ax.plot(eff_waves, opt_flux_ratios/opt_flux_ratios[index], 'sg')
            ax.plot(eff_waves, y[indexes]/opt_flux_ratios[index], 'sg')

            # initial sed and fluxes
            ax3.plot(init_sed_wave, init_sed_flux2*norm2, '--k')  # initial sed
            ax3.plot(eff_waves, sed_fluxes2*norm2, 'ok', ms=12, label='Template values', alpha=0.8, fillstyle='none')  # initial sed fluxes

            # optimized sed and fluxes
            ax3.plot(mang_sed_wave, mang_sed_flux2*norm2, 'red')  # mangled sed
            ax3.plot(eff_waves, obs_fluxes2*norm2,'*r', ms=12, label='Observed values')  # observed fluxes

            ax.set_xlabel(r'Observer-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax.set_ylabel(r'Relative Mangling Function', fontsize=16, family='serif', color='g')
            ax.set_title(f'Mangling Function', fontsize=18, family='serif')
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim((y/opt_flux_ratios[index]).min()*0.8, (y/opt_flux_ratios[index]).max()*1.2)
            ax.minorticks_on()
            ax.tick_params(which='major', length=8, width=1, direction='in', labelsize=16)
            ax.tick_params(which='minor', length=4, width=1, direction='in', labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16, labelcolor='g')

            xticks = x[::len(x)//(len(self.bands)+1)]
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(np.round(xticks/(1+self.z), 0))
            ax2.set_xlabel(r'Rest-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax2.tick_params(which='major', length=8, width=1, direction='in', labelsize=16)
            ax2.tick_params(which='minor', length=4, width=1, direction='in', labelsize=16)
            ax2.xaxis.set_tick_params(labelsize=16)

            ax3.set_ylim(0, None)
            ax3.yaxis.set_tick_params(labelsize=16)
            ax3.set_ylabel(r'Relative Flux', fontsize=16, family='serif', rotation=270, labelpad=20)
            ax3.tick_params(which='major', length=8, width=1, direction='in', labelsize=16)
            ax3.tick_params(which='minor', length=4, width=1, direction='in', labelsize=16)

            ax3.legend(loc='upper right', fontsize=12)

        if save:
            if fig_name is None:
                fig_name = f'{self.name}_mangling_phase{phase}.png'
            f.tight_layout()
            plt.savefig(f'plots/{fig_name}')

        plt.show()

        if verbose:
            print(f'Mangling results, i.e., difference between mangled SED and "observed" magnitudes, at phase {phase}:')
            for band, diff in man['mag_diff'].items():
                print(f'{band}: {diff:.4f} [mags]')


    def check_B_peak(self, threshold=0.2, iter_num=0, maxiter=5, scaling=0.86, verbose=False, **mangle_kwargs):
        """Estimate the B-band peak from the corrected light curves.

        Finds B-band peak and compares it with the initial value. If they do not match within the given threshold,
        re-do the light-curves correction process with the new estimated peak. This whole process is done a several
        times until the threshold or the maximum number of iteration is reached, whichever comes first.

        Parameters
        ----------
        threshold : float, default '0.2'
            Threshold for the difference between the initial B-band peak estimation and the new estimation.
        iter_num : int, default '1'
            This value counts the number of iteration for the light-curves correction process.
        maxiter : int, default '5'
            Maximum number of iterations.
        scaling : float, default '0.86'
            Check 'correct_extinction()' for more information.
        **mangle_kwargs :
            Check 'mangle_sed()' for more information.

        """

        B_band = 'Bessell_B'
        phase, flux, flux_err = fit_gp(self.lc_correct[B_band]['phase'], self.lc_correct[B_band]['flux'],  self.lc_correct[B_band]['flux_err'])

        try:
            # use interpolated data
            peakidxs = peak.indexes(flux, thres=.3, min_dist=5//(phase[1]-phase[0]))
            # pick the index of the first peak in case some band has 2 peaks (like IR bands)
            idx_max = np.asarray([idx for idx in peakidxs if all(flux[:idx]<flux[idx])]).min()
            phase_max = phase[idx_max]  # this needs to be as close to zero as possible
            flux_max = flux[idx_max]
            flux_err_max = flux_err[idx_max]

            # use discrete data
            idx_max = np.argmin(np.abs(phase_max - self.lc_correct[B_band]['phase']))
            flux_max_disc = self.lc_correct[B_band]['flux'][idx_max]

        except:
            raise ValueError(f'Unable to obtain an accurate B-band peak for {self.name}!')

        if (iter_num <= maxiter) and ((np.abs(phase_max) > threshold) or (np.abs(flux_max_disc-flux_max) > flux_err_max)):
        #if (iter_num <= maxiter) and (np.abs(phase_max) > threshold):
            iter_num += 1
            if verbose:
                print(f'{self.name} iteration number {iter_num}')
            self.tmax += phase_max*(1+self.z)
            #print(f'{self.name} iteration number {iter_num}')
            self.set_interp_data(restframe_phases=self.lc_correct[B_band]['phase'])  # set interpolated data with new tmax
            self.correct_light_curve(threshold=threshold, iter_num=iter_num, maxiter=maxiter, scaling=scaling, **mangle_kwargs)
            #self.check_B_peak(threshold=threshold, iter_num=iter_num+1, maxiter=maxiter, scaling=scaling, **mangle_kwargs)

        elif iter_num > maxiter:
            raise ValueError(f'Unable to constrain B-band peak for {self.name}!')

        else:
            self.lc_parameters['phase_max'] = phase_max


    def calculate_lc_params(self):
        """Calculates the SN light curve parameters.

        Estimation of B-band peak apparent magnitude (mb), stretch (dm15) and color ((B-V)max) parameters.
        An interpolation of the corrected light curves is done as well as part of this process.

        """

        ########################################
        ######## Calculate Light Curves ########
        ########################################
        corrected_lcs = {}
        lcs_fluxes = []
        lcs_errs = []
        corr_sed_df = self.sed['corrected_sed']
        phases = corr_sed_df.phase.unique()
        bands = ['Bessell_B', 'Bessell_V']  # THIS IS A TEMPORARY FIX

        for phase in phases:
            phase_df = corr_sed_df[corr_sed_df.phase==phase]
            phase_wave = phase_df.wave.values
            phase_flux = phase_df.flux.values
            phase_err = phase_df.err.values
            bands_fluxes = [run_filter(phase_wave, phase_flux, self.filters[band]['wave'], self.filters[band]['transmission'],
                                                            self.filters[band]['response_type']) for band in bands]
            bands_errs = [run_filter(phase_wave, phase_err, self.filters[band]['wave'], self.filters[band]['transmission'],
                                                            self.filters[band]['response_type']) for band in bands]
            lcs_fluxes.append(bands_fluxes)
            lcs_errs.append(bands_errs)
        lcs_fluxes = np.array(lcs_fluxes)
        lcs_errs = np.array(lcs_errs)
        corrected_lcs = {band:{'phase':phases, 'flux':lcs_fluxes.T[i], 'err':lcs_errs.T[i]} for i, band in enumerate(bands)}
        self.corrected_lcs = corrected_lcs

        bessell_b = 'Bessell_B'
        zp_b = calc_zp(self.filters[bessell_b]['wave'], self.filters[bessell_b]['transmission'],
                        self.filters[bessell_b]['response_type'], 'vega')
        self.corrected_lcs[bessell_b]['zp'] = zp_b

        ########################################
        ### Calculate Light Curve parameters ###
        ########################################
        # B-band peak apparent magnitude
        try:
            phase_b, flux_b, flux_err_b = self.corrected_lcs[bessell_b]['phase'], self.corrected_lcs[bessell_b]['flux'], self.corrected_lcs[bessell_b]['err']
            id_bmax = np.where(phase_b==0.0)[0][0]
            mb = -2.5*np.log10(flux_b[id_bmax]) + zp_b
            dmb = np.abs(2.5*flux_err_b[id_bmax]/(flux_b[id_bmax]*np.log(10)))
        except:
            mb = dmb = np.nan

        # Stretch parameter
        try:
            id_15 = np.where(phase_b==15.0)[0][0]
            B15 = -2.5*np.log10(flux_b[id_15]) + zp_b
            B15_err = np.abs(2.5*flux_err_b[id_15]/(flux_b[id_15]*np.log(10)))
            dm15 = B15 - mb
            dm15err = np.sqrt(dmb**2 + B15_err**2)
        except:
            dm15 = dm15err = np.nan

        # Colour
        try:
            bessell_v = 'Bessell_V'
            zp_v = calc_zp(self.filters[bessell_v]['wave'], self.filters[bessell_v]['transmission'],
                            self.filters[bessell_v]['response_type'], 'vega')
            self.corrected_lcs[bessell_v]['zp'] = zp_v
            phase_v, flux_v, flux_err_v = self.corrected_lcs[bessell_v]['phase'], self.corrected_lcs[bessell_v]['flux'], self.corrected_lcs[bessell_v]['err']

            id_v0 = np.where(phase_v==0.0)[0][0]
            V0 = -2.5*np.log10(flux_v[id_v0]) + zp_v
            V0err = np.abs(2.5*flux_err_v[id_v0]/(flux_v[id_v0]*np.log(10)))
            color = mb - V0
            dcolor = np.sqrt(dmb**2 + V0err**2)
        except:
            color = dcolor = np.nan

        self.lc_parameters = {'mb':mb, 'dmb':dmb, 'dm15':dm15,
                              'dm15err':dm15err, 'color':color, 'dcolor':dcolor}


    def display_results(self, band=None, plot_type='mag', save=False, fig_name=None):
        """Displays the rest-frame light curve for the given band.

        Plots the rest-frame band light curve together with a gaussian fit to it. The parameters estimated with
        'calc_lc_parameters()' are shown as well.

        Parameters
        ----------
        band : str, default 'None'
            Name of the band to be plotted. If 'None', band is set to 'Bessell_B'.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be ''{self.name}_sed{self.phase}.png'.
            Only works if 'save' is set to 'True'.

        """

        assert (plot_type=='mag' or plot_type=='flux'), f'"{plot_type}" is not a valid plot type.'

        mb = self.lc_parameters['mb']
        dmb = self.lc_parameters['dmb']
        color = self.lc_parameters['color']
        dcolor = self.lc_parameters['dcolor']
        dm15 = self.lc_parameters['dm15']
        dm15err = self.lc_parameters['dm15err']

        if band is None:
            band = 'Bessell_B'

        x = np.copy(self.corrected_lcs[band]['phase'])
        y = np.copy(self.corrected_lcs[band]['flux'])
        yerr = np.copy(self.corrected_lcs[band]['err'])
        zp = np.copy(self.corrected_lcs[band]['zp'])
        #x_fit = np.copy(self.lc_final_fits[band]['phase'])
        #y_fit = np.copy(self.lc_final_fits[band]['flux'])
        #yerr_fit = np.copy(self.lc_final_fits[band]['flux_err'])

        if plot_type=='flux':
            exp = np.round(np.log10(y.max()), 0)
            y_norm = 10**exp
            y /= y_norm
            yerr /= y_norm
            #y_fit /= y_norm
            #yerr_fit /= y_norm

        elif plot_type=='mag':
            # y, yerr, y_fit, yerr_fit variables get reassigned
            yerr = np.abs(2.5*yerr/(y*np.log(10)))
            y = -2.5*np.log10(y) + zp
            #yerr_fit = np.abs(2.5*yerr_fit/(y_fit*np.log(10)))
            #y_fit = -2.5*np.log10(y_fit) + self.lc_final_fits[band]['zp']


        f, ax = plt.subplots(figsize=(8,6))
        ax.errorbar(x, y, yerr, fmt='-.o', capsize=3, color='k')
        #ax.plot(x_fit, y_fit, 'r-', alpha=0.5)
        #ax.fill_between(x_fit, y_fit+yerr_fit, y_fit-yerr_fit, alpha=0.5, color='r')

        ax.text(0.75, 0.9,r'm$_B^{\rm max}$=%.3f$\pm$%.3f'%(mb, dmb), ha='center', va='center', fontsize=15, transform=ax.transAxes)
        ax.text(0.75, 0.8,r'$\Delta$m$_{15}$($B$)=%.3f$\pm$%.3f'%(dm15, dm15err), ha='center', va='center', fontsize=15, transform=ax.transAxes)
        ax.text(0.75, 0.7,r'($B-V$)$_{\rm max}$=%.3f$\pm$%.3f'%(color, dcolor), ha='center', va='center', fontsize=15, transform=ax.transAxes)

        ax.set_xlabel(f'Phase with respect to B-band peak [days]', fontsize=16, family='serif')
        ax.set_title(f'{self.name} ({band}, z={self.z:.5})', fontsize=16, family='serif')
        if plot_type=='flux':
            ax.set_ylabel('Flux [10$^{%.0f}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'%exp, fontsize=16, family='serif')
            ax.set_ylim(y.min()*0.90, y.max()*1.05)
        elif plot_type=='mag':
            ax.set_ylabel('Apparent Magnitude', fontsize=16, family='serif')
            ax.set_ylim(y.min()*0.98, y.max()*1.02)
            plt.gca().invert_yaxis()
        ax.minorticks_on()
        ax.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=16)
        ax.tick_params(which='minor', length=4, width=1, direction='in', top=True, right=True, labelsize=16)

        if save:
            if fig_name is None:
                fig_name = f'{self.name}_{band}_results.png'
            f.tight_layout()
            plt.savefig(f'plots/{fig_name}')

        plt.show()
