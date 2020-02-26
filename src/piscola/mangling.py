from .filter_integration import run_filter
from .gaussian_process import fit_2dgp
from .spline import fit_spline

import numpy as np
import lmfit

def mangling_residual(params, phase_array, wave_array, flux_ratios_err_array, sed_df,
                        obs_flux_array, obs_err_array, norm, bands, filters, kernel, x1_edges, x2_edges, normalize_x1):
    """Residual functions for the SED mangling minimization routine.

    Lmfit works in such a way that each parameters needs to have a residual value. In the case of the
    hyperparameters, a residual equal to the sum of the bands's residuals is used given that there is no
    model used to compare these values.

    Parameters
    ----------
    params : lmfit.Parameters()
        Flux values for each band to be minimized.
    eff_waves: array
        Effective wavelengths of the bands.
    flux_ratio_err : array
        "Observed" flux error values divided by the SED template values.
    sed_wave : array
        SED wavelength range
    sed_flux : array
        SED flux density values.
    obs_flux : array
        "Observed" flux values.
    obs_flux_err : array
        "Observed" flux error values.
    bands : list
        List of bands to performe minimization.
    filters : dictionary
        Dictionary with all the filters's information. Same format as 'sn.filters'.
    kernel : str
        Kernel to be used with the gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.
    method : str
        Fitting method. Either 'gp' for gaussian process or 'spline' for spline.

    Returns
    -------
    Array of residuals for each parameter.

    """

    # set up parameters
    phases_list = list(np.unique(phase_array))
    init_param_names = np.hstack(np.array([[f'{band}_{int(phase)}' for phase in phases_list] for band in bands]))
    param_names = [par_name.lstrip("0123456789\'.-").replace("'", "").replace(".", "_").replace("-", "m") for par_name in init_param_names]
    flux_ratios_array = np.asarray([params[par_name].value for par_name in param_names])

    # 2D GP fit
    phaseXwave, ratio_mu, ratio_std = fit_2dgp(phase_array, wave_array, flux_ratios_array, 0.0,  #flux_ratios_err_array
                                                kernel=kernel, x1_edges=x1_edges, x2_edges=x2_edges, normalize_x1=normalize_x1)
    phaseXwave.T[0] = np.round(phaseXwave.T[0], 1)  # the rounding is to avoid stupid python decimals

    ratio_mu *= norm  # de-normalise to compare values with observed light curves

    # Convolve SED with the 2D mangling function and estimate new light curves
    sed_flux_array = []
    for phase in np.unique(phaseXwave.T[0]):
        if phase in phase_array:
            mask = np.where(phaseXwave.T[0]==phase)
            wave = phaseXwave.T[1][mask]
            ratio = ratio_mu[mask]

            phase_df = sed_df[sed_df.phase==phase]
            sed_flux = np.interp(wave, phase_df.wave.values, phase_df.flux.values)
            mod_sed_flux = sed_flux*ratio  # apply mangling function

            bands_fluxes = [run_filter(wave, mod_sed_flux, filters[band]['wave'], filters[band]['transmission'], filters[band]['response_type']) for band in bands]
            sed_flux_array.append(bands_fluxes)

    sed_flux_array = np.hstack(np.array(sed_flux_array).T)  # to have the same shape and order as the observed fluxes
    residuals = -2.5*np.log10(obs_flux_array/sed_flux_array)
    print(np.round(residuals, 3))

    return residuals


def mangle(phase_array, wave_array, flux_ratios_array, flux_ratios_err_array, sed_df,
            obs_flux_array, obs_err_array, bands, filters, kernel, x1_edges, x2_edges, normalize_x1):
    """Mangling routine.

    A mangling of the SED is done by minimizing the the difference between the "observed" fluxes and the fluxes
    coming from the modified SED.

    Parameters
    ----------
    flux_ratio : array
        "Observed" flux values divided by the SED template values.
    flux_ratio_err : array
        "Observed" flux error values divided by the SED template values.
    sed_wave : array
        SED wavelength range
    sed_flux : array
        SED flux density values.
    bands : list
        List of bands to performe minimization.
    filters : dictionary
        Dictionary with all the filters's information. Same format as 'self.filters'.
    obs_flux : array
        "Observed" flux values.
    obs_flux_err : array
        "Observed" flux error values.
    kernel : str
        Kernel to be used with the gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.
    method : str
        Fitting method. Either 'gp' for gaussian process or 'spline' for spline.

    Returns
    -------
    Returns the mangled/modified SED with 1-sigma standard deviation and all the results
    from the mangling routine (these can plotted later to check the results).

    """

    ####################################
    #### optimize mangling function ####
    ####################################
    params = lmfit.Parameters()
    phases_list = list(np.unique(phase_array))
    init_param_names = np.hstack(np.array([[f'{band}_{int(phase)}' for phase in phases_list] for band in bands]))
    # "lmfit.Parameters" does not allow parameter names that do not start with letters.
    # It also does not like things like quotes ('), dots (.), etc. in the parameter names at all.
    param_names = [par_name.lstrip("0123456789\'.-").replace("'", "").replace(".", "_").replace("-", "m") for par_name in init_param_names]

    norm = flux_ratios_array.max()  # we normalize to avoid small numbers in the minimization routine

    for ratio_val, par_name in zip(flux_ratios_array/norm, param_names):
        params.add(par_name, value=ratio_val)#, min=ratio_val*0.5, max=ratio_val*2)

    args=(phase_array, wave_array, flux_ratios_err_array/norm, sed_df,
            obs_flux_array, obs_err_array, norm, bands, filters, kernel, x1_edges, x2_edges, normalize_x1)
    fitter = lmfit.Minimizer(userfcn=mangling_residual, params=params, fcn_args=args)
    result = fitter.minimize(method='nerlder',
                                options={'maxiter':3, 'maxfev':3, 'xatol':1e-3, 'fatol':1e-3, 'adaptive':True}
                            )
    ###############################
    #### use optimized results ####
    ###############################
    # de-normalise results
    print('Optimization passed!')
    opt_flux_ratios_array = np.asarray([result.params[par_name].value for par_name in param_names])*norm
    print(opt_flux_ratios_array)

    phaseXwave, ratio_mu, ratio_std = fit_2dgp(phase_array, wave_array, opt_flux_ratios_array, 0.0,
                                                kernel=kernel, x1_edges=x1_edges, x2_edges=x2_edges, normalize_x1=normalize_x1)
    phaseXwave.T[0] = np.round(phaseXwave.T[0], 1)  # the rounding is to avoid stupid python decimals

    # mangle SED and obtain light curves
    print('SED light curve now!')
    sed_flux_array = []
    for phase in np.unique(phaseXwave.T[0]):
        if phase in phase_array:
            mask = np.where(phaseXwave.T[0]==phase)
            wave = phaseXwave.T[1][mask]
            ratio = ratio_mu[mask]
            err = ratio_std[mask]

            phase_df = sed_df[sed_df.phase==phase]
            sed_flux = np.interp(wave, phase_df.wave.values, phase_df.flux.values)
            mod_sed_flux = sed_flux*ratio  # convolve SED flux with mangling function

            bands_fluxes = [run_filter(wave, mod_sed_flux, filters[band]['wave'], filters[band]['transmission'], filters[band]['response_type']) for band in bands]
            sed_flux_array.append(bands_fluxes)

    sed_flux_array = np.hstack(np.array(sed_flux_array).T)  # to have the same shape and order as the observed fluxes
    print('mangling ready!')
    #### propagate errors ####


    #### save results ####
    '''sed_fluxes = np.empty(0)
    for band in bands:
        band_flux = run_filter(sed_wave, sed_flux,
                               filters[band]['wave'], filters[band]['transmission'], filters[band]['response_type'])
        sed_fluxes = np.r_[sed_fluxes, band_flux]

    mangling_results = {'init_vals':{'waves':eff_waves, 'flux_ratios':flux_ratio, 'flux_ratios_err':flux_ratio_err},
                        'opt_vals':{'waves':eff_waves, 'flux_ratios':opt_flux_ratio, 'flux_ratios_err':flux_ratio_err},
                        'sed_vals':{'waves':eff_waves, 'fluxes':sed_fluxes},
                        'obs_vals':{'waves':eff_waves, 'fluxes':obs_fluxes, 'fluxes_err':obs_flux_err},
                        'opt_fit':{'waves':x_pred, 'flux_ratios':y_pred, 'flux_ratios_err':yerr_pred},
                        'init_sed':{'wave':sed_wave, 'flux':sed_flux},
                        'mangled_sed':{'wave':mangled_wave, 'flux':mangled_flux, 'flux_err':mangled_flux_err},
                        'kernel':kernel,
                        'result':result}'''

    return phaseXwave, ratio_mu, ratio_std
