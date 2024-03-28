import jax
import jaxopt
import numpy as np
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels, transforms

def prepare_gp_inputs(times, wavelengths, fluxes, flux_errors, y_norm, use_log=True):
    """Prepares the inputs for the Gaussian Process model fitting.
    
    Parameters
    ----------
    times: ndarray 
        Light-curve epochs.
    wavelengths: ndarray 
        Light-curve effective wavelength.
    fluxes: ndarray 
        Light-curve fluxes.
    flux_errors: ndarray 
        Light-curve flux errors.
    y_norm: float
        Normalisation used for the fluxes and errors. The maximum
        of the fluxes is recommended.
    use_log: bool, default ``True``.
        Whether to use logarithmic (base 10) scale for the 
        wavelength axis.
        
    Returns
    -------
    X: ndarray
        X-axis array for the Gaussian Process model.
    y: ndarray
        Y-axis array for the Gaussian Process model.
    yerr: ndarray
        Y-axis errors for the Gaussian Process model.
    """
    X = (times, wavelengths)
    if use_log is True:
        X = (times, jnp.log10(wavelengths))
    # normalise fluxes
    y = (fluxes / y_norm).copy()
    yerr = (flux_errors / y_norm).copy()

    return X, y, yerr

def fit_gp_model(times, wavelengths, fluxes, flux_errors, k1='Matern52', use_log=True):
    """Fits a Gaussian Process model to a SN multi-colour light curve.
    
    All input arrays MUST have the same length.

    Parameters
    ----------
    times: ndarray 
        Light-curve epochs.
    wavelengths: ndarray 
        Light-curve effective wavelength.
    fluxes: ndarray 
        Light-curve fluxes.
    flux_errors: ndarray 
        Light-curve flux errors.
    k1: str
        Kernel to be used for the time axis. Either ``Matern52``,
        ``Matern32`` or ``ExpSquared``.
    use_log: bool, default ``True``.
        Whether to use logarithmic (base 10) scale for the 
        wavelength axis.
        
    Returns
    -------
    gp_model: ~tinygp.gp.GaussianProcess
        Gaussian Process multi-colour light-curve model.
    """
    assert k1 in ['Matern52', 'Matern32', 'ExpSquared'], "Not a valid kernel"
    def build_gp(params):
        """Creates a Gaussian Process model.
        """
        # select time-axis kernel
        if k1 == 'Matern52':
            kernel1 = transforms.Subspace(0, kernels.Matern52(scale=jnp.exp(params["log_scale"][0])))
        elif k1 == 'Matern32':
            kernel1 = transforms.Subspace(0, kernels.Matern32(scale=jnp.exp(params["log_scale"][0])))
        else:
            kernel1 = transforms.Subspace(0, kernels.ExpSquared(scale=jnp.exp(params["log_scale"][0])))
        # wavelength-axis kernel
        kernel2 = transforms.Subspace(1, kernels.ExpSquared(scale=jnp.exp(params["log_scale"][1])))
        kernel = jnp.exp(params["log_amp"]) * kernel1 * kernel2
        diag = yerr ** 2 + jnp.exp(2 * params["log_noise"])

        return GaussianProcess(kernel, X, diag=diag)

    @jax.jit
    def loss(params):
        """Loss function for the Gaussian Process hyper-parameters optimisation.
        """
        return -build_gp(params).condition(y).log_probability
    
    y_norm = fluxes.max()  # for normalising the fluxes
    X, y, yerr = prepare_gp_inputs(times, wavelengths, fluxes, 
                                   flux_errors, y_norm, use_log=use_log)
    
    # GP hyper-parameters
    time_scale = 10  # days
    wave_scale = 1000  # angstroms
    if use_log is True:
        wave_scale = jnp.log10(wave_scale)
    params = {
        "log_amp": jnp.log(y.var()),
        "log_scale": jnp.log(np.array([time_scale, wave_scale])),
        "log_noise": jnp.log(np.max(yerr))
    }
    # Train the GP model
    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(params)
    gp_model = build_gp(soln.params)
    
    return gp_model

def fit_single_lightcurve(times, fluxes, flux_errors, k1='Matern52'):
    """Fits a Gaussian Process model to a single SN light curve.
    
    All input arrays MUST have the same length.

    Parameters
    ----------
    times: ndarray 
        Light-curve epochs.
    fluxes: ndarray 
        Light-curve fluxes.
    flux_errors: ndarray 
        Light-curve flux errors.
    k1: str
        Kernel to be used for the time axis. Either ``Matern52``,
        ``Matern32`` or ``ExpSquared``.
        
    Returns
    -------
    gp_model: ~tinygp.gp.GaussianProcess
        Gaussian Process light-curve model.
    """
    assert k1 in ['Matern52', 'Matern32', 'ExpSquared'], "Not a valid kernel"
    def build_gp(params):
        """Creates a Gaussian Process model.
        """
        # select time-axis kernel
        if k1 == 'Matern52':
            kernel1 = kernels.Matern52(scale=jnp.exp(params["log_scale"]))
        elif k1 == 'Matern32':
            kernel1 = kernels.Matern32(scale=jnp.exp(params["log_scale"]))
        else:
            kernel1 = kernels.ExpSquared(scale=jnp.exp(params["log_scale"]))
        kernel = jnp.exp(params["log_amp"]) * kernel1
        diag = yerr ** 2 

        return GaussianProcess(kernel, x, diag=diag)

    @jax.jit
    def loss(params):
        """Loss function for the Gaussian Process hyper-parameters optimisation.
        """
        return -build_gp(params).condition(y).log_probability
    
    x = times.copy()
    y = fluxes.copy() / fluxes.max()
    yerr = flux_errors.copy() / fluxes.max()
    
    # GP hyper-parameters
    time_scale = 10  # days
    params = {
        "log_amp": jnp.log(y.var()),
        "log_scale": jnp.log(time_scale),
    }
    # Train the GP model
    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(params)
    gp_model = build_gp(soln.params)
    
    return gp_model