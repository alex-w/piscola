import numpy as np
import george
import scipy

def fit_2dgp(x1_data, x2_data, y_data, yerr_data, kernel='matern52', x1_edges=None, x2_edges=None, normalize_x1=True, normalize_x2=True):
    """Fits data with gaussian process.

    The package 'george' is used for the gaussian process fit.

    Parameters
    ----------
    x_data : array
        Independent values.
    y_data : array
        Dependent values.
    yerr_data : array, int
        Dependent value errors.
    kernel : str, default 'squaredexp'
        Kernel to be used with the gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.
        If left default, 'matern52' is used to fit light curves and 'squaredexp' for the mangling function.
    mangling: bool, default 'False'
        If 'True', the fit is set to adjust the mangling function.
    x_edges: array-like, default 'None'
        Minimum and maximum x-axis values. These are used to extrapolate both edges if 'mangling==True'.

    Returns
    -------
    Returns the interpolated independent and dependent values with the 1-sigma standard deviation.

    """

    # define the objective function (negative log-likelihood in this case)
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    # and the gradient of the objective function
    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    x1, x2  = np.copy(x1_data), np.copy(x2_data)
    y, yerr = np.copy(y_data), np.copy(yerr_data)

    # normalize data
    y_norm = y.max()
    x1_norm = x2_norm = 1.0
    if normalize_x1:
        x1_norm = 1e4
    if normalize_x2:
        x2_norm = 1e3

    y /= y_norm
    yerr /= y_norm
    x1 /= x1_norm
    x2 /= x2_norm

    X = np.array([x1, x2]).reshape(2, -1).T

    # define kernel
    var, lengths = np.var(y), np.array([np.diff(x1).max(), np.diff(x2).max()])

    if kernel == 'matern52':
        ker = var * george.kernels.Matern52Kernel(lengths, ndim=2)
        #ker = var * (george.kernels.Matern52Kernel(lengths[0], ndim=2, axes=0) + george.kernels.ExpSquaredKernel(lengths[1], ndim=2, axes=1))
    elif kernel == 'matern32':
        ker = var * george.kernels.Matern32Kernel(lengths, ndim=2)
    elif kernel == 'squaredexp':
        ker = var * george.kernels.ExpSquaredKernel(lengths, ndim=2)
    else:
        raise ValueError(f'"{kernel}" is not a valid kernel.')

    mean_function = 0.0
    gp = george.GP(kernel=ker, solver=george.HODLRSolver, mean=mean_function)
    # initial guess
    if np.any(yerr):
        gp.compute(X, yerr)
    else:
        gp.compute(X)

    # optimization routine for hyperparameters
    p0 = gp.get_parameter_vector()
    results = scipy.optimize.minimize(neg_ln_like, p0, jac=grad_neg_ln_like)
    gp.set_parameter_vector(results.x)

    # check edges
    if np.any(x1_edges):
        x1_edges = np.copy(x1_edges)
        x1_edges /= x1_norm
        x1_min, x1_max = x1_edges[0], x1_edges[-1]
    else:
        x1_min, x1_max = x1.min(), x1.max()
    if np.any(x2_edges):
        x2_edges = np.copy(x2_edges)
        x2_edges /= x2_norm
        x2_min, x2_max = x2_edges[0], x2_edges[-1]
    else:
        x2_min, x2_max = x2.min(), x2.max()

    x1_min = np.floor(x1_min*x1_norm)/x1_norm
    x1_max = np.ceil(x1_max*x1_norm)/x1_norm
    x2_min = np.floor(x2_min*x2_norm)/x2_norm
    x2_max = np.ceil(x2_max*x2_norm)/x2_norm
    step1 = 0.1/x1_norm
    step2 = 5/x2_norm
    if not normalize_x1:
        step1 = 1/x1_norm
        step2 = 1/x2_norm

    X_predict = np.array(np.meshgrid(np.arange(x1_min, x1_max+step1, step1),
                             np.arange(x2_min, x2_max+step2, step2))).reshape(2, -1).T

    mu, var = gp.predict(y, X_predict, return_var=True)
    std = np.sqrt(var)

    # de-normalize results
    X_predict *= np.array([x1_norm, x2_norm])
    mu *= y_norm
    std *= y_norm

    return X_predict, mu, std
