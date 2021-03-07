import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def interpolate(ell, cl):
    model = InterpolatedUnivariateSpline(np.log(ell), np.log(cl), ext=2)
    ell_new = np.arange(np.min(ell), np.max(ell), 1)
    cl = np.exp(model(np.log(ell_new)))
    return ell_new, cl
