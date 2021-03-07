import numpy as np


class WlCov(object):
    def __init__(self, ell, num_dens, std_shape, fsky):
        self.ell = np.arange(np.min(ell), np.max(ell), 1)
        self.num_dens = num_dens
        self.std_shape = std_shape
        self.fsky = fsky

    def get_cov(self, cl):
        cl_shot = (cl + self.std_shape**2/(2.*self.num_dens))**2
        prefactor = 2.0/((2.0 * self.ell + 1.)*self.fsky)
        covariance = prefactor * cl_shot
        cov_matrix = np.diag(covariance)
        return cov_matrix

