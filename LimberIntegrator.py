import numpy as np
import pyccl as ccl
from pyccl.background import comoving_radial_distance as chi_a
from scipy.interpolate import InterpolatedUnivariateSpline as inter
from pyccl.power import nonlin_matter_power
from scipy.integrate import simps as simpson
from LensKern import *
from scipy import optimize
from logit import *
from scipy.stats import norm
from tools import *

class LimberIntegrator(object):
    def __init__(self, cosmo, lens_kern1, lens_kern2):

        self.lens_kern1 = lens_kern1
        self.lens_kern2 = lens_kern2
        self.cosmo = cosmo
        self.H0 = cosmo['h'] * 100
        self.om_mat = cosmo['Omega_c'] + cosmo['Omega_b']
        min_chi = np.min([self.lens_kern1.breaks[0], self.lens_kern2.breaks[0]])
        max_chi = np.max([self.lens_kern1.breaks[-1], self.lens_kern2.breaks[-1]])
        num_evaluation = 20
        self.eval_grid = np.linspace(np.finfo(float).eps, max_chi, num=num_evaluation)
        agrid = np.linspace(0.05, 1.0, num=100)
        chi_interpo = np.array([chi_a(cosmo, el) for el in agrid])
        self.interpolator_a_chi = inter(chi_interpo[::-1], agrid[::-1], ext=2)
        self.c = 299792.
        self.integrand_prefac = (9./4.) * self.om_mat**2 * (self.H0/self.c)**4 * (1./self.a_chi(self.eval_grid))**2


    def a_chi(self, chi):
        return self.interpolator_a_chi(chi)

    def get_cl_partial(self, ell, lens_kernel1_eval, lens_kernel2_eval):
        power = np.array([nonlin_matter_power(self.cosmo, ell/chi, float(self.a_chi(chi))) for chi in self.eval_grid])
        integrand = self.integrand_prefac*power*lens_kernel1_eval*lens_kernel2_eval
        result = simpson(integrand, self.eval_grid)
        return result

    def get_cl(self, ell_vec):
        lens_kern1_eval = np.array([self.lens_kern1.evaluate(el) for el in self.eval_grid])
        lens_kern2_eval = np.array([self.lens_kern2.evaluate(el) for el in self.eval_grid])
        result = []
        for i in range(len(ell_vec)):
            result.append(self.get_cl_partial(ell_vec[i], lens_kern1_eval, lens_kern2_eval))

        ell_new, cl_new = interpolate(ell_vec, np.array(result))
        return cl_new

    def _get_grad(self, ell):
        """ i is the tomographic bin number
        """

        grad_window = np.array([self.lens_kern1.gradient(el) for el in self.eval_grid])
        lens_kern2_eval = np.array([self.lens_kern2.evaluate(el) for el in self.eval_grid])
        grad_cl = np.zeros((grad_window.shape[1], ))
        for i in range(grad_window.shape[1]):
            curr_window = grad_window[:, i]
            grad_cl[i] = 2.*self.get_cl_partial(ell, curr_window, lens_kern2_eval)

        return grad_cl

    def get_grad(self, ell_vec):
        grad_cl = []
        for el in ell_vec:
            grad_cl.append(self._get_grad(el))
        grad_cl = np.array(grad_cl)
        new_cl = []
        for i in range(grad_cl.shape[1]):
            ell_new, grad_new = interpolate(ell_vec, grad_cl[:, i])
            new_cl.append(grad_new)

        new_cl = np.array(new_cl)

        return new_cl.T




