import numpy as np
import pyccl as ccl
from pyccl.background import comoving_radial_distance as chi_a
from scipy.interpolate import InterpolatedUnivariateSpline as inter
from pyccl.power import nonlin_matter_power
from scipy.integrate import simpson
from LensKern import *
from scipy import optimize

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
#        print(min_chi)
#        print(max_chi)
        self.eval_grid = np.linspace(np.finfo(float).eps,max_chi, num=num_evaluation)
        agrid = np.linspace(0.05, 1.0, num=100)
        chi_interpo = np.array([chi_a(cosmo, el) for el in agrid])
        self.interpolator_a_chi = inter(chi_interpo[::-1], agrid[::-1], ext=2)
        self.c = 299792.
        self.integrand_prefac = (9./4.) * self.om_mat**2 * (self.H0/self.c)**4 * (1./self.a_chi(self.eval_grid))**2


    def a_chi(self, chi):
        return self.interpolator_a_chi(chi)

    def get_cl_partial(self, ell, lens_kernel1_eval, lens_kernel2_eval):
        #print('entry')
        #print(lens_kernel1_eval[0])
        #print(lens_kernel2_eval[0])
        power = np.array([nonlin_matter_power(self.cosmo, ell/chi, float(self.a_chi(chi))) for chi in self.eval_grid])
        integrand = self.integrand_prefac*power*lens_kernel1_eval*lens_kernel2_eval
        result = simpson(integrand, self.eval_grid)
        return result

    def get_cl(self, ell_vec):
        lens_kern1_eval = np.array([self.lens_kern1.evaluate(el) for el in self.eval_grid])
        lens_kern2_eval = np.array([self.lens_kern2.evaluate(el) for el in self.eval_grid])
        #print('get_cl')
        #print(lens_kern1_eval[0])
        #print(lens_kern2_eval[0])
        result = []
        for i in range(len(ell_vec)):
            #print(ell_vec[i])
            #print(lens_kern1_eval[0])
            #print(lens_kern1_eval[0])
            result.append(self.get_cl_partial(ell_vec[i], lens_kern1_eval, lens_kern2_eval))
        return np.array(result)
        #return np.array([self._get_cl(ell, lens_kern1_eval, lens_kern2_eval)
        #                 for ell in ell_vec])


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

        return np.array(grad_cl)



cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                    transfer_function='bbks')

ell_vec = np.linspace(70,1000, 5)
#print(ell_vec)
chi_grid = np.linspace(100, 4000., 10)
delta_chi = (chi_grid[1] - chi_grid[0])/2.
breaks = chi_grid - delta_chi
breaks = np.append(breaks, breaks[-1] + 2.*delta_chi)

w = np.repeat(1/len(chi_grid), len(chi_grid))

hist_new = Hist(w, breaks)
limb_inte = LimberIntegrator(cosmo, hist_new, hist_new)

result_cl_fid = np.sum(limb_inte.get_cl(ell_vec))
result_grad = limb_inte.get_grad(ell_vec)

def loss_function(w_par):
    hist_new = Hist(w_par, breaks)
    limb_inte = LimberIntegrator(cosmo, hist_new, hist_new)
    cl = np.sum(limb_inte.get_cl(ell_vec))
    return 10**9*(cl - result_cl_fid)**2 + np.sum((w - w_par)**2)


def grad(w_par):
    hist_new = Hist(w_par, breaks)
    limb_inte = LimberIntegrator(cosmo, hist_new, hist_new)
    cl = np.sum(limb_inte.get_cl(ell_vec))
    grad_limb = limb_inte.get_grad(ell_vec)
    return -2.*(result_cl_fid - cl)*np.sum(grad_limb, axis=0)*10**9 - 2* (w - w_par)


xopt = optimize.minimize(loss_function,w+10,method='bfgs',jac=grad)
print(loss_function(w+10))
print(xopt['x'])
print(xopt['fun'])
print(w)
print(loss_function(w))
