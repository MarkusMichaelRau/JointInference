import numpy as np
import pyccl as ccl
from scipy.stats import rv_histogram
from pyccl.background import comoving_radial_distance as chi_a
from scipy.interpolate import InterpolatedUnivariateSpline as inter
from pyccl.power import nonlin_matter_power #(cosmo, k, a
from scipy.integrate import simpson
import time
import numdifftools as nd
#pyccl.background.comoving_radial_distance
#use globally mpc as unit


class LimberIntegration(object):
    def __init__(self, cosmo, chi_grid_breaks, pchi):

        self.cosmo = cosmo
        self.H0 = cosmo['h'] * 100
        self.om_mat = cosmo['Omega_c'] + cosmo['Omega_b']
        self.chi_grid_breaks = chi_grid_breaks
        midpoints = np.array([chi_grid_breaks[i] + (chi_grid_breaks[i + 1] - chi_grid_breaks[i])/2. for i in range(len(chi_grid_breaks)-1)])
        self.comdist_grid = midpoints #units of Mpc
        self.pchi = pchi/np.sum(pchi)
        agrid = np.linspace(0.05, 1.0, num=100)
        chi_interpo = np.array([chi_a(cosmo, el) for el in agrid])
        self.interpolator_a_chi = inter(chi_interpo[::-1], agrid[::-1], ext=2)
        self.c = 299792.


    def a_chi(self, chi):
        return self.interpolator_a_chi(chi)

    def evaluate_lens_kernel(self, pchi):
        #chi is the midpoint
        diff_breaks = self.chi_grid_breaks[1:] - self.chi_grid_breaks[:-1]
        diff_logbreaks = np.log(self.chi_grid_breaks[1:]) - np.log(self.chi_grid_breaks[:-1])
        first_term = np.cumsum(pchi[::-1])[::-1]
        inner_vec = self.pchi/diff_breaks* diff_logbreaks
        second_term = self.comdist_grid*np.cumsum(inner_vec[::-1])[::-1]
        return first_term - second_term


    def _get_cl(self, ell, lens_kernel1, lens_kernel2):
        power = np.array([nonlin_matter_power(self.cosmo, ell/chi, float(self.a_chi(chi))) for chi in self.comdist_grid])
        a_chi = self.a_chi(self.comdist_grid)
        integrand_prefac = (9./4.) * self.om_mat**2 * (self.H0/self.c)**4 * (1./a_chi)**2
        integrand = integrand_prefac*power*lens_kernel1*lens_kernel2
        result = simpson(integrand, self.comdist_grid)
        return result

    def get_cl(self, ell, pchi_1, pchi_2):
        lens_kernel1 = self.evaluate_lens_kernel(pchi_1)
        lens_kernel2 = self.evaluate_lens_kernel(pchi_2)
        return self._get_cl(ell, lens_kernel1, lens_kernel2)

    def get_cl_deriv(self, ell, pchi):
        grad_list = np.zeros((len(self.pchi), ))
        for i in range(len(self.pchi)):
            pchi_curr = np.zeros((len(self.pchi),))
            pchi_curr[i] = pchi[i]
            grad_list[i] = self.get_cl(ell, pchi_curr, pchi) + self.get_cl(ell, pchi, pchi_curr)

        return grad_list


cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05,
                            h=0.7, n_s=0.95, sigma8=0.8,
                            transfer_function='bbks')

chi_grid = np.linspace(100, 4000., 50)
delta_chi = (chi_grid[1] - chi_grid[0])/2.

pz = np.repeat(1/(delta_chi*len(chi_grid)), 50)

breaks = chi_grid - delta_chi
breaks = np.append(breaks, breaks[-1] + 2.*delta_chi)

model_limber = LimberIntegration(cosmo, breaks, pz)
pz_new = np.zeros((len(pz), ))
pz_new[0] = pz[0]
pz_new = pz_new/np.sum(pz_new)
pz_new = pz_new/(2*delta_chi)
print(pz_new)
print(model_limber.evaluate_lens_kernel(pz_new))

def get_cl(val):
    pz_new = np.copy(pz)
    pz_new[0] = val
    return model_limber.evaluate_lens_kernel(pz_new)

print(nd.Derivative(get_cl)(pz[0]))

