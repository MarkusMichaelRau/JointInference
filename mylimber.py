import numpy as np
import pyccl as ccl
from scipy.stats import rv_histogram
from pyccl.background import comoving_radial_distance as chi_a
from scipy.interpolate import InterpolatedUnivariateSpline as inter
from pyccl.power import nonlin_matter_power #(cosmo, k, a
from scipy.integrate import simpson
import time
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

    def evaluate_lens_kernel(self):
        #chi is the midpoint
        diff_breaks = self.chi_grid_breaks[1:] - self.chi_grid_breaks[:-1]
        diff_logbreaks = np.log(self.chi_grid_breaks[1:]) - np.log(self.chi_grid_breaks[:-1])
        first_term = np.cumsum(self.pchi[::-1])[::-1]
        inner_vec = self.pchi/diff_breaks* diff_logbreaks
        second_term = self.comdist_grid*np.cumsum(inner_vec[::-1])[::-1]
        return first_term - second_term
        #kernel_values = np.zeros((len(self.comdist_grid),))
        #for idx, el in enumerate(self.comdist_grid):
        #    kernel_values[idx] = np.sum(self.pchi[idx:]/diff_breaks[idx:]*(diff_breaks[idx:] - self.comdist_grid[idx]*diff_logbreaks[idx:]))

        #return kernel_values
        #return self.pchi.expect(lambda chi_s: (chi_s - chi)/chi_s, lb=chi)

    def evaluate_lens_kernel_deriv(self):
        mat = np.zeros((len(self.comdist_grid), len(self.comdist_grid)))
        #rows derivative after parameters, column chi values
        for i in range(len(mat)):
            mat[i, :i+1] = 1. - self.comdist_grid[:i+1]*1./(self.chi_grid_breaks[i+1] - self.chi_grid_breaks[i])*(np.log(self.chi_grid_breaks[i+1]) - np.log(self.chi_grid_breaks[i]))

        return mat

    def get_cl(self, ell):
        power = np.array([nonlin_matter_power(self.cosmo, ell/chi, float(self.a_chi(chi))) for chi in self.comdist_grid])
        start_time = time.time()
        lens_kernel = self.evaluate_lens_kernel()
        a_chi = self.a_chi(self.comdist_grid)
        integrand_prefac = (9./4.) * self.om_mat**2 * (self.H0/self.c)**4 * (1./a_chi)**2
        integrand = integrand_prefac*power*lens_kernel*lens_kernel
        print("--- %s seconds ---" % (time.time() - start_time))
        result = simpson(integrand, self.comdist_grid)
        return result

    def get_cl_gradient(self, ell):
        power = np.array([nonlin_matter_power(self.cosmo, ell/chi, float(self.a_chi(chi))) for chi in self.comdist_grid])
        lens_kernel_deriv_mat = self.evaluate_lens_kernel_deriv()
        lens_kernel = self.evaluate_lens_kernel()
        a_chi = self.a_chi(self.comdist_grid)
        integrand_prefac = (9./4.) * self.om_mat**2 * (self.H0/self.c)**4 * (1./a_chi)**2
        output_deriv = np.zeros((len(self.comdist_grid),))
        for i in range(len(self.comdist_grid)):
            integrand = integrand_prefac*power*(lens_kernel*lens_kernel_deriv_mat[i, :] + lens_kernel_deriv_mat[i, :]*lens_kernel)
            output_deriv[i] = simpson(integrand, self.comdist_grid)
        return output_deriv


#test code

cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05,
                            h=0.7, n_s=0.95, sigma8=0.8,
                            transfer_function='bbks')

chi_grid = np.linspace(100, 4000., 50)
delta_chi = (chi_grid[1] - chi_grid[0])/2.
breaks = chi_grid - delta_chi
breaks = np.append(breaks, breaks[-1] + 2.*delta_chi)

#model_pchi = rv_histogram((np.repeat(1/4000., 30), breaks))

model_limber = LimberIntegration(cosmo, breaks, np.repeat(1/4000., 50))

output = np.column_stack((np.linspace(70, 1000, 100), [model_limber.get_cl(el) for el in np.linspace(70, 1000, 100)]))

np.savetxt(X=output, fname='cl_test.dat')
output_cl_deriv = np.column_stack((np.linspace(70, 1000, 100), [model_limber.get_cl_gradient(el) for el in np.linspace(70, 1000, 100)]))

np.savetxt(X=output_cl_deriv, fname='cl_test_deriv.dat')


