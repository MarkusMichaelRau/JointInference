import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from pyccl.background import comoving_radial_distance as chi_a
import pyccl as ccl
from scipy.stats import rv_histogram
from matplotlib import pyplot as plt


def interpolate(ell, cl):
    model = InterpolatedUnivariateSpline(np.log(ell), np.log(cl), ext=2)
    ell_new = np.arange(np.min(ell), np.max(ell), 1)
    cl = np.exp(model(np.log(ell_new)))
    return ell_new, cl


def convert_pz_pw(cosmo, breaks_z, nz): 
    a_vec = 1./(1. + breaks_z)
    chi_vec_breaks = chi_a(cosmo, a_vec)
    z_grid = np.linspace(np.min(breaks_z), np.max(breaks_z), num=100)
    model_jacobian = InterpolatedUnivariateSpline(z_grid, chi_a(cosmo, 1/(1 + z_grid)))
    list_jacobian = np.array([model_jacobian.integral(breaks_z[i], breaks_z[i+1]) for i in range(len(nz))])
    nz_w = nz*list_jacobian
    return chi_vec_breaks, nz_w/np.sum(nz_w)




#cosmo_fid = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
#                        transfer_function='bbks')
#
#samples = np.random.normal(1.0, 0.2, size=1000)
#
#hist, histbreaks = np.histogram(samples, 20)
#
#model_rv = rv_histogram((hist, histbreaks))
#grid_z = np.linspace(histbreaks[0], histbreaks[-1], 100)
#
#plt.plot(grid_z, model_rv.pdf(grid_z))
#plt.show()
#chi_vec_breaks, nz_w = convert_pz_pw(cosmo_fid, histbreaks, hist)
#print(len(chi_vec_breaks))
#print(len(nz_w))
#model_rv = rv_histogram((nz_w, chi_vec_breaks))
#grid_w = np.linspace(chi_vec_breaks[0], chi_vec_breaks[-1], 100)
#print(grid_w)
#print([model_rv.pdf(el) for el in grid_w])
#
#plt.plot(grid_w, [model_rv.pdf(el) for el in grid_w])
#plt.show()
#
#


