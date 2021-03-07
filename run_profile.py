#!/opt/packages/anaconda3/bin/python

import sys
from LossFunktions import *
import numpy as np
import pyccl as ccl
import multiprocessing
from ProjGradDescent import *
from scipy.stats import norm
from scipy.stats import rv_histogram
import random

#####################################
# Define the redshift distributions #
#####################################

#reproducability

random.seed(10)

breaks = np.linspace(100, 4000, num=20)
midpoints = np.array([np.mean(breaks[i:i+2]) for i in range(len(breaks)-1)])
pz = norm.pdf(midpoints, 2000, 500)
hist_pz = rv_histogram((pz, breaks))
z_sample = hist_pz.rvs(size=100000)

true_nz = np.histogram(z_sample, breaks, density=True)
std = 20.0

mean_vec = np.array([norm.rvs(el, std) for el in z_sample])

grid_vec = np.zeros((len(mean_vec), len(midpoints)))

for i in range(len(mean_vec)):
    for j in range(len(midpoints)):
        grid_vec[i, j] = norm.cdf(breaks[j+1], mean_vec[i], std) - norm.cdf(breaks[j], mean_vec[i], std)

phot_loss = PhotLoss(grid_vec, breaks)


########################
# Define the Cosmology #
########################


cosmo_fid = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                        transfer_function='bbks')


ell_vec = (70,1000)
chi_grid = np.linspace(100, 4000., 20)


#####################
# Survey Parameters #
#####################

ng = 567268937.282
fsky= 0.3
std_shape = 0.23


def run_optim(sig8):

    cosmo_new = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=sig8, n_s=0.96,
                        transfer_function='bbks')


    model_ini = ProjGradDescent(phot_loss)

    result_w_ini, result_loss_ini, result_grad_ini = model_ini.optim(200)


    model_loss_wl = LossFunctionWL(cosmo_fid, cosmo_new, ell_vec, chi_grid, true_nz[0]/np.sum(true_nz[0]), ng, std_shape, fsky)

    model_joint = JointLossPhot(model_loss_wl, phot_loss)

    model_projgrad = ProjGradDescent(model_joint, result_w_ini[-1])
    result_w, result_loss, result_grad = model_projgrad.optim(200)
    np.savetxt(X=result_w, fname='result_w_proj_grad_descent_t1'+str(sig8)+'.dat')
    np.savetxt(X=result_loss, fname='result_loss_t1'+str(sig8)+'.dat')
    np.savetxt(X=result_grad, fname='result_grad_t1'+str(sig8)+'.dat')



#run_optim(0.70)

pool = multiprocessing.Pool(20)

zip(*pool.map(run_optim, np.linspace(0.70, 0.90, 20)))



