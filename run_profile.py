#!/opt/packages/anaconda3/bin/python  

import sys
from LossFunktions import *
import numpy as np 
import pyccl as ccl
import multiprocessing
from ProjGradDescent import *
#w_s = np.loadtxt('/verafs/home/mrau/LimberGrad/mean_res_vec_s.dat')
#cov_s = np.loadtxt('/verafs/home/mrau/LimberGrad/covariance_res_vec_s.dat')
#
cosmo_fid = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                        transfer_function='bbks')
    
#def run_optimization(sig8=0.8):
# 
#    cosmo_new = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=sig8, n_s=0.96,
#                        transfer_function='bbks')  
#    
#    ell_vec = np.arange(70,1000, 1)
#    chi_grid = np.linspace(100, 4000., 20)
#    w_fid = backtransform_logit(w_s)
#    
#    
#    cov_s = np.diag(np.ones((len(w_fid)-1,)))
#    
#    ng = 567268937.282
#    fsky= 0.3
#    std_shape = 0.23
#    model_loss = LossFunction(cosmo_fid, cosmo_new, ell_vec, chi_grid, w_fid, ng, std_shape, fsky, cov_s)
#    
#    return model_loss.optimize(model_loss.s_fid)
#



mean_pi = np.loadtxt('prior/mean_pi.dat')
cov_pi = np.loadtxt('prior/cov_pi.dat')

gauss_prior = GaussNzPrior(mean_pi, np.diag(np.diag(cov_pi)))
cosmo_fid = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                        transfer_function='bbks')

ell_vec = np.arange(70,1000, 10)
chi_grid = np.linspace(100, 4000., 20)


ng = 567268937.282
fsky= 0.3
std_shape = 0.23


def run_optim(sig8): 
 
    cosmo_new = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=sig8, n_s=0.96,
                        transfer_function='bbks')  

    model_loss_wl = LossFunctionWL(cosmo_fid, cosmo_new, ell_vec, chi_grid, mean_pi, ng, std_shape, fsky)

    model_joint = JointLoss(model_loss_wl, gauss_prior)


    result_w1 = np.loadtxt('result_w_proj_grad_descent_0_81.dat')

    model_projgrad = ProjGradDescent(model_joint, w_init=result_w1[-1], beta=0.3)
    result_w, result_loss, result_grad = model_projgrad.optim(140)
    np.savetxt(X=result_w, fname='result_w_proj_grad_descent_0_3'+str(sig8)+'.dat')
    np.savetxt(X=result_loss, fname='result_loss_0_3'+str(sig8)+'.dat')
    np.savetxt(X=result_grad, fname='result_grad_0_3'+str(sig8)+'.dat')

    model_projgrad = ProjGradDescent(model_joint, w_init=result_w[-1], beta=0.1)
    result_w, result_loss, result_grad = model_projgrad.optim(140)
    np.savetxt(X=result_w, fname='result_w_proj_grad_descent_0_1'+str(sig8)+'.dat')
    np.savetxt(X=result_loss, fname='result_loss_0_1'+str(sig8)+'.dat')
    np.savetxt(X=result_grad, fname='result_grad_0_1'+str(sig8)+'.dat')

    model_projgrad = ProjGradDescent(model_joint, w_init=result_w[-1], beta=0.03)
    result_w, result_loss, result_grad = model_projgrad.optim(140)
    np.savetxt(X=result_w, fname='result_w_proj_grad_descent_0_03'+str(sig8)+'.dat')
    np.savetxt(X=result_loss, fname='result_loss_0_03'+str(sig8)+'.dat')
    np.savetxt(X=result_grad, fname='result_grad_0_03'+str(sig8)+'.dat')


run_optim(0.8)

#pool = multiprocessing.Pool(10)    

#zip(*pool.map(run_optim, np.linspace(0.75, 0.85, 10)))
    


