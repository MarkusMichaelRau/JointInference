import numpy as np
from LimberIntegrator import *
from LensKern import *
from cov import *
from scipy.optimize import fmin_bfgs


class LossFunction(object):
    def __init__(self, cosmo, ell_vec, chi_grid, w_fid, ng, std_shape, fsky, cov_s):

        self.cosmo = cosmo
        self.ell_vec = ell_vec
        self.chi_grid = chi_grid

        delta_chi = (chi_grid[1] - chi_grid[0])/2.
        breaks = chi_grid - delta_chi
        self.breaks = np.append(breaks, breaks[-1] + 2.*delta_chi)
        self.w_fid = w_fid
        self.s_fid = transformation_logit(self.w_fid)

        self.fid_hist = Hist(self.w_fid, self.breaks)
        self.limb_inte = LimberIntegrator(self.cosmo, self.fid_hist, self.fid_hist)
        self.cl_fid = self.limb_inte.get_cl(self.ell_vec)

        wl_cov = WlCov(self.ell_vec, ng, std_shape, fsky)
        cov_cl = wl_cov.get_cov(self.cl_fid)

        self.inv_cov_cl = np.linalg.inv(cov_cl)
        self.inv_cov_s = np.linalg.inv(cov_s)

    def loss_function(self, s_vec):
        w_par = backtransform_logit(s_vec)
        hist_new = Hist(w_par, self.breaks)
        limb_inte = LimberIntegrator(self.cosmo, hist_new, hist_new)
        cl = limb_inte.get_cl(self.ell_vec)
        diff_cl = self.cl_fid  - cl
        diff_s = self.s_fid - s_vec
        return diff_cl.dot(self.inv_cov_cl.dot(diff_cl)) + diff_s.dot(self.inv_cov_s.dot(diff_s))

    def grad(self, s_vec):
        w_par = backtransform_logit(s_vec)
        gradient_logit = factory_derivative(w_par, s_vec)
        jacobian = np.zeros((len(s_vec), len(w_par)))
        for j in range(len(s_vec)):
            for i in range(len(w_par)):
                jacobian[j, i] = gradient_logit(i, j)

        hist_new = Hist(w_par, self.breaks)
        limb_inte = LimberIntegrator(self.cosmo, hist_new, hist_new)
        cl = limb_inte.get_cl(self.ell_vec)
        diff_cl = self.cl_fid - cl
        diff_s = self.s_fid - s_vec
        grad_limb = limb_inte.get_grad(self.ell_vec)
        return - self.inv_cov_cl.dot(diff_cl).dot(grad_limb).dot(jacobian.T) - self.inv_cov_s.dot(diff_s)

    def optimize(self, s_start):
        def callbackF(Xi):
            print('{0: 3.6f} {1: 3.6f}'.format( self.loss_function(Xi), self.grad(Xi)))


        xopt = fmin_bfgs(self.loss_function, s_start, fprime=self.grad, callback=callbackF, full_output=True,
              retall=False)
        return xopt


cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                    transfer_function='bbks')

w_s = np.loadtxt('mean_res_vec_s.dat')
cov_s = np.loadtxt('covariance_res_vec_s.dat')


ell_vec = np.arange(70,1000, 1)
chi_grid = np.linspace(100, 4000., 20)
w_fid = backtransform_logit(w_s)


cov_s = np.diag(np.ones((len(w_fid)-1,)))

ng = 567268937.282
fsky= 0.3
std_shape = 0.23
model_loss = LossFunction(cosmo, ell_vec, chi_grid, w_fid, ng, std_shape, fsky, cov_s)

xopt = model_loss.optimize(model_loss.s_fid+0.01*model_loss.s_fid)


print(xopt)
print(w_s)

with open('xopt_cosmo_hmcmc.pickle', 'wb') as outfile:
    pickle.dump(xopt, outfile)





