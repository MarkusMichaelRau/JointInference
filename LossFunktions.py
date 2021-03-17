import numpy as np
from LimberIntegrator import *
from LensKern import *
from cov import *
from scipy.optimize import fmin_bfgs
from scipy.interpolate import InterpolatedUnivariateSpline
from tools import *
import abc 

class LossFunctionWL(object):
    def __init__(self, cosmo_fid, cosmo_new, ell_lims, z_breaks, nz_fid, ng, std_shape, fsky):
        assert len(ell_lims) == 2
        self.cosmo_fid = cosmo_fid
        self.cosmo_new = cosmo_new
        nbins_ell = 10
        self.ell_vec = np.logspace(np.log10(ell_lims[0]), np.log10(ell_lims[1]), 10)

        self.pi_dim = len(w_fid)

        self.breaks = chi_breaks
        self.w_fid = w_fid

        self.fid_hist = Hist(self.w_fid, self.breaks)
        self.limb_inte = LimberIntegrator(self.cosmo_fid, self.fid_hist, self.fid_hist)
        self.cl_fid = self.limb_inte.get_cl(self.ell_vec)

        wl_cov = WlCov(self.ell_vec, ng, std_shape, fsky)
        cov_cl = wl_cov.get_cov(self.cl_fid)

        self.inv_cov_cl = np.linalg.inv(cov_cl)

    def loss(self, pi):
        hist_new = Hist(pi, self.breaks)
        limb_inte = LimberIntegrator(self.cosmo_new, hist_new, hist_new)
        cl = limb_inte.get_cl(self.ell_vec)
        diff_cl = self.cl_fid  - cl
        return 0.5*diff_cl.dot(self.inv_cov_cl.dot(diff_cl))

    def grad(self, pi):

        hist_new = Hist(pi, self.breaks)
        limb_inte = LimberIntegrator(self.cosmo_new, hist_new, hist_new)
        cl = limb_inte.get_cl(self.ell_vec)
        diff_cl = self.cl_fid - cl
        grad_limb = limb_inte.get_grad(self.ell_vec)
        return - self.inv_cov_cl.dot(diff_cl).dot(grad_limb)


class GaussNzPrior(object):
    def __init__(self,  mean_pi, cov_pi):

        self.mean_pi = mean_pi
        self.cov_pi = cov_pi
        self.inv_cov_pi = np.linalg.inv(cov_pi)
        self.pi_dim = len(self.mean_pi)

    def loss(self, pi):
        diff_pi = (pi - self.mean_pi)
        return 0.5*diff_pi.dot(self.inv_cov_pi.dot(diff_pi))

    def grad(self, pi):
        diff_pi = (pi - self.mean_pi)
        return self.inv_cov_pi.dot(diff_pi)


class SmoothnessPrior(object): 
    def __init__(self, gamma, pi_dim): 
        self.gamma = gamma
        self.pi_dim = pi_dim
        self.mat = self.get_smooth_matrix(pi_dim) 
        if pi_dim < 3: 
            raise ValueError('Values not supported!')
        
    def loss(self, pi):
        return self.gamma*pi.dot(self.mat.dot(pi)) - self.pi_dim/2.*np.log(self.gamma)

    def grad(self, pi): 
        return 2.*self.gamma*self.mat.dot(pi)

    def get_smooth_matrix(self, dim):
        mat = np.tri(dim, dim, k=0) * 6. + np.tri(dim, dim, k=-1) * -4. + np.tri(dim, dim, k=1)* -4.+ np.tri(dim, dim,
        k=2) * 1. + np.tri(dim, dim, k=-2)*1.
        mat[0, 0] = 5.
        mat[-1, -1] = 5.
        return mat

   

class PhotLoss(object):
    def __init__(self, grid_vec, breaks):
        self.grid_vec = grid_vec
        self.breaks= breaks
        self.delta = breaks[1] - breaks[0]
        self.pi_dim = len(breaks) - 1
    
    def set_grid_vec(self, grid_vec): 
        self.grid_vec = grid_vec

    def get_grid_vec(self): 
        return self.grid_vec

    def loss(self, pi):
        return -np.sum(np.log((pi/self.delta).dot(self.grid_vec.T) + sys.float_info.epsilon))

    def grad(self, pi):
        denom=(pi/self.delta).dot(self.grid_vec.T) + sys.float_info.epsilon
        grad_res = np.zeros((self.pi_dim,))
        for k in range(len(grad_res)):
            int_term = self.grid_vec[:, k]/self.delta
            denom_term = 1./denom
            grad_res[k] = denom_term.dot(int_term)
        return -grad_res


class JointLossPrior(object):
    def __init__(self, WlLoss, GaussNzPrior):
        self.WlLoss = WlLoss
        self.GaussNzPrior = GaussNzPrior
        assert self.WlLoss.pi_dim == self.GaussNzPrior.pi_dim
        self.pi_dim = self.WlLoss.pi_dim

    def loss(self, pi):
        return self.WlLoss.loss(pi) + self.GaussNzPrior.loss(pi)

    def grad(self, pi):
        return self.WlLoss.grad(pi) + self.GaussNzPrior.grad(pi)


class JointLossPhot(object):
    def __init__(self, WlLoss, PhotLoss):
        self.WlLoss = WlLoss
        self.PhotLoss = PhotLoss
        assert self.WlLoss.pi_dim == self.PhotLoss.pi_dim
        self.pi_dim = self.WlLoss.pi_dim

    def loss(self, pi):
        return self.WlLoss.loss(pi) + self.PhotLoss.loss(pi)

    def grad(self, pi):
        return self.WlLoss.grad(pi) + self.PhotLoss.grad(pi)




