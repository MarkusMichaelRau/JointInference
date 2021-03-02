import numpy as np
from LimberIntegrator import *
from LensKern import *
from cov import *
from scipy.optimize import fmin_bfgs


class LossFunctionWL(object):
    def __init__(self, cosmo_fid, cosmo_new, ell_vec, chi_grid, w_fid, ng, std_shape, fsky):

        self.cosmo_fid = cosmo_fid
        self.cosmo_new = cosmo_new
        self.ell_vec = ell_vec
        self.chi_grid = chi_grid

        self.pi_dim = len(self.chi_grid)

        delta_chi = (chi_grid[1] - chi_grid[0])/2.
        breaks = chi_grid - delta_chi
        self.breaks = np.append(breaks, breaks[-1] + 2.*delta_chi)
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


class PhotLoss(object):
    def __init__(self, grid_vec, breaks):
        self.grid_vec = grid_vec
        self.breaks= breaks
        self.delta = breaks[1] - breaks[0]
        self.pi_dim = len(breaks) - 1

    def loss(self, pi):
        return -np.sum(np.log((pi/self.delta).dot(self.grid_vec.T)+ sys.float_info.epsilon))

    def gradient(self, pi):
        denom=(pi/self.delta).dot(self.grid_vec.T)
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




