import numpy as np
from scipy.optimize import minimize_scalar

def projSimplex(vY, ballRadius, stopThr):
    """ from https://github.com/RoyiAvital/Projects/blob/master/Optimization/BallProjection/BallProjection.pdf
    """
    zeroVec = np.zeros((len(vY),))
    paramMu = np.min(vY) - ballRadius
    objFun = np.sum(np.maximum(vY - paramMu, zeroVec))-ballRadius
    while(np.abs(objFun) > stopThr):
        objFun = np.sum(np.maximum(vY - paramMu, zeroVec))-ballRadius
        df = np.sum(-((vY-paramMu)>0).astype(int))
        paramMu = paramMu - (objFun/(df+np.finfo(float).eps))
    return np.maximum(vY - paramMu, zeroVec)




class ProjGradDescent(object):
    def __init__(self, loss_fnkt, w_init=None, golden_limits=(0.000000001, 0.0001)):
        self.loss_fnkt = loss_fnkt
        if w_init is None:
            self.w_init = np.repeat(1./self.loss_fnkt.pi_dim, self.loss_fnkt.pi_dim)
        else:
            self.w_init = w_init

        self.golden_limits = golden_limits

    def exact_line_search(self, w, grad):
        f = lambda t: self.loss_fnkt.loss(projSimplex(w-t*grad, 1, 0.00001))
        min_res = minimize_scalar(f, bracket=self.golden_limits)
        if (min_res['x'] < self.golden_limits[0]) or (min_res['x'] > self.golden_limits[1]):
            print('Hit limit')
        return min_res['x']


    def optim(self, n_iter):
        trace_w = [self.w_init]
        trace_loss = []
        trace_grad = []
        for it in range(n_iter):
            print('iteration')
            print(it)
            trace_loss.append(self.loss_fnkt.loss(trace_w[-1]))
            print('loss')
            print(trace_loss[-1])

            grad_curr = self.loss_fnkt.grad(trace_w[-1])
            t = self.exact_line_search(trace_w[-1], grad_curr)
            print('stepsize')
            print(t)
            w_new = trace_w[-1] - t*grad_curr
            w_new = projSimplex(w_new, 1, 0.001)
            print('wnew')
            print(w_new)
            trace_w.append(w_new)
            trace_grad.append(grad_curr)
        trace_w = np.array(trace_w)
        trace_loss = np.array(trace_loss)
        trace_grad = np.array(trace_grad)
        return trace_w, trace_loss, trace_grad


#from LossFunktions import GaussNzPrior
#print('hihi')
#mean_pi = np.loadtxt('prior/mean_pi.dat')
#cov_pi = np.loadtxt('prior/cov_pi.dat')
#
#gauss_prior = GaussNzPrior(mean_pi, np.diag(np.diag(cov_pi)))
#
#grad_desc = ProjGradDescent(gauss_prior)
#
#result_w, result_loss, result_grad = grad_desc.optim(10000)
#
