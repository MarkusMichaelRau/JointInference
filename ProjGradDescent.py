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
    def __init__(self, loss_fnkt, w_init=None, golden_limits=(0.0000000001, 0.001)):
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
    
    def curr_optim(self, w): 
        grad_curr = self.loss_fnkt.grad(trace_w[-1])
        t = self.exact_line_search(trace_w[-1], grad_curr)
        w_new = trace_w[-1] - t*grad_curr
        w_new = projSimplex(w_new, 1, 0.001)

        return w_new, grad_curr

    def optim(self, n_iter):
        trace_w = [self.w_init]
        trace_loss = []
        trace_grad = []
        for it in range(n_iter):
            #grad_curr = self.loss_fnkt.grad(trace_w[-1])
            #t = self.exact_line_search(trace_w[-1], grad_curr)
            #w_new = trace_w[-1] - t*grad_curr
            #w_new = projSimplex(w_new, 1, 0.001)
            w_new, grad_curr = self.curr_optim(trace_w[-1]) 

            trace_w.append(w_new)
            trace_loss.append(self.loss_fnkt.loss(trace_w[-1]))
            trace_grad.append(grad_curr)

        trace_w = np.array(trace_w)
        trace_loss = np.array(trace_loss)
        trace_grad = np.array(trace_grad)
        return trace_w, trace_loss, trace_grad


class StochProjGradDescent(ProjGradDescent): 
    def __init__(self, loss_fnkt, w_init=None, golden_limits=(0.0000000001, 0.001), batch_size=5000): 
        self.batch_size = batch_size
        self.grid_vec = loss_fnkt.get_grid_vec()
        super(student, self).__init__(loss_fnkt, w_init, golden_limits)
   
    def iterate_minibatches(self, grid_vec, shuffle=False):
        if shuffle:
            indices = np.arange(grid_vec.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, grid_vec.shape[0] - self.batch_size + 1, self.batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            yield grid_vec[excerpt]

    def optim(self, epochs): 
        trace_w = [self.w_init]
        trace_loss = []
        trace_grad = []
        for it in range(epochs):
            for batch in iterate_minibatches(self.grid_vec, shuffle=True):
                self.loss_fnkt.set_grid_vec(batch)
                w_new, grad_curr = self.curr_optim(trace_w[-1])
                trace_w.append(w_new)
                trace_loss.append(self.loss_fnkt.loss(trace_w[-1]))
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
