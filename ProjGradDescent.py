import numpy as np 

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
    def __init__(self, loss, w_init=None, beta=0.5): 
        self.loss = loss
        if w_init is None:
            self.w_init = np.repeat(1./self.loss.pi_dim, self.loss.pi_dim)
        else:
            self.w_init = w_init
        self.beta = beta
        
    def backtrack_cond(self, t, w, grad): 
        return self.loss.loss(w - t*grad) > self.loss.loss(w) - 0.5*t*grad.dot(grad)
        
    def optim(self, n_iter):
        trace_w = [self.w_init]
        trace_loss = []
        trace_grad = []
        for it in range(n_iter): 
            print(it)
            grad_curr = self.loss.grad(trace_w[-1])
            t = 1
            while(self.backtrack_cond(t, trace_w[-1], grad_curr)):
                t = self.beta * t           
            print(t)
            print(grad_curr.dot(grad_curr)) 
            w_new = trace_w[-1] - t*grad_curr
            trace_w.append(projSimplex(w_new, 1, 0.001))
            curr_loss = self.loss.loss(trace_w[-1])
            print(curr_loss)
            trace_loss.append(curr_loss)
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
