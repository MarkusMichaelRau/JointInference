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
        grad_curr = self.loss_fnkt.grad(w)
        t = self.exact_line_search(w, grad_curr)
        w_new = w - t*grad_curr
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
    def __init__(self, loss_fnkt, batch_class, w_init=None, golden_limits=(0.0000000001, 0.001), batch_size=10000):
        self.batch_size = batch_size
        self.batch = batch_class
        super(StochProjGradDescent, self).__init__(loss_fnkt, w_init, golden_limits)


    def optim(self, epochs, batch_sizes=(10, 5) ):
        trace_w = [self.w_init]
        trace_loss = []
        trace_grad = []
        for it in range(epochs):
            for batch in self.self.batch.get_batch(batch_sizes):
                self.loss_fnkt.set_grid_vec(batch)
                w_new, grad_curr = self.curr_optim(trace_w[-1])
                trace_w.append(w_new)
                trace_loss.append(self.loss_fnkt.loss(trace_w[-1]))
                trace_grad.append(grad_curr)

        trace_w = np.array(trace_w)
        trace_loss = np.array(trace_loss)
        trace_grad = np.array(trace_grad)
        return trace_w, trace_loss, trace_grad


#if __name__ == '__main__':
#    #!/opt/packages/anaconda3/bin/python
#    from matplotlib import pyplot as plt
#    import sys
#    from LossFunktions import *
#    import numpy as np
#    import pyccl as ccl
#    import multiprocessing
#    from ProjGradDescent import *
#    from scipy.stats import norm
#    from scipy.stats import rv_histogram
#    import random
#
#    #####################################
#    # Define the redshift distributions #
#    #####################################
#
#    #reproducability
#
#    random.seed(10)
#
#    breaks = np.linspace(700, 3000, num=20)
#
#    midpoints = np.array([np.mean(breaks[i:i+2]) for i in range(len(breaks)-1)])
#    pz = norm.pdf(midpoints, 2000, 500)
#    hist_pz = rv_histogram((pz, breaks))
#    z_sample = hist_pz.rvs(size=100000)
#
#    true_nz = np.histogram(z_sample, breaks, density=True)
#
##    np.savetxt(X=np.column_stack((midpoints, true_nz[0])), fname='true_nz_test_prior.dat')
#
#    std = 40.0
#
#    mean_vec = np.array([norm.rvs(el, std) for el in z_sample])
#
#    grid_vec = np.zeros((len(mean_vec), len(midpoints)))
#

 #   for i in range(len(mean_vec)):
 #       print(i)
 #       for j in range(len(midpoints)):
 #           grid_vec[i, j] = norm.cdf(breaks[j+1], mean_vec[i], std) - norm.cdf(breaks[j], mean_vec[i], std)
 #
 #   np.savetxt(X=grid_vec, fname='test_data.dat')
#    grid_vec = np.loadtxt('test_data.dat')
#    model_phot_loss = PhotLoss(grid_vec, breaks)
#
#    model = StochProjGradDescent(model_phot_loss, batch_size=5000)
#
#    test = model.optim(200)
#    plt.figure()
#    plt.plot(test[1])
#    plt.show()
#    plt.figure()
#    plt.plot(midpoints, test[0][0])
#    plt.plot(midpoints, test[0][-1])
#    plt.show()
#    print(test)
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
