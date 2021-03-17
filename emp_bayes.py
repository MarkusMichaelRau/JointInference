import numpy as np
from matplotlib import pyplot as plt 
from laplace_approx import *
from LossFunktions import *

trace_gamma = [1]
num_iter = 100


class GetMLGamma(object): 
    def __init__(self, grid_list, breaks): 
        self.grid_list = grid_list
        self.PhotLoss = PhotLoss(grid_list, breaks)

    
    def get_ml(self, gamma, num_iter=300): 
        smooth_prior = SmoothnessPrior(gamma, self.PhotLoss.pi_dim)
        model_joint = JointLossPrior(self.PhotLoss, smooth_prior)
        model_projgrad = ProjGradDescent(model_joint)

        result_w, result_loss, result_grad = model_projgrad.optim(num_iter) 
        return result_w[-1]


    def get_neg_hess(self, gamma, grid_like, max_value): 
        pi_trafo = transformation_logit(max_value)
        log_like_h = get_hessian_y(grid_like, max_value, pi_trafo)

        smooth_prior = SmoothnessPrior(gamma, self.PhotLoss.pi_dim)
        log_prior_h = get_hessian_logprior(max_value, smooth_prior.mat, gamma)

        inv_cov = -(log_like_h + log_prior_h)
        return inv_cov




if __name__ == '__main__': 
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

    breaks = np.linspace(700, 3000, num=20)
    
    midpoints = np.array([np.mean(breaks[i:i+2]) for i in range(len(breaks)-1)])
    pz = norm.pdf(midpoints, 2000, 500)
    hist_pz = rv_histogram((pz, breaks))
    z_sample = hist_pz.rvs(size=5000)

    true_nz = np.histogram(z_sample, breaks, density=True)

#    np.savetxt(X=np.column_stack((midpoints, true_nz[0])), fname='true_nz_test_prior.dat')

    std = 40.0

    mean_vec = np.array([norm.rvs(el, std) for el in z_sample])

    grid_vec = np.zeros((len(mean_vec), len(midpoints)))

    for i in range(len(mean_vec)):
        print(i)
        for j in range(len(midpoints)):
            grid_vec[i, j] = norm.cdf(breaks[j+1], mean_vec[i], std) - norm.cdf(breaks[j], mean_vec[i], std)


#    grid_vec = np.loadtxt('grid_vec.dat')

    trace_gamma = [10] 
    trace_ml = []
    trace_h_neg = []
    for it in range(20): 
        model_gamma = GetMLGamma(grid_vec, breaks)
        ml = model_gamma.get_ml(trace_gamma[-1])
        trace_ml.append(ml)
        h_neg = model_gamma.get_neg_hess(trace_gamma[-1], grid_vec, ml)
        trace_h_neg.append(h_neg)
        v, d, vh = np.linalg.svd(h_neg)


        samples_multiv = []
        nsamp = 1000
        mean_vecs = transformation_logit(ml)
        for _ in range(nsamp): 
            d_inv_sq = np.diag(1./np.sqrt(d))
            samp_vec = mean_vecs + v.dot(d_inv_sq.dot(np.random.normal(size=len(h_neg))))
            samples_multiv.append(samp_vec)
        
        samples_multiv = np.array(samples_multiv)
        samples_pi = np.array([backtransform_logit(el) for el in samples_multiv])
        plt.figure()
        plt.plot(midpoints, pz/np.sum(pz), color='black', label='True')
        plt.plot(midpoints, ml, label='ML')
        plt.plot(midpoints, np.mean(samples_pi, axis=0), label='Sample Mean', color='blue')
        plt.fill_between(midpoints, np.percentile(samples_pi, 10, axis=0), np.percentile(samples_pi, 90, axis=0),
        alpha=0.3, color='blue')
        plt.title(str(trace_gamma[-1]), fontsize=14)
        plt.savefig('plot_empirical_bayes'+str(it)+'.pdf', bbox_inches='tight')
        print(samples_pi)
        smooth_prior = SmoothnessPrior(trace_gamma[-1], grid_vec.shape[1])
        sum_term = np.sum([el.dot(smooth_prior.mat.dot(el)) for el in samples_pi])
        mean_term = 2./(grid_vec.shape[1]*nsamp)
        trace_gamma.append(1./(mean_term*sum_term))
        
        print('new gamma')
        print(1./(mean_term*sum_term))
        print(trace_gamma[-1])

    print(trace_gamma)
    output = {'ML': trace_ml, 'h_neg': trace_h_neg, 'gamma': trace_gamma, 'breaks': breaks, 'grid_vec': grid_vec}
    import pickle
    with open('output_empirical_bayes.pickle', 'wb') as outfile: 
        pickle.dump(output, outfile)
    np.savetxt(X=np.array(trace_gamma), fname='trace_gamma_new.dat')
#    print(np.mean(samples_pi, axis=0))
#    from matplotlib import pyplot as plt
#    plt.plot(midpoints, ml)
#    plt.plot(midpoints, np.mean(samples_pi, axis=0))
#    plt.fill_between(midpoints, np.percentile(samples_pi, 10, axis=0), np.percentile(samples_pi, 90, axis=0),
#    alpha=0.3)
#    plt.show()
#    print(samples_multiv)
#    print(v)
#    print(d)
#    print(vh)
#   # 
   # print('h')
   # print(h_neg)
   # np.savetxt(X=h_neg, fname='h_new.dat')
   # print(np.linalg.pinv(h_neg))
   # np.savetxt(X=np.linalg.pinv(h_neg), fname='MoorePenrose.dat')
   # print(np.sum(np.linalg.pinv(h_neg) - np.linalg.pinv(h_neg).T))


