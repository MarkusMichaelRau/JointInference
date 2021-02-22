import numpy as np
import pyccl as ccl
from scipy.integrate import simpson
import time
import numdifftools as nd
from scipy.stats import rv_histogram


class Histogram(rv_histogram):
    def __init__(self, w, breaks):
        self.breaks = breaks
        self.midpoints = np.array([np.mean([self.breaks[i+1], self.breaks[i]])
                          for i in range(len(w))])
        self.size_breaks = self.breaks[1:] - self.breaks[:-1]
        self.w = w
        super().__init__((self.w/self.size_breaks, self.breaks))



class LensingKernel(object):
    def __init__(self, histogram):
        self.histogram = histogram

    def distance_ratio(self, chi):
        return lambda chi_s: (chi_s - chi)/chi_s

    def evaluate(self, chi):
        return self.histogram.expect(self.distance_ratio(chi), lb=chi)


class LensKern(object):
    def __init__(self, lb, rb):
        """ All in comoving distances unit [Mpc]
        """
        self.lb = lb
        self.rb = rb

    def __call__(self, chi):
        if chi < self.lb:
            return (self.rb - self.lb) - chi * (np.log(self.rb) - np.log(self.lb))
        elif (chi >= self.lb) & (chi < self.rb):
            return (self.rb - chi) - chi*(np.log(self.rb) - np.log(chi))
        elif chi >= self.rb:
            return 0
        else:
            raise ValueError("Unknown value for \chi")


class Hist(object):
    def __init__(self, vec_weights, breaks):
        self.breaks = breaks
        #print(self.breaks)
        self.vec_weights = vec_weights
        assert np.isclose(np.sum(self.vec_weights), 1)
        self.delta = []
        self.list_kernel = []
        for i in range(len(self.breaks) - 1):
            self.list_kernel.append(LensKern(self.breaks[i], self.breaks[i+1]))
            self.delta.append(breaks[i+1] - breaks[i])
        self.delta = np.array(self.delta)


    def evaluate(self, chi):
        #print(chi)
        if (chi < self.breaks[0]) | (chi > self.breaks[-1]):
            print('evaluate')
            print(chi)
            print(self.breaks[0])
            print(self.breaks[-1])
            raise ValueError('Chi is outside the specified range.')

        list_lens_kernel = np.sum([el(chi)/self.delta[idx]*self.vec_weights[idx] for idx, el in enumerate(self.list_kernel)])
        return list_lens_kernel


    def gradient(self, chi):
        if (chi <= self.breaks[0]) | (chi >= self.breaks[-1]):
            raise ValueError('Chi is outside the specified range.')

        return np.array([self.list_kernel[index_bin](chi)/self.delta[index_bin] for index_bin in range(len(self.list_kernel))])

#test this

chi_grid = np.linspace(100, 4000., 10)
delta_chi = (chi_grid[1] - chi_grid[0])/2.
breaks = chi_grid - delta_chi
breaks = np.append(breaks, breaks[-1] + 2.*delta_chi)

w = np.repeat(1/len(chi_grid), len(chi_grid))

hist_new = Hist(w, breaks)

print(hist_new.evaluate(300))
print(hist_new.gradient(300))

hist = Histogram(w, breaks)

lens_kernel = LensingKernel(hist)
print(lens_kernel.evaluate(300))

def numerical_grad(w0):
    w_copy = np.copy(w)
    w_copy[0] = w0
    hist_new = Histogram(w_copy, breaks)
    lens_kernel_new = LensingKernel(hist_new)
    return lens_kernel_new.evaluate(300)


print(nd.Derivative(numerical_grad)(1/len(chi_grid)))


w_new = np.zeros((len(w), ))
w_new[0] = 1.
hist = Histogram(w_new, breaks)
lens_kernel = LensingKernel(hist)
#output = np.column_stack((np.linspace(100, 4000, num=100), np.array([lens_kernel.evaluate(el) for el in np.linspace(100, 4000, num=100)])))
#np.savetxt(X=output, fname='lens_kernel_gradient_small.dat')
print(lens_kernel.evaluate(300))


