import numpy as np

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
        self.vec_weights = vec_weights
        #assert np.isclose(np.sum(self.vec_weights), 1)
        self.delta = []
        self.list_kernel = []
        for i in range(len(self.breaks) - 1):
            self.list_kernel.append(LensKern(self.breaks[i], self.breaks[i+1]))
            self.delta.append(breaks[i+1] - breaks[i])
        self.delta = np.array(self.delta)


    def evaluate(self, chi):
        if (chi < self.breaks[0]) | (chi > self.breaks[-1]):
            raise ValueError('Chi is outside the specified range.')
        #print('evaluate')
        #print(chi)
        #print(self.breaks[0])
        #print(self.breaks[-1])

        list_lens_kernel = np.sum([el(chi)/self.delta[idx]*self.vec_weights[idx] for idx, el in enumerate(self.list_kernel)])
        return list_lens_kernel


    def gradient(self, chi):
        if (chi < self.breaks[0]) | (chi > self.breaks[-1]):
            raise ValueError('Chi is outside the specified range.')

        return np.array([self.list_kernel[index_bin](chi)/self.delta[index_bin] for index_bin in range(len(self.list_kernel))])



