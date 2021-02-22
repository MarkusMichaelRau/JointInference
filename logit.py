import numpy as np
import sys

def transformation_logit(pi_vec):
    return np.log(pi_vec[:-1]/pi_vec[-1] + sys.float_info.epsilon)

def backtransform_logit(s_vec):
    denom = 1. + np.sum(np.exp(s_vec))

    vec_pi = np.exp(s_vec)/denom
    vec_pi = vec_pi.tolist()
    vec_pi.append(1/denom)
    return np.array(vec_pi)


def factory_derivative(pi_vec, pi_trafo):
    sum_pi_trafo = (1. + np.sum([np.exp(el) for el in pi_trafo]))
    def get_derivative(i, j):
        if (i == j) and (i < len(pi_vec) - 1):
            return pi_vec[i] - pi_vec[i]**2
        elif (i != j) and (i < len(pi_vec) - 1):
            return -pi_vec[i]*pi_vec[j]
        else:
            return -pi_vec[j]*1./sum_pi_trafo

    return get_derivative




