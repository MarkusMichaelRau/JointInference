import numpy as np
import sys
from decimal import *

def transformation_logit(pi_vec):
    return np.log(pi_vec[:-1]/(pi_vec[-1] + sys.float_info.epsilon) + sys.float_info.epsilon)


#def log_sum_exp_trick(s_vec): 
#    print(np.sum(s_vec + np.min(s_vec)))
#    return 1. + np.exp(np.log(np.sum(s_vec + np.min(s_vec))) - np.min(s_vec))

def backtransform_logit(s):
    s_vec = np.array([Decimal(el) for el in s])
    denom = Decimal(1.) + np.sum(np.exp(s_vec))
    vec_pi = np.exp(s_vec)/denom 
    vec_pi = vec_pi.tolist()
    vec_pi.append(1/denom)
    
    #print(np.array([np.float(el) for el in vec_pi]))
    return np.array([np.float(el) for el in vec_pi])



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




