import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import dirichlet
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import cauchy
from scipy.stats import lognorm
from scipy.special import erf
import pickle
import sys
from logit import *

def factory_derivative(pi_vec, pi_trafo):
    sum_pi_trafo = (1. + np.sum([np.exp(el) for el in pi_trafo]))
    def get_derivative(i, j):
        if (i == j) and (i < len(pi_vec) - 1):
            return pi_vec[i] - pi_vec[i]**2
        elif (i != j) and (i < len(pi_vec) - 1):
            return -pi_vec[i]*pi_vec[j]
        else:
            return -pi_vec[j]*1./sum_pi_trafo


    def get_secondderiv(alpha, i, j):
        if (i == j) and (i < len(pi_vec) - 1):
            deriv_ialpha = get_derivative(i, alpha)
            return deriv_ialpha - 2.*pi_vec[i]*deriv_ialpha
        elif (i != j) and (i < len(pi_vec) - 1):
            deriv_ialpha = get_derivative(i, alpha)
            deriv_jalpha = get_derivative(j, alpha)
            return -deriv_ialpha*pi_vec[j] - pi_vec[i]*deriv_jalpha
        else:
            deriv_jalpha = get_derivative(j, alpha)
            return (pi_vec[j]*pi_vec[alpha] - deriv_jalpha)*1./sum_pi_trafo

    return get_derivative, get_secondderiv


def get_hessian_y(grid_like, max_values, pi_trafo):
    deriv, secondderiv = factory_derivative(max_values, pi_trafo)
    hessian = np.zeros((len(pi_trafo), len(pi_trafo)))

    Ntot = grid_like.shape[1]

    deriv_mat = np.zeros((Ntot, len(hessian)))
    for dj in range(deriv_mat.shape[0]):
        for di in range(deriv_mat.shape[1]):
            deriv_mat[dj, di] = deriv(dj, di)

    second_deriv_mat = np.zeros((Ntot, len(pi_trafo), len(pi_trafo)))
    for j in range(second_deriv_mat.shape[0]):
        for alpha in range(second_deriv_mat.shape[1]):
            for z in range(second_deriv_mat.shape[2]):
                second_deriv_mat[j, alpha, z] = secondderiv(alpha, j, z)



    denominator1 = grid_like.dot(max_values)
    for z in range(len(hessian)):
        nominator1_first = deriv_mat[:, z].dot(grid_like.T)
        for alpha in range(len(hessian)):
            nominator1_second = deriv_mat[:, alpha].dot(grid_like.T)
            #second term
            nominator2 = second_deriv_mat[:, alpha, z].dot(grid_like.T)
            first_term = - (nominator1_first * nominator1_second)/(denominator1**2)
            second_term = nominator2/denominator1
            curr_sum_terms = first_term + second_term
            hessian[z, alpha] = np.sum(curr_sum_terms)

    return hessian



def get_hessian_logprior(max_values, Q, gamma): 
    pi_trafo = transformation_logit(max_values)

    deriv, secondderiv = factory_derivative(max_values, pi_trafo)
    h = np.zeros((len(pi_trafo), len(pi_trafo)))

    for alpha in range(Q.shape[0]-1): 
        for beta in range(Q.shape[1]-1): 
            second_deriv_vec = np.array([secondderiv(beta, i, alpha) for i in range(len(max_values))])
            first_term = second_deriv_vec.dot(Q.dot(max_values))

            alpha_deriv_vec = np.array([deriv(i, alpha) for i in range(len(max_values))])
            beta_deriv_vec = np.array([deriv(j, beta) for j in range(len(max_values))])
            second_term = alpha_deriv_vec.dot(Q.dot(beta_deriv_vec))
            h[beta, alpha] = -2.*gamma*(first_term + second_term)

    return h
            

