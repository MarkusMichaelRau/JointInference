import pyccl as ccl
import numpy as np
from scipy.stats import truncnorm
import numdifftools as nd


def get_nz_idx(idx, nz):
    pseudo_nz = np.zeros((len(nz), ))
    pseudo_nz[idx] = 1
    return pseudo_nz


def get_angular(n1):
    # Create new Cosmology object with a given set of parameters. This keeps track
    # of previously-computed cosmological functions
    cosmo = ccl.Cosmology(
                Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                    transfer_function='bbks')

    # Define a simple binned galaxy number density curve as a function of redshift
    z_n = np.linspace(0., 1., 200)
    n = np.ones(z_n.shape)
    n[0] = n1

    # Create objects to represent tracers of the weak lensing signal with this
    # number density (with has_intrinsic_alignment=False)
    lens1 = ccl.NewWeakLensingTracer(cosmo, dndz=(z_n, n))
    lens2 = ccl.NewWeakLensingTracer(cosmo, dndz=(z_n, n))

    # Calculate the angular cross-spectrum of the two tracers as a function of ell
    ell = np.arange(2, 10)
    cls = ccl.angular_cl(cosmo, lens1, lens2, ell)

    #derivative
    new_nz_idx = get_nz_idx(1, n)
    lens_deriv = ccl.NewWeakLensingTracer(cosmo, dndz=(z_n, new_nz_idx))
    cls_deriv = ccl.angular_cl(cosmo, lens_deriv, lens_deriv, ell)
    return cls, cls_deriv



def numerical_deriv(n_para):
    cosmo = ccl.Cosmology(
                Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                    transfer_function='bbks')

    # Define a simple binned galaxy number density curve as a function of redshift
    z_n = np.linspace(0., 1., 2000)
    n = (1 - n_para)*truncnorm.pdf(z_n,0, 1.0, 0.4, 0.01) + n_para*truncnorm.pdf(z_n,0, 1.0, 0.7, 0.01)

    # Create objects to represent tracers of the weak lensing signal with this
    # number density (with has_intrinsic_alignment=False)

    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))

    ell = np.arange(2, 1000, 10)
    import time
    start_time = time.time()
    for _ in range(1000):
        lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))

    # Calculate the angular cross-spectrum of the two tracers as a function of ell
        cls = ccl.angular_cl(cosmo, lens1, lens2, ell)
    print("--- %s seconds ---" % (time.time() - start_time))
    return cls


def analytical_deriv():

    ell = np.arange(70, 1000, 10)
    cosmo = ccl.Cosmology(
                Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                    transfer_function='bbks')

    # Define a simple binned galaxy number density curve as a function of redshift
    z_n = np.linspace(0., 1., 200)
    n = truncnorm.pdf(z_n,0, 1, 0.4, 0.02)
    #nz_deriv = truncnorm.pdf(z_n,0, 1, 0.7, 0.05) - truncnorm.pdf(z_n, 0, 1, 0.4, 0.05)
    lens_orig = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
    #lens_deriv = ccl.WeakLensingTracer(cosmo, dndz=(z_n, nz_deriv))
    cls_deriv = ccl.angular_cl(cosmo, lens_orig, lens_orig, ell)

    return 2*cls_deriv





#First test the numerical derivative
deriv_model = nd.Derivative(numerical_deriv, n=1)

#print(deriv_model(0.5))
print(analytical_deriv())


