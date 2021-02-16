from pyccl.tracers import * 
from scipy.stats import rv_histogram


def mids_to_breaks(mids): 
    """
    Assume equal size binning
    """
    delta_z = (mids[1] - mids[0])/2.
    left_break = mids - delta_z
    breaks = np.array(left_break.tolist() + [left_break[-1] + 2*delta_z])
    return breaks

class NewWeakLensingTracer(Tracer):
    """Specific `Tracer` associated to galaxy shape distortions including
    lensing shear and intrinsic alignments within the L-NLA model.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
        dndz (tuple of arrays): A tuple of arrays (z, N(z))
            giving the redshift distribution of the objects. The units are
            arbitrary; N(z) will be normalized to unity.
        has_shear (bool): set to `False` if you want to omit the lensing shear
            contribution from this tracer.
        ia_bias (tuple of arrays, optional): A tuple of arrays
            (z, A_IA(z)) giving the intrinsic alignment amplitude A_IA(z).
            If `None`, the tracer is assumped to not have intrinsic
            alignments. Defaults to None.
        use_A_ia (bool): set to True to use the conventional IA
            normalization. Set to False to use the raw input amplitude,
            which will usually be 1 for use with PT IA modeling.
            Defaults to True.
    """
    def __init__(self, cosmo, dndz, has_shear=True, ia_bias=None,
                 use_A_ia=True):
        self._trc = []

        # we need the distance functions at the C layer
        cosmo.compute_distances()

        from scipy.interpolate import interp1d
        z_n, n = _check_array_params(dndz, 'dndz')
        breaks = mids_to_breaks(z_n)
        self._dndz = rv_histogram((n, breaks))
        #self._dndz = interp1d(z_n, n, bounds_error=False,
        #                      fill_value=0)

        if has_shear:
            kernel_l = get_lensing_kernel(cosmo, dndz)
            if (cosmo['sigma_0'] == 0):
                # GR case
                self.add_tracer(cosmo, kernel=kernel_l,
                                der_bessel=-1, der_angles=2)
            else:
                # MG case
                self._MG_add_tracer(cosmo, kernel_l, z_n,
                                    der_bessel=-1, der_angles=2)
        if ia_bias is not None:  # Has intrinsic alignments
            z_a, tmp_a = _check_array_params(ia_bias, 'ia_bias')
            # Kernel
            kernel_i = get_density_kernel(cosmo, dndz)
            if use_A_ia:
                # Normalize so that A_IA=1
                D = growth_factor(cosmo, 1./(1+z_a))
                # Transfer
                # See Joachimi et al. (2011), arXiv: 1008.3491, Eq. 6.
                # and note that we use C_1= 5e-14 from arXiv:0705.0166
                rho_m = lib.cvar.constants.RHO_CRITICAL * cosmo['Omega_m']
                a = - tmp_a * 5e-14 * rho_m / D
            else:
                # use the raw input normalization. Normally, this will be 1
                # to allow nonlinear PT IA models, where normalization is
                # already applied to the power spectrum.
                a = tmp_a
            # Reverse order for increasing a
            t_a = (1./(1+z_a[::-1]), a[::-1])
            self.add_tracer(cosmo, kernel=kernel_i, transfer_a=t_a,
                            der_bessel=-1, der_angles=2)


