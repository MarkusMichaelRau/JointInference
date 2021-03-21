import numpy as np
from astropy.table import Table


class TrapJoint(object):

    def __init__(self, n):
        self.n = n

    def get(self, breaks, grid):
        new_grid = []
        breaks_old = np.copy(breaks)
        grid_old = np.copy(grid)
        num_bins = len(breaks) - 1
        delta = breaks[1] - breaks[0] # assume equal binning
        if num_bins%self.n != 0:
            breaks_old = np.append(breaks, breaks[-1] + delta)
            grid_old = np.column_stack((grid_old, np.zeros((grid_old.shape[0],))))
            num_bins += 1
        print(grid_old.shape[1]/self.n)
        new_pz = np.zeros((grid_old.shape[0], int(grid_old.shape[1]/self.n)))
        for i in range(0, num_bins, self.n):
            for k in range(i, i+self.n-1):
                new_pz[:, int(i/self.n)] += (grid_old[:, k] + grid_old[:, k+1])/2.*delta
                #if k > int(grid_old.shape[1]/self.n):
                #    new_pz[:, i] += (grid_old[:, k] + 0)/2.*delta
                #else:
                #    new_pz[:, i] += (grid_old[:, k] + grid_old[:, k+1])/2.*delta

        return breaks_old[::self.n], new_pz

def convert_mid_to_breaks(mid):
    delta = mid[1] - mid[0]
    breaks = mid - delta/2.
    breaks = np.append(breaks, breaks[-1] + delta)
    return breaks

#fname='/verafs/scratch/phy200017p/share/HSC_Y1_tractwise_with_PDF/S16A_v2.0/HECTOMAP_tracts/16005_pz_pdf_mizuki.fits'
#fname_bins = '/verafs/scratch/phy200017p/share/HSC_Y1_tractwise_with_PDF/S16A_v2.0/pz_pdf_bins_mizuki.fits'
#data = np.array(Table.read(fname)['P(z)'])
#midpoints =  np.array(Table.read(fname_bins)['BINS'])
#breaks = convert_mid_to_breaks(midpoints)
#
#
#model_trapjoint = TrapJoint(2)
#
#breaks_new, grid_new = model_trapjoint.get(breaks, data)
#print(breaks)
#print(breaks_new)
#
#print(data[0])
#print(grid_new[0])
#assert len(breaks_new) == grid_new.shape[1] + 1
#
