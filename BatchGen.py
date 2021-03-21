import numpy as np
from astropy.table import Table, vstack


class PropSizeBatch(object):
    def __init__(self, list_files):
        self.list_files = list_files
        self.size_list = [len(Table.read(el)) for el in list_files]


    def get_batch(self, size_batches=(10, 10)):
        assert len(size_batches) == 2
        cluster_idx = np.random.choice(len(self.size_list), p=self.size_list/np.sum(self.size_list), replace=True, size=size_batches[0])
        list_pz = Table()
        for idx in range(len(self.size_list)):
            size_selected = np.sum(cluster_idx == idx)
            data_curr = Table.read(self.list_files[idx])
            sel_idx = np.random.choice(self.size_list[idx], size_selected, replace=False)
            if idx == 0:
                list_pz = data_curr[sel_idx]
            else:
                list_pz = vstack([list_pz, data_curr[sel_idx]])

        list_pz = np.array(list_pz)
        yield from self.iterate_minibatches(list_pz, size_batches[1], shuffle=True)

    def iterate_minibatches(self, grid_vec, batch_size, shuffle=True):
        if shuffle:
            indices = np.arange(grid_vec.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, grid_vec.shape[0] - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield grid_vec[excerpt]



