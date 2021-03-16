''' create pytorch_geometric dataset
'''
import os.path as osp
import numpy as np
from astropy.io import fits
import pandas as pd
import torch
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Dataset, Data
from create_graph import create_graph, draw_graph


class GPDdataset_full(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GPDdataset_full, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['flat_rnd0.fits']

    @property
    def processed_file_names(self):
        return ['data_1.pt']

    @property
    def num_classes(self):
        return 3

    def download(self):
        pass

    def process(self):

        raw_data_index = [0, 1, 10, 11, 12, 13, 14]
        j = 0

        for indx in raw_data_index:

            hdu = fits.open(osp.join(self.raw_dir, f'flat_rnd{indx}.fits'))
            events_data = hdu['EVENTS'].data

            xmin = events_data['MIN_CHIPX']
            xmax = events_data['MAX_CHIPX']
            ymin = events_data['MIN_CHIPY']
            ymax = events_data['MAX_CHIPY']
            num_cols = np.array((xmax - xmin + 1), dtype=np.int64)
            num_rows = np.array((ymax - ymin + 1), dtype=np.int64)
            roi_size = np.array(events_data['ROI_SIZE'], dtype=np.int64)
            assert np.array_equal(num_cols * num_rows, roi_size)
            pha = events_data['PIX_PHAS']

            ground_truth = np.array(hdu['MONTE_CARLO'].data['ABS_Z'], dtype=np.float64)
            energy = np.array(hdu['MONTE_CARLO'].data['ENERGY'], dtype=np.float64)

            
            for i in range(0, 200000):

                if energy[i] <= 8.9 and energy[i] >= 4.0:

                    if ground_truth[i] <= 0.86:
                        y_gt = 0
                    elif ground_truth[i] >= 10.8:
                        y_gt = 2
                    else:
                        y_gt = 1

                    indx_out, indx_in = create_graph(num_rows[i], num_cols[i])
                    edge_index = torch.tensor([indx_out, indx_in], dtype=torch.long)

                    x = torch.tensor([[k] for k in pha[i]], dtype=torch.float)
                    y = torch.tensor([y_gt], dtype=torch.long)

                    data = Data(x=x, edge_index=edge_index, y=y)
                    torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(j)))
                    j += 1
            
            print(f'processed {j} events')

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

    def len(self):
        return 513272


class GPDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GPDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['flat_rnd0.fits']

    @property
    def processed_file_names(self):
        return ['data_1.pt']

    @property
    def range(self):
        return 200000

    @property
    def num_classes(self):
        return 3

    def download(self):
        pass

    def process(self):

        # data_list = []

        hdu = fits.open(osp.join(self.raw_dir, 'flat_rnd0.fits'))
        events_data = hdu['EVENTS'].data

        xmin = events_data['MIN_CHIPX']
        xmax = events_data['MAX_CHIPX']
        ymin = events_data['MIN_CHIPY']
        ymax = events_data['MAX_CHIPY']
        num_cols = np.array((xmax - xmin + 1), dtype=np.int64)
        num_rows = np.array((ymax - ymin + 1), dtype=np.int64)
        roi_size = np.array(events_data['ROI_SIZE'], dtype=np.int64)
        assert np.array_equal(num_cols * num_rows, roi_size)
        pha = events_data['PIX_PHAS']

        ground_truth = np.array(hdu['MONTE_CARLO'].data['ABS_Z'], dtype=np.float64)
        energy = np.array(hdu['MONTE_CARLO'].data['ENERGY'], dtype=np.float64)

        j = 0
        for i in range(0, self.range):

            if ground_truth[i] <= 0.86:
                y_gt = 0
            elif ground_truth[i] >= 10.8:
                y_gt = 2
            else:
                y_gt = 1

            if energy[i] <= 8.9 and energy[i] >= 4.0:
                y_e = 1
            if energy[i] >= 8.9:
                y_e = 2
            else:
                y_e = 0

            indx_out, indx_in = create_graph(num_rows[i], num_cols[i])
            edge_index = torch.tensor([indx_out, indx_in], dtype=torch.long)

            x = torch.tensor([[k] for k in pha[i]], dtype=torch.float)
            y = torch.tensor([y_gt], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(j)))
            j += 1
            # data_list.append(data)

        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

    def len(self):
        return self.range



if __name__ == '__main__':

    dataset = GPDdataset_full('data/')
    '''
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')

    data = dataset[123]  # Get the first graph object.

    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    
    G = utils.to_networkx(data)
    draw_graph(G)
    '''