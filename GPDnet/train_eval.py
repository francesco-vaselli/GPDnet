'''train a graph net for classification
'''
import os.path as osp
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits
import pandas as pd
import torch
from torch.nn import Linear
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, BatchNorm, SAGEConv
from torch_geometric.nn import global_mean_pool, TopKPooling
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Dataset, Data, DataLoader
from create_graph import create_graph, draw_graph
from create_dataset_sf import GPDdataset_full
import scikitplot

class GCN(torch.nn.Module):
    """desperate attempt at creating a reasonable nn with pooling

    :param torch: [description]
    :type torch: [type]
    """    
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.batchn1 = BatchNorm(dataset.num_node_features)
        self.conv1 = SAGEConv(dataset.num_node_features, hidden_channels)
        self.batchn2 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.batchn3 = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.batchn4 = BatchNorm(hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)
        self.batchn5 = BatchNorm(hidden_channels)
        self.conv5 = SAGEConv(hidden_channels, hidden_channels)
        self.batchn6 = BatchNorm(hidden_channels)
        self.conv6 = SAGEConv(hidden_channels, hidden_channels)
        self.pool3 = TopKPooling(hidden_channels, ratio=0.8)
        self.batchn7 = BatchNorm(hidden_channels)
        self.conv7 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.batchn1(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.batchn2(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = self.batchn3(x)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.batchn4(x)
        x = self.conv4(x, edge_index)
        x = x.relu()
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x = self.batchn5(x)
        x = self.conv5(x, edge_index)
        x = x.relu()
        x = self.batchn6(x)
        x = self.conv6(x, edge_index)
        x = x.relu()
        # x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x = self.batchn7(x)
        x = self.conv7(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x


class GCN2(torch.nn.Module):
    """layman approach to graph convolutional networks: just 4 conv layers and readout!

    :param torch: [description]
    :type torch: [type]
    """    
    def __init__(self, hidden_channels):
        super(GCN2, self).__init__()
        self.batchn1 = BatchNorm(dataset.num_node_features)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.batchn2 = BatchNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.batchn3 = BatchNorm(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.batchn4 = BatchNorm(hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.batchn1(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.batchn2(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.batchn3(x)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.batchn4(x)
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x


if __name__=='__main__':

    dataset = GPDdataset_full('./data/')

    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print('\n')

    data = dataset[203]  # Get a graph object.

    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print('\n')
    
    # G = utils.to_networkx(data)
    # draw_graph(G)
    #shuffle and split
    torch.manual_seed(42)
    dataset = dataset.shuffle()

    train_dataset = dataset[:20000]
    val_dataset = dataset[20000:21000]
    test_dataset = dataset[513000:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(eval_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    network = GCN2(hidden_channels=64)
    print(network)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    network.to(device)

    n_epochs = 10
    train_losses = []
    train_counter = []
    val_losses = []
    val_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    log_interval = 50


    def train(epoch):
        network.train()
        for batch_idx, (data) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            out = network(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                (batch_idx*train_loader.batch_size) + ((epoch-1)*len(train_loader.dataset)))


    def val():
        network.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = network(data.x, data.edge_index, data.batch)
                val_loss += criterion(out, data.y).item()
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        val_loss /= np.rint(len(val_loader.dataset)/val_loader.batch_size)
        val_losses.append(val_loss)
        print('\nValidation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

    # train
    val()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        val()

    # training results
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('Categorical Cross Entropy')
    plt.show()

    # evaluate on test partition
    lst_pred = []
    lst_y = []

    for data in test_loader:
        data = data.to(device)
        batch_out = network(data.x, data.edge_index, data.batch).cpu().data.numpy()
        batch_truth = data.y.cpu().data.numpy()
        lst_pred.append(batch_out)
        lst_y.append(batch_truth)

    # concatenate batches
    predicted = Variable(torch.FloatTensor(np.concatenate(lst_pred)))
    # apply softmax
    predicted = torch.nn.functional.softmax(predicted, dim=1)
    predicted_prob = predicted.cpu().data.numpy()
    # teke most probable as prediction
    predicted_class = np.argmax(predicted_prob, axis=1)

    # concatenate batches
    true_val = list(chain.from_iterable(lst_y))

    # show results
    scikitplot.metrics.plot_confusion_matrix(true_val, predicted_class)
    plt.show()
    scikitplot.metrics.plot_precision_recall(true_val, predicted_prob)
    plt.show()
    scikitplot.metrics.plot_roc(true_val, predicted_prob)
    plt.show()