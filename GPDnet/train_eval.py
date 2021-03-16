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
from torch_geometric.nn import GCNConv, GraphConv, BatchNorm
from torch_geometric.nn import global_mean_pool, EdgePooling, SAGPooling
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Dataset, Data, DataLoader
from create_graph import create_graph, draw_graph
from create_dataset_sf import GPDdataset_full
import scikitplot


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(42)
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
        x = SAGPooling(x, edge_index)
        x = self.batchn3(x)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = SAGPooling(x, edge_index)
        x = self.batchn4(x)
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x


class GCN2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN2, self).__init__()
        # torch.manual_seed(42)
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
    # torch.manual_seed(43)
    dataset = dataset.shuffle()

    train_dataset = dataset[:20000]
    test_dataset = dataset[20000:21000]
    eval_dataset = dataset[513000:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(f'Number of eval graphs: {len(eval_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    network = GCN2(hidden_channels=64)
    print(network)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    network.to(device)

    '''
    
    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    for epoch in range(1, 20):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    '''
    n_epochs = 10
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    log_interval = 100


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


    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = network(data.x, data.edge_index, data.batch)
                test_loss += criterion(out, data.y).item()
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        test_loss /= np.rint(len(test_loader.dataset)/test_loader.batch_size)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('Categorical Cross Entropy')
    plt.show()

    lst_pred = []
    lst_y = []

    for data in eval_loader:
        data = data.to(device)
        batch_out = network(data.x, data.edge_index, data.batch).cpu().data.numpy()
        batch_truth = data.y.cpu().data.numpy()
        lst_pred.append(batch_out)
        lst_y.append(batch_truth)

    predicted = Variable(torch.FloatTensor(np.concatenate(lst_pred)))
    predicted = torch.nn.functional.softmax(predicted, dim=1)
    predicted_prob = predicted.cpu().data.numpy()
    predicted_class = np.argmax(predicted_prob, axis=1)

    true_val = list(chain.from_iterable(lst_y))

    scikitplot.metrics.plot_confusion_matrix(true_val, predicted_class)
    plt.show()
    scikitplot.metrics.plot_precision_recall(true_val, predicted_prob)
    plt.show()
    scikitplot.metrics.plot_roc(true_val, predicted_prob)
    plt.show()