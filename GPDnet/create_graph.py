'''first test for creating the graph
'''
import numpy as np
import torch
from torch_geometric import utils, data
import networkx as nx
import matplotlib.pyplot as plt
import timeit
from numba import jit


def draw_graph(nx_graph):
    fig, axes = plt.subplots(1,1,dpi=100)
    # pos=nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos=nx.kamada_kawai_layout(nx_graph), ax=axes, node_size=100, font_size=6, with_labels=True)
    plt.show()


@jit(nopython=True)
def create_graph(row, col):

    roi_size = row * col
    roi_indx = np.arange(roi_size)

    indx_out = []
    indx_in = []
    
    for j in range(col):
        if j == 0:
            for i in range(row-1):
                indx_out.append(roi_indx[i])
                indx_in.append(roi_indx[i+1])
                indx_out.append(roi_indx[i+1])
                indx_in.append(roi_indx[i])
            
        if j%2 == 1:
            for i in range(row-1):
                indx_out.append(roi_indx[j*row+i])
                indx_in.append(roi_indx[j*row+i+1])
                indx_out.append(roi_indx[j*row+i+1])
                indx_in.append(roi_indx[j*row+i])
            
            for i in range(row):

                if i != 0:
                    # create link with left-upper
                    indx_out.append(roi_indx[j*row+i])
                    indx_in.append(roi_indx[(j-1)*row+i-1])
                    indx_out.append(roi_indx[(j-1)*row+i-1])
                    indx_in.append(roi_indx[j*row+i])

                # create link with right-upper
                indx_out.append(roi_indx[j*row+i])
                indx_in.append(roi_indx[(j-1)*row+i])
                indx_out.append(roi_indx[(j-1)*row+i])
                indx_in.append(roi_indx[j*row+i])

        if j != 0 and j%2 == 0:
            for i in range(row-1):
                indx_out.append(roi_indx[j*row+i])
                indx_in.append(roi_indx[j*row+i+1])
                indx_out.append(roi_indx[j*row+i+1])
                indx_in.append(roi_indx[j*row+i])

            for i in range(row):

                # create link with left-upper
                indx_out.append(roi_indx[j*row+i])
                indx_in.append(roi_indx[(j-1)*row+i])
                indx_out.append(roi_indx[(j-1)*row+i])
                indx_in.append(roi_indx[j*row+i])

                if i != row-1:

                    # create link with right-upper
                    indx_out.append(roi_indx[j*row+i])
                    indx_in.append(roi_indx[(j-1)*row+i+1])
                    indx_out.append(roi_indx[(j-1)*row+i+1])
                    indx_in.append(roi_indx[j*row+i])

    return indx_out, indx_in


def wrapper(func, *args, **kwargs):
    """wrapper to measure functions execution time through timeit.

    :param function func: user defined function
    :param type *args: `*args` of function
    :param type **kwargs: `**kwargs` of function
    :return: wrapped function with no arguments needed.
    :rtype: wrapped_function

    """
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


if __name__ == '__main__': 

    row = 22
    col = 26
    wrap = wrapper(create_graph, row, col)
    print(timeit.timeit(wrap, number=100000))
    '''
    indx_out, indx_in = create_graph(row, col)
    # coo = np.array([indx_out, indx_in], np.uint8)
    coo = torch.tensor([indx_out, indx_in], dtype=torch.long)
    graph = data.Data(edge_index=coo)
    G = utils.to_networkx(graph)
    draw_graph(G)
    # nx.draw_kamada_kawai(G)
    print(indx_out, '\n', indx_in)
    '''
