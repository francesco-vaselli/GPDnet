'''create the graphs from the raw events
'''
import numpy as np
import torch
from torch_geometric import utils, data
import networkx as nx
import matplotlib.pyplot as plt
import timeit
from numba import jit


def draw_graph(nx_graph):
    """utility func to draw the graph

    :param nx_graph: a networkx graph
    :type nx_graph: [type]
    """    
    fig, axes = plt.subplots(1,1,dpi=100)
    nx.draw(nx_graph, pos=nx.kamada_kawai_layout(nx_graph), ax=axes, node_size=100, font_size=6, with_labels=True)
    plt.show()


# take advantage of just in time compilationm (>10x faster!!)
@jit(nopython=True)
def create_graph(row, col):
    """create an "empty" graph (only nodes and undirected edges, no features)
        in an "even-r" configuration (following the GPD workbook)
        The graph is outputted in COO notation; i.e. a list of nodes which
        send the link and another list of the recieving nodes.

    :param row: the number of rows activated at redout, it is supposed to be even for the GPD
    :type row: int
    :param col: the number of columns activated at redout, it is supposed to be even for the GPD
    :type col: int
    :return: two lists containing the senders and the recieving nodes
    :rtype: lists
    """
    roi_size = row * col
    roi_indx = np.arange(roi_size)

    indx_out = []
    indx_in = []
    
    for j in range(col):
        # treat col 0 as exception
        if j == 0:
            # create links along row
            for i in range(row-1):
                indx_out.append(roi_indx[i])
                indx_in.append(roi_indx[i+1])
                indx_out.append(roi_indx[i+1])
                indx_in.append(roi_indx[i])

        # select odd columns
        if j%2 == 1:
            # create links along row
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
                
        # select even columns
        if j != 0 and j%2 == 0:
            # create links along row
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
    
    indx_out, indx_in = create_graph(row, col)
    coo = torch.tensor([indx_out, indx_in], dtype=torch.long)
    graph = data.Data(edge_index=coo)
    G = utils.to_networkx(graph)
    draw_graph(G)
    print(indx_out, '\n', indx_in)

