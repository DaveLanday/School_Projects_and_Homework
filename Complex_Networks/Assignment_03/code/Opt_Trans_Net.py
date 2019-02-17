#!/usr/bin/env python
# -*-coding: utf-8 -*-


#Opt_Trans_Net.py
#David W. Landay
#LAST UPDATED: 02-07-2019

import numpy as np
import networkx as nx
#import scipy as sp
#import pydot
import matplotlib.pyplot as plt
import sys,os

##################################FUNCTIONS#####################################
def plot_lattice(network, save=False):
    """
        PLOT_LATTICE: Draws the network on a hexagonal lattice.

                    ARGS:
                        network: a networkx graph. Type: networkx Graph object
                        save:    an option to save the drawing. Type: Boolean

                 RETURNS:
                        Outputs a graphical representation of the network.
    """

    #get the central node:
    center_node = (len(network.nodes) - 1) / 2

    #spring out from all of its neighbors:
    springFrom = [edges for edges in network.edges if center_node in edges]

    pos = nx.kamada_kawai_layout(network, center=(0,0), weight=None) #this gives me the hexagon!

    nx.draw_networkx_nodes(network, pos=pos, node_size=50, node_shape='h', node_color='k')
    widths = [np.sqrt(network[u][v]['weight']) for u,v in network.edges()]
    nx.draw_networkx_edges(network, pos=pos, center=springFrom, edge_color='gray', width=widths)
    plt.show()

def symmetrize(idx, val, mat):
    """
        SYMMETRIZE: stores a value at both the ij_th and ji_th indices in a
                    matrix.

                ARGS:
                    idx: a tuple representation of the ij_th positionselfself.
                         Type: tuple
                    val: the value to store at (i,j) and (j,i). Type: tuple
                    mat: a numpy array. Type: np.ndarray

            RETURNS:
                    NONE. Edits a matrix in place
    """
    mat[idx] = val
    mat[idx[1], idx[0]] = val

################################################################################

#create the system:
class Transport(object):
    """
        TRANSPORT:
    """

    def __init__(self, src=0, base=2):

        #location of the source node:
        self.src = src

        #number of nodes in the base:
        self.base = base

        #num_nodes in middle of hexagons:
        self.middle = self.base + (self.base - 1)

        #number of nodes in the lattice:
        self.num_nodes = 0

        #create a list of nodes:
        self.nodes = []

        #create a list of edges:
        self.edges = []

        #create a networkx graph:
        self.G = nx.Graph()

        #create the adjacency matrix:
        self.adj_mat = np.empty((self.num_nodes, self.num_nodes), dtype=float)

    def get_num_nodes(self):
        """
            GET_NUM_NODES: calculates the number of nodes in a hexagonal
                           lattice with all side lengths = b
        """

        self.num_nodes = self.base**2 + self.base*(self.base - 1) + (self.base - 1)**2

    def check_valid_src(self):
        """
            CHECK_VALID_SRC: given the base length of the src, and the input src
                            we want to check if that src (node number) exists
                            in the hexagonal lattice
        """
        if self.src in range(self.num_nodes):
            return True
        else:
            return False

    def get_nodes(self):
        """
            GET_NODES: Populates a list of all nodes
        """
        self.nodes = [i for i in range(self.num_nodes)]

    def get_edges(self, u_or_l):
        """
            GET_EDGES: gets tuple representation of edges from an upper or lower
                       section of a hexagon.

                       ARGS:
                            u_or_l: list of lists representing upper or lower
                                    section of a hexagon. Type: list

                    RETURNS:
                            a list of tuples representing edges
        """

        edges = []

        #get edges based on hexagonal geometry:
        for row,nxt in zip(u_or_l, u_or_l[1:]):

            #neighbors that are immediately to the right:
            horiz = [i for i in zip(row, row[1:])]
            vert  = [i for i in zip(row, nxt)]
            diag  = [i for i in zip(row, nxt[1:])]
            cent  = [i for i in zip(nxt, nxt[1:])]

            #update edges:
            edges.extend(horiz)
            edges.extend(vert)
            edges.extend(cent)
            edges.extend(diag)


        return edges

    def make_edge_list(self):
        """
            MAKE_EDGE_LIST: creates edges between edges in the network such that
                            the edges produce a hexagonal lattice.
        """

        #store edges:
        edge_list = []

        #indices we care about:
        nodesInEachRow = [i for i in range(self.base, self.middle + 1)]
        nodesInEachRow.extend(nodesInEachRow[::-1][1:])
        indices = np.cumsum(nodesInEachRow)

        #store setup lists:
        setup = [self.nodes[:indices[0]]]

        #use the #_of_nodes in each row to build the setup list:
        for idx,i in enumerate(indices):
            try:
                setup.append(self.nodes[i:indices[idx+1]])
            except IndexError:
                pass

        #store upper and lower half of triangle:
        upper = self.get_edges(setup[:self.base])
        lower = self.get_edges(setup[::-1][:self.base])

        #now we have to make the edgelist:
        edge_list.extend(upper+lower)
        self.edges = edge_list

    def make_adj(self):
        """
            MAKE_ADJ: constructs network and the adjacency matrix of the network
        """
        #make an adjacency matrix:
        self.G.add_nodes_from(self.nodes)
        self.G.add_edges_from(self.edges,weight=1)
        self.adj_mat = nx.adjacency_matrix(self.G).todense()

    def run_sim(self, i_0, gamma):
        """
            RUN_SIM: runs the Bohn and Magnasco network simulation on a
                     hexagonal lattice

                    ARGS:
                        gamma: the scale factor (tuneable param). Type: Float
                               between 0 and 1.
                RETURNS:
                        output of the simulation on a hexagonal lattice.
        """
        #Define paramters:
        Gamma = (2*gamma)/(gamma + 1)

        #define i_k, set the src current:
        i_k = np.full(self.num_nodes,
                      (-i_0/(self.num_nodes-1)))
        i_k[self.src] = i_0

        #initialize a random set of conductances and weights:
        k_kl = np.random.uniform(low=0, high=1000, size=len(self.G.edges))
        k_kl = (k_kl/np.sum(k_kl))**(1/gamma)

        #get the conductance matrix K:
        K = np.zeros(self.adj_mat.shape)

        #set the values in positions corresponding to adj_mat:
        for idx,edge in enumerate(self.G.edges):
            symmetrize(edge, k_kl[idx], K)

        #update n times:
        for i in range(30):

            #get Lambda:
            L = np.diag(np.sum(K, axis=1))

            #get A:
            A = (L - K)

            #calculate potential at nodes by solving system of linear equations:
            U_k = np.linalg.solve(A, i_k).reshape(1,self.num_nodes)
            #print(self.adj_mat.shape)
            #print(U_k.shape)

            #get the potentials among the links:
            U = np.multiply(U_k,self.adj_mat)
            U = U-U.T

            #get the currents at each edge:
            I_kl = np.multiply(K,U)

            #update i_k:
            i_k = np.sum(I_kl,axis=1)

            #update K:
            mask = I_kl !=0
            K = np.copy(I_kl)
            K[mask] = abs(K[mask])**-(Gamma-2)
            print(np.sum(np.power(np.abs(K),Gamma)))
        #set the I_kl to the weights of the edges:
        for u,v in self.G.edges():
            self.G[u][v]['weight'] = np.abs(I_kl[(u,v)])

        #plot the lattice:
        plot_lattice(self.G, False)

if __name__ == '__main__':

    #create a network:
    hex_net = Transport(91,8)

    #build the network:
    hex_net.get_num_nodes()

    #check if valid source:
    hex_net.check_valid_src()

    #get a list of nodes:
    hex_net.get_nodes()

    #create the edge list:
    hex_net.make_edge_list()
    #print(hex_net.edges)

    #make an adjacency matrix:
    hex_net.make_adj()

    #run the simulation:
    hex_net.run_sim(40, 0.5)
