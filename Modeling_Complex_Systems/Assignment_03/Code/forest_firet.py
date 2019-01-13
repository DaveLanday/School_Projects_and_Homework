#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation
import pygame
import pygame.surfarray as surfarray
from collections import Counter
import sys

#________________________________FUNCTIONS:_____________________________________
def get_neighbors( size, mat):
    """
        GET_NEIGHBORS: Create 2D array of lists, whereby a list represents
                       the neighbors corresponding to a site on the lattice.
                       There are no periodic boundaries, so edges of the
                       lattice have 3 neighbors, corners will have 2
                       neighbors, and every other site will have 4 neighbors

                       ARGS:
                            size: the size of the square Lattice.
                                  Example: size=20 for a lattice that is 20 x 20
                                  Type: Integer

                            mat: the 2D array you wish to evaluate.
                                 Type: (n,n) array
    """

    t = []
    for i in range(size):
        neighbors = []
        for j in range(size):

            #top left corner of grid:
            if i == 0 and j == 0:
                temp = [mat[i][j+1], mat[i+1][j]]

            #top right corner of grid:
            elif i == 0 and j == size-1:
                temp = [mat[i][j-1], mat[i+1][j]]

            #top row fixed and column not on a left or right edge:
            if i == 0 and j>0 and j < size - 1:
                temp = [mat[i][j-1], mat[i][j+1], mat[i+1][j]]

            #row not a top or bottom edge, left column fixed:
            elif i>0 and i < size -1 and j == 0:
                temp = [mat[i-1][j], mat[i][j+1], mat[i+1][j]]

            #bottom left corner of grid:
            elif i == size-1 and j == 0:
                temp = [mat[i-1][j], mat[i][j+1]]

            #botom right corner of grid:
            elif i == size - 1 and j == size - 1:
                temp = [mat[i][j-1], mat[i-1][j]]

            #row not a top or bottom edge, right most column fixed:
            elif i > 0 and i < size - 1 and j == size - 1:
                temp = [mat[i-1][j], mat[i][j-1], mat[i+1][j]]

            #bottom row fixed, column not a left or right edge:
            elif i == size - 1 and j>0 and j<size - 1:
                temp = [mat[i][j-1], mat[i-1][j], mat[i][j+1]]

            #otherwise, cell is in middle of grid:
            elif i != size - 1 and i != 0 and j != size - 1 and j!=0:
                temp = [mat[i][j-1], mat[i-1][j], mat[i][j+1], mat[i+1][j]]
            neighbors.append(temp)
        t.append(neighbors)
    return np.array(t)

def load_matrices(path_name):
    """
        LOAD_MATRICES: loads a numpy array of 2d arrays from a specified path.
                     Type: String representing name of a numpy file.
    """

    #load the array:
    arr = np.load(path_name)

    #return it:
    return arr

def unpack_necessary(list_of_arrays):
    """
        UNPACK_NECESSARY: turns a list of lists of arrays into one list of
                          arrays. Unpacking is necessary if you are loading an
                          asynchronous set of matrices.

                          ARGS:
                                list_of_arrays: a list of arrays that you load
                                                in after running Max's code for
                                                asynchronous updates

                          RETURNS:
                                a list of 2D arrays
    """

    #define a new array to house the matrices:
    singular_array_of_matrices = []

    #loop over array and unpack
    for mat in list_of_arrays:
        for arr in mat:
            singular_array_of_matrices.append(arr)


    #return list as an array:
    return np.array(singular_array_of_matrices)

def save_animation(mat, path):
    """
        SAVE_ANIMATION: saves an array or list of arrays as a .npy file

                      ARGS:
                            mat: a 2D arrays or list of 2D arrays.
                                 Type: array-like

                            path: file name for saved matrices. Type: String
                                  representing a path

                    RETURNS:
                            a .npy file
    """

    np.save(path, mat)

def simulate(landscape, time, z_max):
    """
        SIMULATE: generates a simulation of the forest fire using a
                  synchronous update technique.

                  ARGS:
                        landscape: a landscape object. Type: Class Landscape()

                        time: duration of time (# of frames) you wish to
                              simulate the forest fire. Type: int

                        z_max: maximum allowable height for any site on the
                               lattice. Type: int < 1/2 * self.L

               RETURNS:
                        None,
                        stores an array of length 'time', of landscapes
                        representing the state of each site at each
                        time step, in landscape.to_animate.
    """
    #create the landscape:
    landscape.mat =landscape.create_land()

    #define the topography:
    landscape.top = landscape.define_topography(z_max)

    #get the topgraphical neighbors:
    landscape.top_neighbors = get_neighbors(landscape.L, landscape.top)

    #calculate delta_z:
    landscape.delta_z = landscape.calculate_delta_z()

    #calculate number of neighbors at each site:
    landscape.N_s = landscape.calculate_N_s()

    #update the landscape for each time step:
    for t in range(time):

        #strike lightning:
        landscape.lightning_strike()

        #store the update:
        landscape.to_animate.append(landscape.mat)

        #get neighbor state information:
        landscape.mat_neighbors = get_neighbors(landscape.L, landscape.mat)

        #assess the sum of delta z:
        landscape.delta_z_sum = landscape.calc_sum_delta_z()

        #calculate number of neighors that are on fire:
        #landscape.N_f = landscape.calculate_N_f()

        #get gamma:
        landscape.gamma = landscape.get_gamma()

        #calculate the probability of a fire spreading to each site:
        landscape.p_f = landscape.calculate_p_fire(z_max)

        #turn fire to ash:
        landscape.turn_to_ash()
        landscape.to_animate.append(landscape.mat)

        #spread the fire:
        for i in range(landscape.L):
            for j in range(landscape.L):
                landscape.mat[i,j] = np.random.choice([1, landscape.mat[i,j]], p=[landscape.p_f[i,j], 1-landscape.p_f[i, j]])

        #attempt to grow trees back:
        landscape.grow_trees()

        #store the updates:
        landscape.to_animate.append(landscape.mat)

def to_png(to_animate, pathname):
    """
        TO_PNG: saves a png of a 2D array

                ARGS:
                    to_animate: list of 2D arrays. Type: List

                    pathname: Name of file to save. Type: String

                RETURNS:
                        None
    """
    fig = plt.figure(figsize=(8,8))

    for i in range(to_animate):
        plt.imshow(to_animate[i])
        plt.savefig(str(i)+pathname+'.png',dpi=72)

#____________________________CREATE THE LATTICE:________________________________

#make a random savanah/forest:
class Landscape(object):
    """
        LANDSCAPE: initializes a savannah with trees and grass
    """
    def __init__(self, L=20, land='random', topography='random',
                 to_animate=[], p_init=0.5, p_tree=0.6):
        """
            INIT: creates a Landscape object that is a matrix of and a matrix
                  that describes the topography of the landscape

                  ARGS:
                        L:      the dimension of the landscape (must be an even
                                number). Type: Integer

                        land: a string that initializes a matrix with trees. The
                              string indicates how the trees should be placed on
                              the lattice; i.e randomly,  or with some specified
                              probability. Type: String

                        topography: a string that initializes a matrix with
                                    height values that represents the change in
                                    topography. Type: String

                  OPTIONAL:

                        to_animate: an array of L x L matrices that have been
                                    pre-calculated and are ready to be animated.
                                    Type: numpy array of 2d matrices
        """
        #size of the lattice:
        if L%2 != 0:
            print("L must be an even number")
            sys.exit(1)
        else:
            self.L = L

        #how the lattice is generated:
        self.land = land
        self.topography = topography

        #landscape matrix:
        self.mat = np.array([])

        #topography matrix:
        self.top = np.array([])

        #keep track of the topographic neighbors:
        self.top_neighbors = np.array([])
        self.mat_neighbors = np.array([])

        #keep track of delta z:
        self.delta_z = np.array([])
        self.delta_z_sum = np.array([])

        #keep track of N_s:
        self.N_s = np.array([[]])

        #keep track of N_f:
        self.N_f = np.array([])

        #keep track of p_fire:
        self.p_fire = np.array([])

        #array of mats to animate:
        self.to_animate = to_animate

        #define the probability of a fire at a site (default: 0.5):
        self.p_init = p_init
        self.gamma = np.array([])

        #define the probability of a tree growing back:
        self.p_tree = p_tree

    def create_land(self):
        """
            CREATE_LAND: generates a landscape, LxL matrix, given the state of
                         parameter 'random'.

                         ARGS:
                                tree: the probability of seeing a tree at any
                                      site on the lattice. Type: array of length
                                      2.
        """

        #look at the requested landscape:

        #if random, then initialize a lattice with random tree placement:
        if self.land == 'random':
            mat = np.random.choice([0, 2],
                                   p=None,
                                   replace=True,
                                   size=(self.L, self.L))

        #otherwise, generate the lattice as dictated by p_tree:
        else:
            mat = np.random.choice([0,2],
                                   p=self.p_tree,
                                   replace=True,
                                   size=(self.L, self.L))

        #return the lattice:
        return mat

    def define_topography(self, max_h):
        """
            DEFINE_TOPOLOGY: creates a random, or uniform, terrain for which to
                             project the landscape onto

                             ARGS:
                                    max_h: maximum allowable height for any site
                                           on the lattice. Note, max_h cannot exceed
                                           half of the lattice dimension, L.
                                           Type: Integer
        """



        #create a range of heights:
        heights = np.arange(1, max_h + 1, 1)

        #generate a random topology:
        if self.topography == 'random':
            topography = np.random.choice(heights,
                                        p=None,
                                        replace=True,
                                        size=(self.L,self.L))

        #generate a sloped plane topography:
        elif self.topography == 'slant':

            #we will set the upper bound on maximum height to be self.L:
            try:
                #handle when less than 1/2*self.L and not a divisor of L:
                if max_h < self.L/2 and self.L%max_h != 0:
                    repeat = np.repeat(heights, self.L/max_h)
                    diff = self.L - len(repeat)
                    for i in range(diff):
                        topography = np.insert(repeat,0, 1)
                        topography = np.insert(topography,-1, max_h)
                    #fill the topography matrix:
                    topography = np.full( (self.L, self.L), topography)
                else:
                    topography = np.full( (self.L, self.L), np.repeat(heights,int(self.L/max_h)))
            except ValueError:
                return ValueError('We have set an upper bound on max_h. Choose a smaller value.')

        #generate a parabolic topography:
        elif self.topography == 'bowl':
            #create a blank matrix based on L:

            #max_h =1/2*self.L:
            try:
                if max_h == self.L/2:
                    temp = np.array([[1,1],[1,1]])
                    for i in heights[1:]:
                        temp = np.pad(temp,
                                      pad_width=((1,1),(1,1)),
                                      mode='constant',
                                      constant_values=i)
                    topography = temp
                elif max_h < self.L/2 and self.L%max_h !=0:
                    temp = np.array([[1,1],[1,1]])
                    for i in heights:
                        while len(heights) != self.L/2:
                            heights = np.insert(heights,0,i)

                    #re-sort the heights:
                    np.sort(heights)
                    for i in heights[1:]:
                        temp = np.pad(temp,
                                      pad_width=((1,1),(1,1)),
                                      mode='constant',
                                      constant_values=i)
                    topography = temp
                elif max_h < self.L/2 and self.L%max_h ==0:
                    multiplier = (self.L/max_h)/2
                    heights = np.repeat(heights,multiplier)

                    temp = np.array([[1,1],[1,1]])
                    for i in heights[1:]:
                        temp = np.pad(temp,
                                      pad_width=((1,1),(1,1)),
                                      mode='constant',
                                      constant_values=i)
                    topography = temp
            except ValueError:
                return ValueError('We have set an upper bound on max_h. Choose a smaller value.')


        return topography

    def show_topography(self):
        """
            SHOW_TOPOGRAPHY: creates a 3D visual of the initialized topography.

                             ARGS:
                                    NONE

                            RETURNS:
                                    3D surface plot representing the landscape's
                                    topography, based on the topography matrix.
        """

        X, Y = np.mgrid[:self.L, :self.L]

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        surf = ax.plot_surface(X,Y,self.top)
        plt.show()

    def calculate_delta_z(self):
        """
            CALCULATE_DELTA_Z: takes the topology matrix generated by the method
                               'define_topology()' and calculates a the change
                               in height for each of the surrounding neighboring
                               sites relative to every site on the lattice.

                               ARGS: None

                               Returns: a matrix of tuples of size 4 or less.
                                        Sites that don't have 4 neighbors will
                                        have tuples of smaller size.
        """

        #for each of the four heights, calculate the delta-z with respect to i,j
        delta = []

        #systematically change the heights to represent deltas:
        for i in range(self.L):
            temp = []
            for j in range(self.L):
                temp.append([self.top_neighbors[i][j][k] - self.top[i][j] for k in range(len(self.top_neighbors[i][j]))])
            delta.append(temp)

        return np.array(delta)

    def calc_sum_delta_z(self):
        """
            CALC_SUM_DELTA_Z: calculates the sum of all delta z surrounding a
                              site given the matrix caluclated by
                              'calculate_delta_z'
        """

        #generate a matrix of 0's:
        place_holder = np.zeros((self.L,self.L), dtype=int)

        #calculae the number of neighbors on fire:
        self.N_f = self.calculate_N_f()

        #average each delta z list and store it in place_holder:
        for i in range(self.L):
            for j in range(self.L):

                #keep value 0  if N_f[i, j] = 0:
                if self.N_f[i, j] != 0:
                    ones = []
                    for idx, el in enumerate(self.mat_neighbors[i, j]):
                        ones.append(self.delta_z[i,j][idx])

                    place_holder[i,j] = sum(ones)

        #return avg_delta_z:
        return place_holder

    def lightning_strike(self):
        """
            LIGHTNING_STRIKE: Choose a site on the lattice to throw lightning at
                              and thus ignite it >:D

                              ARGS:
                                    None

                              RETURNS:
                                     None
        """
        #randomly choose a site on the landscape to ignite:
        i = np.random.randint(0,self.L-1)
        j = np.random.randint(0,self.L-1)

        #assign it a value of 1 (1 for fire >:D):
        self.mat[i, j] = int(1)

    def calculate_N_f(self):
        """
            CALCULATE_N_F: Calculates the number of neghbors that are on fire

                           ARGS:
                                None

                        RETURNS:
                                a 2d array representing the number of neighbors
                                that are on fire, relative to each site in the
                                array
        """
        #create an array of zeros to be filled with N_f values:
        N_f = np.zeros((self.L, self.L), dtype=int)

        #populate each site in N_f with the correct number of bruning neighbors:
        for i in range(self.L):
            for j in range(self.L):
                nf = Counter(self.mat_neighbors[i, j])
                nf = nf[1]
                N_f[i, j] = nf

        #return N_f:
        return N_f

    def calculate_N_s(self):
        """
            CALCULATE_N_S: Calculate the number of neighbors each site on the
                           lattice has.

                           ARGS:
                                None

                       RETRURNS:
                                an array representing the number of neighbors at
                                each site (2D array).
        """

        #define a place to store N_s counts:
        N_s = np.zeros((self.L, self.L), dtype=int)

        #replace sites at N_s with corresponding number of neighbors:
        for i in range(self.L):
            for j in range(self.L):
                N_s[i, j] = len(self.top_neighbors[i, j])

        return N_s

    def get_gamma(self):
        """
            GET_GAMMA: retrieves the gamma matrix based on states that are 1;
                       on fire. Sites that are on fire will have
                       gamma = self.p_init, while sites that are burnt or trees
                       will have gamma = 0.

                       ARGS:
                            None

                    RETURNS:
                            None
        """

        #create a gamma:
        gamma = np.zeros((self.L, self.L), dtype=float)

        #find out which of the neighbors are on fire:
        for i in range(self.L):
            for j in range(self.L):
                if self.N_f[i, j] !=0:

                    #set the (i,j)th element of gamma to self.p_init:
                    gamma[i, j] = self.p_init

        #return gamma:
        return gamma

    def calculate_p_fire(self, z_max):
        """
            CALCULATE_P_FIRE: Calculates the probability of a fire at each site
                              based on the difference in height of each of its
                              neighbors and the number of neighbors that are
                              burning. The probability that a site will catch
                              fire, P, is dictated by the following equation:

 P = f(N_s, delta_z) = gamma + (1 - gamma) * ( sum_delta_z * N_f / z_max * N_s )

            N_s     : total number of neighboring sites

            sum_delta_z : the sum change in height of each neighbor with
                          respect to the site

            z_max   : the maximum allowable height on the lattice

            N_f     : the number of neighbors on fire

            gamma   : the probability of a site being on fire at any time,
                      self.gamma

                            ARGS:
                                z_max: the maximum allowable height on the
                                       lattice. Type: int

                            RETURNS:
                                an L x L matrix of probabilities
        """

        #get the p matrix:
        p = self.gamma + (1  - self.gamma) * ((self.delta_z_sum * self.N_f) / (z_max  * self.N_s))

        #return the 2D p array:
        return p

    def turn_to_ash(self):
        """
            TURN_TO_ASH: turns all sites on the lattice that are on fire
                         (state = 1) to ash (state = 0). Resets self.mat

                         ARGS:
                                None

                      RETURNS:
                                None
        """
        #turn all 1's into 0's:
        self.mat[self.mat == 1] = 0

    def grow_trees(self):
        """
            GROW_TREES: grows a tree at a site that has been turned to ash with
                        probability self.p_tree

                        ARGS:
                            None

                     RETURNS:
                            None
        """
        self.mat[self.mat == 0] = np.random.choice([0, 2], p=[1-self.p_tree, self.p_tree])


    def animate(self):
        """
            ANIMATE: takes an array of matrices representing the landscape a
                     different time steps and animates them as a simulation.

                     ARGS:
                            NONE

                    RETURNS:
                            Frame-by-frame animation of the simulation
        """

        # #initialize a pygame screen:
        # display = pygame.display.set_mode((500, 500),  pygame.RESIZABLE)
        # #sdisplay.toggle_fullscreen()
        # running = True
        #
        # while running:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             running = False
        #
        #     #generate  a surface:
        #     for i in self.to_animate:
        #         surf = pygame.surfarray.make_surface(i)
        #         display.blit(surf, (0, 0))
        #         pygame.display.update()
        # pygame.quit()
        pass


if __name__ == '__main__':
    z_max = 125
    landscape = Landscape(L=250, topography='slant')
    landscape.top = landscape.define_topography(z_max)
    #save_animation(landscape.top, '50x50_slant_zmax_25')
    #landscape.show_topography()
    #landscape.to_animate = load_matrices('bowl_mats.npy')

    # fig = plt.figure()
    # plot = plt.matshow(landscape.to_animate[0], fignum=0)
    #
    # def init():
    #     plot.set_data(lalandscape.to_animate[0])
    #     return [plot]
    #
    # def update(j):
    #     plot.set_data(landscape.to_animate[j])
    #     return [plot]
    #
    #
    # anim = FuncAnimation(fig, update, init_func = init, frames=50, interval = 30, blit=True)
    #
    # plt.show()
    #landscape.mat = landscape.create_land()
    # landscape.lightning_strike()
    # landscape.top = landscape.define_topography(z_max)
    # landscape.top_neighbors = get_neighbors(landscape.L, landscape.top)
    # landscape.mat_neighbors = get_neighbors(landscape.L, landscape.mat)

    # print(landscape.mat_neighbors)
    # landscape.delta_z = landscape.calculate_delta_z()
    # #print(landscape.mat[landscape.mat == 1])
    # landscape.delta_z_sum = landscape.calc_sum_delta_z()
    # landscape.N_s = landscape.calculate_N_s()
    # landscape.gamma = landscape.get_gamma()
    #print(landscape.gamma[landscape.gamma !=0])
    # landscape.p_f = landscape.calculate_p_fire(z_max)
    # print(landscape.p_f)
    simulate(landscape, 100, z_max)
    print(landscape.to_animate)
    #landscape.animate()
