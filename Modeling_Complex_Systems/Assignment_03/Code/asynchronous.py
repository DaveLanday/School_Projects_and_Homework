# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:22:42 2018
@author: mgreen13
"""
import numpy as np
from collections import Counter
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from scipy import signal
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
plt.ioff()
class Cell():

    # constructor for cell

    def __init__(self,x,y,z,state):
        self.x = x
        self.y = y
        self.position = x,y
        self.z = z
        self.state = state
        self.visited = False

        self.dz1 = None
        self.dz2 = None
        self.dz3 = None
        self.dz4 = None
        self.nStates = 2
        self.p = []

    def getNState(self,landscape):
        i = self.x
        j = self.y

        try:
            n1 = landscape[i-1,j].getState()
        except:
            IndexError
        try:
            n2 = landscape[i,j+1].getState()
        except:
            IndexError
        try:
            n3 = landscape[i+1,j].getState()
        except:
            IndexError
        try:
            n4 = landscape[i,j-1].getState()
        except:
            IndexError
        # Build case for each border area, upper/lower left corner, upper/lower right corner
        # All Four borders
        # Upper Left Corner (No n1 or n4)
        if i == 0 and j == 0:
            return(n2,n3)
        # Upper right corner(no n1 or n2)
        elif i==0 and j==len(landscape)-1:
            return(n3,n4)
        # Lower left corner(no n3,n4)
        elif i == len(landscape)-1 and j == 0:
            return(n1,n2)
        # Lower right corner(no n2 or n3)
        elif i == (len(landscape)-1) and j == (len(landscape)-1):
            return(n1,n4)
        # On top of matrix
        elif i ==0:
            return(n2,n3,n4)
        # Bottom of matrix
        elif i == len(landscape)-1:
            return(n1,n2,n4)
        # Right side of matrix
        elif j == len(landscape)-1:
            return(n1,n3,n4)
        # Left Side of matrix
        elif j == 0:
            return(n1,n2,n3)
        else:
            return(n1,n2,n3,n4)

    def getN(self,landscape):
        i,j = self.getPosition()
        # TRY EXCEPT BLOCK TO ATTEMPT TO ASSIGN NEIGHBOR LOCATIONS
        try:
            n1 = landscape[i-1,j].getPosition()
        except:
            IndexError
        try:
            n2 = landscape[i,j+1].getPosition()
        except:
            IndexError
        try:
            n3 = landscape[i+1,j].getPosition()
        except:
            IndexError
        try:
            n4 = landscape[i,j-1].getPosition()
        except:
            IndexError
            # Build case for each border area, upper/lower left corner, upper/lower right corner
        # All Four borders

        # Upper Left Corner (No n1 or n4)
        if i == 0 and j == 0:
            return(n2,n3)
        # Upper right corner(no n1 or n2)
        elif i==0 and j==len(landscape)-1:
            return(n3,n4)
        # Lower left corner(no n3,n4)
        elif i == len(landscape)-1 and j == 0:
            return(n1,n2)
        # Lower right corner(no n2 or n3)
        elif i == (len(landscape)-1) and j == (len(landscape)-1):
            return(n1,n4)
        # On top of matrix
        elif i ==0:
            return(n2,n3,n4)
        # Bottom of matrix
        elif i == len(landscape)-1:
            return(n1,n2,n4)
        # Right side of matrix
        elif j == len(landscape)-1:
            return(n1,n3,n4)
        # Left Side of matrix
        elif j == 0:
            return(n1,n2,n3)
        else:
            return(n1,n2,n3,n4)

    # getter for state of cell
    def getState(self):
        return self.state

    #setter for state of cell
    def setState(self,state):
        self.state = state

    # Get position of cell in matrix
    def getPosition(self):
        return(self.x,self.y)

    # Get height of cell
    def getZ(self):
        return(self.z)


    # Set dz values between site and neighbouring nodes
    def setDz(self,landscape):
        #INITIALIZD DELZ AS NONE
        self.dz2 = None
        self.dz4 = None
        self.dz1 = None
        self.dz3 = None

        # Exception for higher borders of grid
        try:
            self.dz1 = landscape[i,j].getZ() - landscape[i+1,j].getZ()
            self.dz3 = landscape[i,j].getZ() - landscape[i,j+1].getZ()
        except:
            IndexError
        # Exception for lower borders of grid
        if i!= 0:
            self.dz2 = landscape[i,j].getZ() - landscape[i-1,j].getZ()
        if j!= 0:
            self.dz4 = landscape[i,j].getZ() - landscape[i,j-1].getZ()

    def getDz(self):
        return(self.dz1,self.dz2,self.dz3,self.dz4)

    def getDzSum(self,landscape):
        nbs = self.getN(landscape)
        zs = []
        for n in nbs:
            if landscape[n].state == 1:
                zs.append(self.z - landscape[n].getZ())
        avgDz= np.sum(zs)
        return(avgDz)

def stateMat(landscape):

    """
    Retrieve matrix of states from landscape
    """
    mat = np.zeros([len(landscape),len(landscape)])
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            mat[i,j] = landscape[i,j].state
    return(mat)

def zMat(landscape):
    mat = np.zeros([len(landscape),len(landscape)])
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            mat[i,j] = landscape[i,j].z
    return(mat)


def  growTree(p,landscape):
    """GROW TREE AT I,J WITH PROBABILITY P"""
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            if landscape[i,j].getState() == 0:
                if np.random.rand(1) < p:
                    landscape[i,j].setState(2)
    return(landscape)



def lightStrike(land,gamma, zMax,maxN):

    """
    Asynchronous update fire strike
   INPUTS
       1) Landscape matrix: matrix of forest cell objects
       2) Gamma: probability space partition
       3) zMax: Maximum height of cell in forest
       4) maxN: maximum number of neighrbors a cell is allowed to have

   OUTPUTS
       1) Updated landscape matrix
       """
    stateMaps = []
    unvisited = []
    fired = []
    starting_z = []
    for k in range(1):
        i = np.random.randint(0,len(land))
        j = np.random.randint(0,len(land))
        #SET STATE OF CELL TO FIRE
        landscape[i,j].setState(1)
        starting_z.append(landscape[i,j].z)
        unvisited.extend(land[i,j].getN(land))
        fired.append((i,j))
    while len(unvisited) != 0:
        for indn,n in enumerate(unvisited):
            nStates = land[n].getNState(land)
            dzSum = land[n].getDzSum(land)
            nF = Counter(nStates)
            nF = nF[1]
            nS = len(nStates)
            for nNeigh in land[n].getN(land):
                if land[nNeigh].getState() == 2 and nNeigh != (i,j):
                    unvisited.append(nNeigh)
            pFire = gamma + (1-gamma)*(dzSum*nF)/(nS*zMax)

            land[0,0].p.append(pFire)
            if np.random.rand(1)<pFire:
                land[n].setState(1)
                unvisited.pop(indn)
                fired.append(n)
                mapS = stateMat(land)
                stateMaps.append(mapS)

    for fire in fired:
        land[fire].setState(0)
    stateMaps.append(stateMat(land))

    return(land,stateMaps,fired,starting_z)

bowl = np.load("150x150_bowl_z_10.npy")
hill = np.load("150x150_slant_z_10.npy")
hillsmall = np.load("50x50_slant_zmax_25.npy")
bowlSmall = np.load("50x50_bowl_zmax_10.npy")
#zVals= np.random.randint(1,10,[N,N])
zVals = bowlSmall
N = len(zVals)
landscape = np.ndarray([N,N],dtype = Cell)
for i,ik in enumerate(zVals):
    for j,jk in enumerate(ik):
        z = zVals[i,j]
        a = Cell(i,j,z,0)
        landscape[i,j] = a

# SET HEIGHTS OF CELLS
for i in list(range(len(landscape))):
            for j in list(range(len(landscape))):
                landscape[i][j].setDz(landscape)
statemaps = []
firedMaps = []
startZList = []
for i in range(20):
    landscape = growTree(.8,landscape)
    landscape,statemap,fired,startZ= lightStrike(landscape,.8,25,4)
    statemaps.append(statemap)
    firedMaps.append(fired)
    startZList.append(startZ)


masterStateMaps = []
for u in statemaps:
    for maps in u:
        masterStateMaps.append(maps)


# # ------------------------------ANALYSIS --------------------------------
# fireTotal = []
# for i in range(100):
#     fireTotal.append([])

# for ind,u in enumerate(firedMaps):
#     zss = []
#     for v in u:
#         zss.append(landscape[v].z)
#     fireTotal[ind] = zss
# s1 = []
# s2 = []
# s3 = []
# s4 = []
# s5 = []
# s6 = []
# s7 = []
# s8 = []
# s9 = []
# s10 = []
# for i in range(100):
#     if startZList[i] == [1]:
#         s1.append(fireTotal[i])
#     if startZList[i] ==[2]:
#         s2.append(fireTotal[i])
#     if startZList[i] ==[13]:
#         s3.append(fireTotal[i])
#     if startZList[i] ==[4]:
#         s4.append(fireTotal[i])
#     if startZList[i] ==[5]:
#         s5.append(fireTotal[i])
#     if startZList[i] ==[6]:
#         s6.append(fireTotal[i])
#     if startZList[i] ==[7]:
#         s7.append(fireTotal[i])
#     if startZList[i] ==[8]:
#         s8.append(fireTotal[i])
#     if startZList[i] ==[9]:
#         s9.append(fireTotal[i])
#     if startZList[i] ==[10]:
#         s10.append(fireTotal[i])

# plt.figure(figsize = (15,10) )
# for path in fireTotal:
#     plt.plot(np.linspace(1,len(path),len(path)),path,".")



# plt.figure(figsize = (15,10))
# plt.plot(np.linspace(1,len(s4[1]),len(s4[1])),s4[1],'.',alpha = .4)
# plt.plot(np.linspace(1,len(s4[1]),len(s4[1])),signal.savgol_filter(s4[1],53,3),'.',color = "black",label = "Savitzk-Golay Filter")
# plt.title("Path of Fire with Initial Z = 1")
# plt.xlabel("Time t")
# plt.ylabel("Z")
# plt.legend(loc = "lower right")
# plt.figure(figsize = (15,10))
# plt.plot(np.linspace(1,len(s4[4]),len(s4[4])),s4[4],'.')
# plt.plot(np.linspace(1,len(s4[4]),len(s4[4])),signal.savgol_filter(s4[4],53,3),'.',color = "black",label = "Savitzk-Golay Filter")
# plt.title("Path of Fire with Initial Z = 1")
# plt.xlabel("Time t")
# plt.ylabel("Z")
# plt.legend(loc = "lower right")
# rate_step = 10
# indexs = np.linspace(1,800,80)
# diffs = []
# for i in indexs:
#     try:
#         diff = s8[3][i+1]-s8[3][i]
#         diffs.append(diff)
#     except:
#         IndexError
# # Want to find average del z from each i8[nitial z to show
# # that the initial z determines how the fire will
# plt.plot(np.linspace(1,len(fireTotal[40]),len(fireTotal[40])),fireTotal[40])
# for f in fired:
#     zs.append(landscape[f].z)


# masterStateMaps = []
# for u in statemaps:
#     for maps in u:
#         masterStateMaps.append(maps)
# from matplotlib.animation import FuncAnimation
# mastersub = np.array(mastersub)
# np.save("bowl_mats",mastersub)
# mastersub = np.load("bowl_mats.npy")
# fig, ax = plt.subplots(figsize=(15, 10));
# cmap = ListedColormap(['w', 'r', 'green'])
# cax = ax.matshow(masterStateMaps[500],cmap=cmap)
# plt.contour(zVals, colors = "b")
# plt.show()
# mastersub = masterStateMaps[1:2000]
# for i,frame in enumerate(mastersub):
#     fig, ax = plt.subplots(figsize=(15, 10))
#     cmap = ListedColormap(['w', 'r', 'green'])
#     cax = ax.matshow(frame,cmap=cmap)
#     plt.contour(zVals, colors = "b")
#     figname = "{}.png".format(i)
#     plt.savefig(figname)
#     plt.close(fig)
