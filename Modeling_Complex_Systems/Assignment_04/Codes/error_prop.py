# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:03:25 2018

@author: mgreen13
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.stats as stats


# DEFINE DIFFERENTIAL EQUATIONS FOR 3 SPECIES MODEL

def dP_dt75(P,t):
    A = np.array([[0.5,0.5,.1],[-.5,-.1,.1],[.75,.1,.1]],dtype=np.float64)
    dx1 = P[0]*(A[0,0]*(1-P[0])+A[0,1]*(1-P[1])+A[0,2]*(1-P[2]))
    dx2 = P[1]*(A[1,0]*(1-P[0])+A[1,1]*(1-P[1])+A[1,2]*(1-P[2]))
    dx3 = P[2]*(A[2,0]*(1-P[0])+A[2,1]*(1-P[1])+A[2,2]*(1-P[2]))
    return(np.array([dx1,dx2,dx3]))
   
def dP_dt12(P,t):
    A = np.array([[0.5,0.5,.1],[-.5,-.1,.1],[1.2,.1,.1]],dtype=np.float64)
    dx1 = P[0]*(A[0,0]*(1-P[0])+A[0,1]*(1-P[1])+A[0,2]*(1-P[2]))
    dx2 = P[1]*(A[1,0]*(1-P[0])+A[1,1]*(1-P[1])+A[1,2]*(1-P[2]))
    dx3 = P[2]*(A[2,0]*(1-P[0])+A[2,1]*(1-P[1])+A[2,2]*(1-P[2]))
    return(np.array([dx1,dx2,dx3]))


def dP_dt15(P,t):
    A = np.array([[0.5,0.5,.1],[-.5,-.1,.1],[1.5,.1,.1]],dtype=np.float64)
    dx1 = P[0]*(A[0,0]*(1-P[0])+A[0,1]*(1-P[1])+A[0,2]*(1-P[2]))
    dx2 = P[1]*(A[1,0]*(1-P[0])+A[1,1]*(1-P[1])+A[1,2]*(1-P[2]))
    dx3 = P[2]*(A[2,0]*(1-P[0])+A[2,1]*(1-P[1])+A[2,2]*(1-P[2]))
    return(np.array([dx1,dx2,dx3]))
    
def pyInt(func,h,n):
    ts = np.linspace(0,n*h,n)
    y0 = [.1,.2,.3]
    
    y = odeint(func,y0,ts)
    return(ts,y)

# EULERS METHOD FOR NUMERICAL INTEGRATION
def euler(f,x0,h,n):
    xn = n*h
    # Time steps
    ts = np.linspace(x0,xn,n)
    # create array to store approximated y-values
    y = np.zeros([n,3],dtype=np.float64)
    #store intial values
    y[0,:] = [.1,.2,.3]
    # MARCH FORWARD! 
    for ind in list(range(n-1)):
        # Iterate over function evaluations for x1,x2,x3
        y[ind+1,:] = h*(f(y[ind],ts))+y[ind]  
    return(ts,y)
    
def rk2(f,x0,h,n):
    xn = n*h
     # Time steps
    ts = np.linspace(x0,xn,n)
    # create array to store approximated y-value
    y = np.zeros([n,3],dtype=np.float64)
    #store intial values
    y[0,:] = [.1,.2,.3]
    
    for ind,i in enumerate(ts):
        try:
            y[ind+1,:] = y[ind] + h*((f(y[ind],i))+np.array(f(y[ind]+h*f(y[ind],i),ts[ind+1])))/2
        except:
            IndexError
    return(ts,y)
 
    
y0 = [.1,.2,.3]
n =200
dP_dt = [dP_dt75,dP_dt12,dP_dt15]

# TODO: CALCULATE ERROR 

#log spaced numbers
hTypes = np.linspace(0.01,1,100)


error_RK = []
error_E = []
# VARYING ALPHA
for a,func in enumerate(dP_dt):
    max_error_RK = []
    max_error_e = []
    # VARYING STEP SIZE
    for h in hTypes:
        [tRk,yRk] = rk2(dP_dt[a],0,h,50)
        [tE,yE] = euler(dP_dt[a],0,h,50)
        [tDef, yDef] = pyInt(dP_dt[a],h,50)
        
        # PULL FROM X1
        yrk2 = yRk[:,0]
        ye = yE[:,0]
        yb = yDef[:,0]
        
        # CALCULATE MAXIMUM ABSOLUTE ERROR
        diff1 = ye-yb
        maxdiff1= max(diff1,key = abs)
        maxdiff1 = abs(maxdiff1)
        
        diff2 = yrk2-yb
        maxdiff2= max(diff2,key = abs)
        maxdiff2 = abs(maxdiff2)
        # Append max abs error for each value of h
        max_error_e.append(maxdiff1)
        max_error_RK.append(maxdiff2)
    # append list of max abs error for each a across h values. 
    error_RK.append(max_error_RK)
    error_E.append(max_error_e)


for ind,hval in enumerate(dP_dt):
    x = np.log10(hTypes)
    y1 = np.log10(error_RK[ind])
    y2 = np.log10(error_E[ind])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y1)
    slope2, intercept2, r_value, p_value, std_err = stats.linregress(x,y2)
    line = slope*x+intercept
    line2 = slope2*x+intercept2
    
    
    a = [.75,1.2,1.5]
    plt.figure(figsize = (15,9))
    plt.plot(x,y1,'x',color = 'blue',label = "RK2 Error")
    plt.plot(x,line,label = "y = {}x + {}".format(round(slope,3),round(intercept,3)),linewidth = 3)

    plt.plot(x,y2,'o',color = 'orange',label = "Euler's Error")
    plt.plot(x,line2,label = "y = {}x + {}".format(round(slope2,3),round(intercept2,3)),linewidth = 3)

    plt.legend(loc = "upper left")
    plt.xlabel("timestep (h)",fontsize = 14)
    plt.ylabel("Error in Approximation",fontsize = 14)
    plt.title("Maximum Absolute Error for Varying Stepsize: "+ r'$\alpha$' +" = {}".format(a[ind]),fontsize = 14)

    




