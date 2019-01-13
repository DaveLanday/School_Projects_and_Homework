#!/usr/bin/env python
# -*-coding: utf-8 -*-


#epidemic_spreading.py
#David W. Landay
#LAST UPDATED: 11-19-2018

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#write some functions:
def gen_degree_dist(p, k_max):
    """
        GEN_DEGREE_DIST: generates the degree distribution for our hypothetical
                         network, along with the initial conditions.

                         ARGS:
                              p: the probability to generate the degree
                                 geometric degree distribution.
                                 Type: Float

                              k_max: the largest allowable degree in the
                                     network. Type: Int
    """
    #generate the degree distribution:
    p_k  = [p*(1-p)**(k) for k in range(0, k_max)]
    p_k  = np.array(p_k)

    #create the distributions:
    p_nk = p_k*0.6
    p_vk = p_k - p_nk

    #get the proportion of I_vk and I_nk (assume that most infected people will not be vaccinated):
    I_nk = p_nk*0.01
    I_vk = p_vk*0.01

    return p_nk,p_vk,I_nk,I_vk,p_k

def gen_smart_degree_dist(p, k_max, p_c):
    """
        GEN_SMART_DEGREE_DIST: generates the degree distribution when the vax.
                               campaign convinces the upper 1 - p_c percent of
                               the nodes with highest degree to get vaccinated.
                               They are smart nodes...

                               ARGS:
                                    p: the probability to generate the degree
                                       geometric degree distribution.
                                       Type: Float

                                    k_max: the largest allowable degree in the
                                           network. Type: Int

                                    p_c: the threshold proportion, above which
                                         individuals adopt the vaccine.
                                         Type: Float
    """
    #generate the degree distribution:
    p_k  = [p*(1-p)**(k) for k in range(0, k_max)]
    p_k  = np.array(p_k)

    #calculate the cdf:
    cdf_k = np.cumsum(p_k)

    #find the index that corresponds to k such that cdf_k =~ 0.6:
    closest_idx = min(range(len(cdf_k)), key=lambda i: abs(cdf_k[i] - p_c))

    #create the distributions of vaccinated and non vaccinated:
    p_nk = p_k*0.6
    p_vk = p_k - p_nk

    #add p_nk upper 1 - p_c% to p_vk:
    p_vk[closest_idx:] += p_nk[closest_idx:]

    #remove the upper 1-p_c percent from p_nk:
    p_nk[closest_idx:] = 0

    #define the initial proportions:
    I_nk = p_nk*0.01
    I_vk = p_vk*0.01

    #return initial values:
    return p_nk,p_vk,I_nk,I_vk,p_k

def plot_degree_dist(k, p_k, display=True):
    """
        PLOT_DEGREE_DIST: plots the geometric degree distribution and the cdf of
                          the geometric degree distribution

                          ARGS:
                               p_k: the geometric degree distribution.
                                    Type: nparray

                               k: an array containing the allowable degrees from
                                  1 to k_max. Type: nparray

                               display: if true, will display the plot in
                                        in addition to saving it as a .png.
                                        default is True. Type: Boolean

                         RETURNS:
                                None. Displays a plot if display set to True.
    """

    #plot the degree distribution:
    plt.figure(figsize=(10,8))
    plt.title(r'Geometric Degree Distribution for $k_{} = ${}'.format('{max}',k[-1]))
    plt.xlabel(r'degree: $k$')
    plt.ylabel(r'proportion of nodes of degree $k$: $p_k$')
    plt.plot(k, p_k, color='blue')
    plt.savefig('geometric_degree_distribution_k_'+str(k[-1])+'.png')
    if display==True:
        plt.show()

    #calculate the cdf:
    cdf_k = np.cumsum(p_k)

    #find the index that corresponds to k such that cdf_k =~ 0.6:
    closest_idx = min(range(len(cdf_k)), key=lambda i: abs(cdf_k[i]-0.6))

    #plot it:
    plt.figure(figsize=(10,8))
    plt.title(r'CDF of degree for $k_{} = ${}'.format('{max}',k[-1]))
    plt.xlabel(r'degree: $k$')
    plt.ylabel(r'proportion of nodes $\leq$ degree $k$: $p_{k_{\leq}}$')
    plt.plot(k, cdf_k, color='orange')
    plt.axvline(k[closest_idx], color='red', linestyle='--')
    plt.savefig('CDF_Geometric_degree_distribution_k_'+str(k[-1])+'.png')
    if display==True:
        plt.show()

def calc_moment_closure(k, I_nk, p_nk, I_vk, p_vk, f):
    """
        CALC_MOMENT_CLOSURE: calculates the moment closure at time t

                            ARGS:
                                 k: an array containing the allowable degrees from
                                    1 to k_max. Type: nparray

                                 I_nk: an array containing proportions of
                                       infected and non-vaccinated nodes for
                                       each degree-k. Type: nparray

                                 p_nk: an array containing the proportions of
                                       non-vaccinated nodes for each degree-k.
                                       Type: nparray

                                 I_vk: an array containing proportion of
                                       infected and vaccinated nodes for
                                       each degree-k. Analagous to a failed
                                       influenza vaccine. Type: nparray

                                 p_vk: an array containing the proportions of
                                       vaccinated nodes for each degree-k.
                                       Type: nparray

                                 f: the factor by which infection spreads to and
                                    from vaccinated nodes. Type: Float

                            RETURNS:
                                    Float representing the moment closure at
                                    time t
    """

    #calculate theta_N
    theta_N = sum(k*I_nk)/sum(k*p_nk)

    #calculate theta_F
    theta_V = sum(k*I_vk)/sum(k*p_vk)

    #return the moment closure:
    return theta_N + f*theta_V

def dIk_dt(I_k, t, k, p_vk, p_nk, lamb, f):
    """
        DIK_DT: calculates the rate of change of the fracion of infected edges

                ARGS:
                     I_k: The set of all I_nk and Ivk. The set will be of length
                          2*k and the first k elements represent the I_nk. The
                          proceeding k elements represents I_vk. Type: nparray

                     t: an array of of consecutive integer time steps.
                        Type: nparray

                     k: an array containing the allowable degrees from
                        1 to k_max. Type: nparray

                     p_vk: an array containing the proportions of
                           vaccinated nodes for each degree-k.
                           Type: nparray

                     p_nk: an array containing the proportions of
                           non-vaccinated nodes for each degree-k.
                           Type: nparray

                     lamb: the lambda value in the heterogeneous mean-field eqn.
                           that represents the rate at which infection spreads.
                           Type: Float

                     f: the factor by which infection spreads to and
                        from vaccinated nodes. Type: Float

                RETURNS: a 1-d nparray of length 2*k representing the change in
                         proportino for I_k, the number of infected nodes for
                         each degree-k
    """
    #index to split (keep in mind odd number of degree k):
    if len(I_k)%2 == 0:
        split = int(len(I_k)/2)
    else:
        split = int((len(I_k) + 1)/2)

    #calculate moment closure:
    moment_closure = calc_moment_closure(k, I_k[:split], p_nk, I_k[split:], p_vk, f)

    #calculate the derivatives:
    ink = lamb*k*(p_nk - I_k[:split])*moment_closure - I_k[:split]
    ivk = f*lamb*k*(p_vk - I_k[split:])*moment_closure - I_k[split:]

    #calculate:
    dIk_dt = np.concatenate((ink,ivk), axis=None)
    return dIk_dt


def siv(lamb, f, p_nk, p_vk, I_nk, I_vk, k, t):
    """
        SIV: runs the susceptible-infected-vaccinated model via heterogeneous
             mean-field on a network

             ARGS:
                     lamb: the lambda value in the heterogeneous mean-field eqn.
                           that represents the rate at which infection spreads.
                           Type: Float

                     f: the factor by which infection spreads to and
                        from vaccinated nodes. Type: Float

                     p_vk: an array containing the proportions of
                           vaccinated nodes for each degree-k.
                           Type: nparray

                     p_nk: an array containing the proportions of
                           non-vaccinated nodes for each degree-k.
                           Type: nparray

                     p_vk: an array containing the proportions of
                           vaccinated nodes for each degree-k.
                           Type: nparray

                     I_nk: an array containing proportions of
                           infected and non-vaccinated nodes for
                           each degree-k. Type: nparray

                     I_vk: an array containing proportion of
                           infected and vaccinated nodes for
                           each degree-k. Analagous to a failed
                           influenza vaccine. Type: nparray

                     k: an array containing the allowable degrees from
                        1 to k_max. Type: nparray

                     t: an array of of consecutive integer time steps.
                        Type: nparray

             RETURNS: a 1-d nparray of length 2*k representing the I_k values at
                      the t+1_th time step.

    """

    #init two arrays to store t rows of k definite integral calculations:
    Ik = np.zeros((len(t), 2*len(k)))

    #populate with the initial conditions:
    Ik[0,] = np.concatenate((I_nk,I_vk), axis=None)

    #define what happens at each timestep:
    for i in range(len(t)-1):

        #get the timestep to integrate over:
        ts = [t[i], t[i+1]]
        Ik[i+1,] = odeint(dIk_dt,Ik[i,], ts, args=(k, p_vk, p_nk, lamb, f))[1]

    #return
    return Ik

def plot_scenario(f, lamb, p_nk, p_vk, I_nk, I_vk, k, t, count, plot=False):
    """
        PLOT_REGIME: plots an individual scenario for given f and lambda:

                ARGS:
                    f: the factor by which infection spreads to and
                       from vaccinated nodes. Type: Float

                    lamb: the lambda value in the heterogeneous mean-field eqn.
                          that represents the rate at which infection spreads.
                          Type: Float

                    p_nk: an array containing the proportions of
                          non-vaccinated nodes for each degree-k.
                          Type: nparray

                    p_vk: an array containing the proportions of
                          vaccinated nodes for each degree-k.
                          Type: nparray

                    I_nk: an array containing proportions of
                          infected and non-vaccinated nodes for
                          each degree-k. Type: nparray

                    I_vk: an array containing proportion of
                          infected and vaccinated nodes for
                          each degree-k. Analagous to a failed
                          influenza vaccine. Type: nparray

                    k: an array containing the allowable degrees from
                       1 to k_max. Type: nparray

                    t: an array of of consecutive integer time steps.
                       Type: nparray

                    count: file naming convention for plot output; gets padded
                           by three 0's in the function. Type: Int

                    plot: indicates whether or not to plot any outputself.
                          Type: Boolean

                RETURNS: Two integer values representing the proportion of
                         infected and susceptible individuals, respectively,
                         at the last time step.
    """

    #calculate proportion of infected
    I = np.zeros(len(t))
    Ik = siv(lamb, f, p_nk, p_vk, I_nk, I_vk, k, t)
    for i in range(len(I)):
        I[i] = sum(Ik[i])

    #calculate the susceptible population:
    S = 1-I

    if plot==True:
        plt.figure(figsize = (8,6))
        plt.plot(t,I, 'o', color='red', label='Number Infected')
        plt.plot(t,S, 'o', color='blue', label='Number Susceptible')
        plt.title(r'Epidemic Dynamics on Randomly Initialized Network, $f = ${}'.format(round(f,3)))
        plt.xlabel(r'time step: $t$')
        plt.ylabel(r'Population')
        plt.legend(loc='upper right')

        plt.savefig('000'+str(count)+'.png')
    return I[-1,], S[-1,]

if __name__ == '__main__':

#############################_INITIAL_CONDITIONS_###############################

    #how much time should we give the system?
    t = np.linspace(0, 100, 101)

    #number of infected nodes with degree 1:
    max_k = 20

    #choose a infection rate lambda:
    lamb = 0.3

    #vaccine effect f (slows the rate of vaccinated individuals changing states)
    f = np.linspace(0,1,21) #steps of 0.05

#######################_GENERATE_INITIAL_POPULATION_############################
    p_nk,p_vk,I_nk,I_vk,p_k = gen_degree_dist(0.25, max_k)

    #generate the values of k:
    k = 1 + np.indices(I_nk.shape).reshape(I_nk.shape,)
    split = int(len(k)/2)

#############################_RUN_THE_MODEL_####################################
    #calculate the infected population:
    fig,ax = fig, ax = plt.subplots(figsize = (8,6))
    plt.title(r'Epidemic Spreading Due to $f$, Rate of Transmission Between Vaccinated Individuals')
    plt.xlabel(r'spread between vaccinated factor: $f$')
    plt.ylabel(r'Population')
    plt.legend(loc='upper right')
    count = 0
    for val in f:
        count+=1
        I, S = plot_scenario(val, lamb, p_nk, p_vk, I_nk, I_vk, k, t, count, plot=False)
        plt.plot(val,I,'.', color='red', label='Infected')
        plt.plot(val,S,'.', color='blue', label='Susceptible')
    plt.show()

###############################_PART_D_#########################################
    #plot_degree_dist(k, p_k, display=True)

    #generate with addoption campaign:
    p_nk,p_vk,I_nk,I_vk,p_k = gen_smart_degree_dist(0.25, max_k, 0.6)

    fig,ax = fig, ax = plt.subplots(figsize = (8,6))
    plt.title(r'Epidemic Spreading Due to $f$, With Addoption Campaign')
    plt.xlabel(r'spread between vaccinated factor: $f$')
    plt.ylabel(r'Population')
    plt.legend(loc='upper right')

    for val in f:
        count+=1
        I, S = plot_scenario(val, lamb, p_nk, p_vk, I_nk, I_vk, k, t, count, plot=False)
        plt.plot(val,I,'.', color='red', label='Infected')
        plt.plot(val,S,'.', color='blue', label='Susceptible')
    plt.show()
