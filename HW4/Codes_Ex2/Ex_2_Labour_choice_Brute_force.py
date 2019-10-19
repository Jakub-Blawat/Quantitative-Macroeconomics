# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 00:35:51 2019

@author: kuba
"""
import sympy as sp
import numpy as np
import matplotlib.pyplot as mpl
import scipy as scp
import quantecon as qe


#Set parameters:
beta=0.988 #standard US rate of impatience
theta=0.679 #labour share
delta=0.013 #standard US depreciation rate
kappa=5.24
nu=2

qe.tic()

# Set number of grid points.
n_k = 100
n_h = 100

# Set lower and upper bound of the state space.
k_low = 0.01
k_up  = 80
h_low = 0.01
h_up  = 80

# Set tolerance of error.
epsilon = 0.01

# Set grid points.
k_grid=np.linspace(k_low,k_up,n_k)
h_grid=np.linspace(h_low,h_up,n_h)

# Set an initial value function.
V_0 = np.zeros((n_k))

# Set return matrix
m = np.zeros((n_k,n_k,n_h))    

# Set empty policy functions:
k_policy = np.zeros((n_k))
h_policy = np.zeros((n_k))
c_policy = np.zeros((n_k))

# Define functions:
def y_func(k, h):
    return k**(1-theta)*h**theta

def u(c, h):
    return  np.log(c)-kappa*((h**(1+(1/nu)))/(1+(1/nu)))


#Bellman equation
def bellman_operator(V,return_policies=False):
    V_next = np.zeros((n_k))
    m = np.zeros((n_k,n_k,n_h)) 
    
    for ik, k in enumerate(k_grid):
        for igh, gh in enumerate(h_grid):
            ### Fill up return matrix: All possible choices of k' (k_grid) and labor (gh) per each state (k)
            m[ik,:,igh] =  u((y_func(k,igh) +(1-delta)*k - k_grid),gh) +beta*V
            
        V_next[ik] = np.nanmax(m[ik,:,:])        
        k_policy[ik] = k_grid[np.unravel_index(np.argmax(m[ik,:,:], axis=None), m[ik,:,:].shape)[0]]
        h_policy[ik] = h_grid[np.unravel_index(np.nanargmax(m[ik,:,:], axis=None), m[ik,:,:].shape)[1]]
        c_policy[ik] =  y_func(k,h_policy[ik]) +(1-delta)*k - k_policy[ik]          
    if return_policies==True:
        return V_next, k_policy, h_policy, c_policy
    else:
        return V_next

## Compute fixed point in the bellman equation
qe.tic()
V = qe.compute_fixed_point(bellman_operator, V_0, max_iter=1000, error_tol=epsilon, print_skip=20)
V, g_k, g_h, g_c = bellman_operator(V, return_policies=True)

qe.toc()

# Plot
fig, ax = mpl.subplots()
ax.plot(k_grid,V, label='V(k)')
mpl.show()

fig, ax = mpl.subplots()
ax.plot(k_grid,g_h, label='h(k)')
ax.plot(k_grid,g_c, label='c(k)')
ax.legend()
mpl.show()

fig, ax = mpl.subplots()
ax.plot(k_grid,g_k, label='k1(k)')
ax.legend()
mpl.show()