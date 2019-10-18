# -*- coding: utf-8 -*-
"""
@author: Jakub Blawat
"""

import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe

beta=0.988   
theta=.679
delta=.013
kappa=5.24
nu=2
fine = 300

qe.tic()

# STEP 1: DISCRETIZE THE STATE SPACE. 
k_ss=(1/(1-theta)*((1/beta)+delta-1))**(-1/theta)
k=np.linspace(0.01,1.5*k_ss,fine-1)#evenly spaced grid. 

#STEP 2: Initial guess
V = np.empty(shape=[fine-1, fine-1])
V[:,0]=np.zeros((fine-1))


#STEP 3: Feasible return matrix (M).
Z1,Z2=np.meshgrid(k,k)

# Feasible combinations
def feasibility(z1,z2):
    return z1**(1-theta) + (1-delta)*z1 - z2

#Evaluate the feasibility of different combinations of k_t and k_t+1.
N = feasibility(Z1,Z2)

# Feasible utility function
def utility(z1,z2):
    for i in range(fine-1):
        for j in range (fine-1):
                if N[i,j]>=0:
                    return np.log10(z1**(1-theta) + (1-delta)*z1 - z2) - (kappa/(1+(1/nu)))
            
#Define the feasible return matrix
M = utility(Z1,Z2)
M[np.isnan(M)] = -1000

#STEP 4: VALUE FUNCTION MATRIX ITERATION
X = np.empty(shape=[fine-1, fine-1])
G= np.empty(shape=[fine-1, fine-1])
count=0
for s in range(0,fine-2):
    epsilon=0.01
    for i in range(fine-1):
        for j in range(fine-1):
                X[i,j]=M[i,j]+(beta*V[:,s][j])        
    for i in range(0,fine-1):
        V[:,s+1][i]= np.amax(X[:,i]) #Iteration
        G[:,s][i]=np.argmax(X[:,i]) # Policy function at each iteration. 
        for i in range(0,fine-1):
            if abs(V[:,s+1][i]-V[:,s][i])> epsilon:
                continue
            else:
                count +=1
                break

qe.toc()

print('Number of iterations:', count)

# PLOT 
k=np.linspace(0.01,1000,fine-1)
plt.plot(k,V[:,fine-2])
plt.title('Brute Force Value Function')
plt.xlabel('Capital')
plt.ylabel('Utility')
