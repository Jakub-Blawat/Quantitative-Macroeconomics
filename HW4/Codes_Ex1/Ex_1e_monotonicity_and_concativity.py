# -*- coding: utf-8 -*-
"""
@author: Jakub Bławat
"""

import numpy as np
import matplotlib as mpl
import time

beta=0.988 #standard US rate of impatience
theta=0.679 #labour share
delta=0.013 #standard US depreciation rate
h=1 #fixed labour supply
kappa=0
#kappa=5.24
nu=2

k_low=5 #set lower bound for the capital grid
k_high=55 #set upper bound for the capital grid
fine=101 #set density of the grid

k_grid=np.linspace(k_low, k_high, fine)

vfpx=np.zeros((1,fine-1)) #initialising x-grid for plotting valfun
for i in range(0, fine-1):
    vfpx[0,i]=k_grid[i]

tic=time.time()

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log((k_grid[i]**(1-theta))*((h)**theta)-k_grid[j]+(1-delta)*k_grid[i])-kappa*(h**(1+1/nu))/(1+1/nu)
        else: M[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=1000 #maximum number of iterations
conv=0.001 #convergence level

chi=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

#first iteration goes as it used to go, because we need initial g
valfun_old=np.copy(valfun) #store old valfun
for i in range(0, fine-1):
    chi[i,]=M[i,]+beta*valfun #update chi
for i in range(0, fine-1):
    valfun[0,i]=np.amax(chi[i,]) #update valfun
    g[0,i]=np.argmax(chi[i,])
progress=abs(np.amax(valfun-valfun_old)) #check progress
iter=iter+1

while iter<maxiter and progress>conv:
#for j in range(0, maxiter):
    valfun_old=np.copy(valfun)
    for i in range(0, fine-1):
        for j in range(int(g[0,i]), fine-1):
            chi[i,j]=M[i,j]+beta*valfun[0,j]
            if j>0 and chi[i,j]<chi[i,j-1]:
                #g[0,i]=j
                break
            valfun[0,i]=np.amax(chi[i,]) #update valfun
            g[0,i]=np.argmax(chi[i,])
    progress=abs(np.amax(valfun-valfun_old))
    iter=iter+1

toc=time.time()

mpl.pyplot.scatter(vfpx, valfun, s=1, color='blue')
mpl.pyplot.xlabel('capital')
mpl.pyplot.ylabel('value function')
mpl.pyplot.legend()
#mpl.pyplot.xlim([0, 200])
#mpl.pyplot.ylim([0, -500])
mpl.pyplot.title(r"Value function under g-monotonicity and valfun-concavity")
mpl.pyplot.show()

print("Value function iteration required ", iter, " iterations")
print("Value function iterations took ", toc-tic, " seconds")