# -*- coding: utf-8 -*-
"""
@author: Jakub Bławat
"""

import sympy as sp
import numpy as np
import matplotlib as mpl
import scipy as scp

tauc=0.2 #let's introduce taxation rate on consumption
tauk=0 #same on capital

beta=0.98 #standard US rate of impatience
theta=0.67 #labour share
delta=0.0625 #standard US depreciation rate
h=0.31 #fixed labour supply
#z=2*1.63
z=1.63 #selected technology level that gives capout=4

k_low=0 #set lower bound for the capital grid
k_high=20 #set upper bound for the capital grid
fine=2001 #set density of the grid

k_grid=np.linspace(k_low, k_high, fine)

i=1
for i in range(0, fine-1):
    k_grid[i]=round(k_grid[i],2)

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if ((1-tauk)*k_grid[i]**(1-theta))*((z*h)**theta)-(1-tauk)*k_grid[j]+(1-delta)*(1-tauk)*k_grid[i]>=0:
            M[i,j]=np.log((1-tauc)*((1-tauk)*k_grid[i]**(1-theta)*(z*h)**theta-(1-tauk)*k_grid[j]+(1-delta)*(1-tauk)*k_grid[i]))
        else: M[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=1000 #maximum number of iterations
conv=0.001 #convergence level

chi=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

while iter<maxiter and progress>conv:
#for j in range(0, maxiter):
    valfun_old=np.copy(valfun) #store old valfun
    for i in range(0, fine-1):
        chi[i,]=M[i,]+beta*valfun #update chi
    for i in range(0, fine-1):
        valfun[0,i]=np.amax(chi[i,]) #update valfun
        g[0,i]=np.argmax(chi[i,])
    progress=abs(np.amax(valfun-valfun_old)) #check progress
    iter=iter+1

#Here value function iteration ends

#Now we compute steady state level of capital

maxiter=1000
conv=0.001
progress=1
k_init=10 #set the initial value of capital for the convergence path computation
k_path=np.zeros((1, maxiter)) #initialize array that stores capital path
k_path[0, 0]=(1-tauk)*k_init
iter=1 #number of started iterations
while iter<maxiter:# and progress>conv: #this is to FIND steady state, for simulating capital path delete second condition!
    i=np.where(k_grid==round((1-tauk)*k_init, 2)) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    k_path[0,iter]=k_grid[l] #next element in k_path is element of k_grid whose index is stored in g[j]
    progress=abs(k_path[0,iter]-k_path[0,iter-1])
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1
    
k_ss=k_path[0, iter-1]
y_ss=k_ss**(1-theta)*(z*h)**theta
i_ss=delta*k_ss    
c_ss=y_ss-i_ss
    
c_path=np.zeros((1, maxiter))
i_path=np.zeros((1, maxiter))
y_path=np.zeros((1, maxiter))

#compute consumption path
iter=1
k_init=4
while iter<maxiter-1: 
    i=np.where(k_grid==round((1-tauk)*k_init, 2)) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    c_path[0,iter]=k_path[0,iter]**(1-theta)*((z*h)**theta)-k_path[0,iter+1]+(1-delta)*k_path[0,iter]
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1
    
#compute output path
iter=1
k_init=10
while iter<maxiter-1: 
    i=np.where(k_grid==round((1-tauk)*k_init, 2)) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    y_path[0,iter]=k_path[0,iter]**(1-theta)*((z*h)**theta)
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

#compute savings path    
iter=1
k_init=10
while iter<maxiter-1: 
    i=np.where(k_grid==round((1-tauk)*k_init, 2)) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    i_path[0,iter]=y_path[0,iter]-c_path[0,iter]
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1
    
#define grid for plotting
plot_grid=np.zeros((1, maxiter))
for i in range(0, maxiter):
    plot_grid[0,i]=i+1
    
mpl.pyplot.scatter(plot_grid, c_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('consumption')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 15])
mpl.pyplot.title(r"Consumption path in economy with consumption tax")
mpl.pyplot.show()
