# -*- coding: utf-8 -*-
"""
@author: Jakub Bławat
"""

##############################################################################
#Q1 a), b)
##############################################################################

import sympy as sp
import numpy as np
import matplotlib as mpl

beta=0.98 #standard US rate of impatience
theta=0.67 #labour share
delta=0.0625 #standard US depreciation rate
h=0.31 #fixed labour supply
z=2*1.63
#z=1.63 #selected technology level that gives capout=4

k_low=0 #set lower bound for the capital grid
k_high=50 #set upper bound for the capital grid
fine=1001 #set density of the grid

k_grid=np.linspace(k_low, k_high, fine)

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*((z*h)**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M[i,j]=np.log(k_grid[i]**(1-theta)*(z*h)**theta-k_grid[j]+(1-delta)*k_grid[i])
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
k_init=25 #set the initial value of capital for the convergence path computation
k_path=np.zeros((1, maxiter)) #initialize array that stores capital path
k_path[0, 0]=k_init
iter=1 #number of started iterations
while iter<maxiter:# and progress>conv: #this is to FIND steady state, for simulating capital path delete second condition!
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
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
capout=k_ss/y_ss
invout=i_ss/y_ss

#print("capout with z=", z, "equals ", capout)
print("invout with z=", z, "equals ", invout)

#define grid for plotting
for i in range(0, maxiter):
    plot_grid[0,i]=i+1

#plot actually
mpl.pyplot.scatter(plot_grid, k_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('capital')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 25])
mpl.pyplot.title(r"Capital path for $z=1.63$")
mpl.pyplot.show()

##############################################################################
#Q1 c)
##############################################################################

#remember to compute part c) having completed all calculations above under z=2*1.63, but starting from k_ss=4 (i.e., from k_ss computed for z=1.63) 

c_path=np.zeros((1, maxiter))
i_path=np.zeros((1, maxiter))
y_path=np.zeros((1, maxiter))
iter=1
k_init=4

#compute capital path
while iter<maxiter: 
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    k_path[0,iter]=k_grid[l] #next element in k_path is element of k_grid whose index is stored in g[j]
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

mpl.pyplot.scatter(plot_grid, k_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('capital')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 25])
mpl.pyplot.title(r"Transition path of capital from $z=1.63$ to $z=3.26$")
mpl.pyplot.show()

#compute consumption path
iter=1
k_init=4
while iter<maxiter-1: 
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    c_path[0,iter]=k_path[0,iter]**(1-theta)*((z*h)**theta)-k_path[0,iter+1]+(1-delta)*k_path[0,iter]
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

mpl.pyplot.scatter(plot_grid, c_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('consumption')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 5])
mpl.pyplot.title(r"Transition path of consumption from $z=1.63$ to $z=3.26$")
mpl.pyplot.show()

#compute output path
iter=1
k_init=4
while iter<maxiter-1: 
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    y_path[0,iter]=k_path[0,iter]**(1-theta)*((z*h)**theta)
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

mpl.pyplot.scatter(plot_grid, y_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('output')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 5])
mpl.pyplot.title(r"Transition path of consumption from $z=1.63$ to $z=3.26$")
mpl.pyplot.show()

#compute savings=investments path
iter=1
k_init=4
while iter<maxiter-1: 
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    i_path[0,iter]=y_path[0,iter]-c_path[0,iter]
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

mpl.pyplot.scatter(plot_grid, i_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('savings')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 5])
mpl.pyplot.title(r"Transition path of savings from $z=1.63$ to $z=3.26$")
mpl.pyplot.show()

##############################################################################
#Q1 d)
##############################################################################

#We need to obtain M1, chi1 and g1 for low z

beta=0.98 #standard US rate of impatience
theta=0.67 #labour share
delta=0.0625 #standard US depreciation rate
h=0.31 #fixed labour supply
#z=2*1.63
z=1.63 #selected technology level that gives capout=4

k_low=0 #set lower bound for the capital grid
k_high=50 #set upper bound for the capital grid
fine=1001 #set density of the grid

k_grid=np.linspace(k_low, k_high, fine)

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g1=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M1=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*((z*h)**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M1[i,j]=np.log(k_grid[i]**(1-theta)*(z*h)**theta-k_grid[j]+(1-delta)*k_grid[i])
        else: M1[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=1000 #maximum number of iterations
conv=0.001 #convergence level

chi1=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

while iter<maxiter and progress>conv:
#for j in range(0, maxiter):
    valfun_old=np.copy(valfun) #store old valfun
    for i in range(0, fine-1):
        chi1[i,]=M1[i,]+beta*valfun #update chi
    for i in range(0, fine-1):
        valfun[0,i]=np.amax(chi1[i,]) #update valfun
        g1[0,i]=np.argmax(chi1[i,])
    progress=abs(np.amax(valfun-valfun_old)) #check progress
    iter=iter+1

#Here value function iteration ends

#Now we compute steady state level of capital

maxiter=1000
conv=0.001
progress=1
k_init=25 #set the initial value of capital for the convergence path computation
k_path1=np.zeros((1, maxiter)) #initialize array that stores capital path
k_path1[0, 0]=k_init
iter=1 #number of started iterations
while iter<maxiter:# and progress>conv: #this is to FIND steady state, for simulating capital path delete second condition!
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    k_path1[0,iter]=k_grid[l] #next element in k_path is element of k_grid whose index is stored in g[j]
    progress=abs(k_path1[0,iter]-k_path1[0,iter-1])
    k_init=np.copy(k_path1[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1
    
k_ss1=k_path1[0, iter-1]
y_ss1=k_ss1**(1-theta)*(z*h)**theta
i_ss1=delta*k_ss1    
c_ss1=y_ss1-i_ss1



#We need to obtain M2, chi2 and g2 for high z



beta=0.98 #standard US rate of impatience
theta=0.67 #labour share
delta=0.0625 #standard US depreciation rate
h=0.31 #fixed labour supply
z=2*1.63
#z=1.63 #selected technology level that gives capout=4

k_low=0 #set lower bound for the capital grid
k_high=50 #set upper bound for the capital grid
fine=1001 #set density of the grid

k_grid=np.linspace(k_low, k_high, fine)

valfun=np.zeros((1,fine-1)) #create array of zeros to store value function
g2=np.zeros((1, fine-1)) #create array to store indices of optimal capital levels (useful for policy function)

#Now we'll build the return matrix:

M2=np.zeros((fine-1, fine-1)) #initialize the matrix
for i in range(0, fine-1):
    for j in range(0, fine-1):
        if (k_grid[i]**(1-theta))*((z*h)**theta)-k_grid[j]+(1-delta)*k_grid[i]>=0:
            M2[i,j]=np.log(k_grid[i]**(1-theta)*(z*h)**theta-k_grid[j]+(1-delta)*k_grid[i])
        else: M2[i,j]=-1000000
        
#Now true iteration starts
        
iter=0 #number of completed iterations
maxiter=1000 #maximum number of iterations
conv=0.001 #convergence level

chi2=np.zeros((fine-1, fine-1)) #initialize chi
progress=1

while iter<maxiter and progress>conv:
#for j in range(0, maxiter):
    valfun_old=np.copy(valfun) #store old valfun
    for i in range(0, fine-1):
        chi2[i,]=M2[i,]+beta*valfun #update chi
    for i in range(0, fine-1):
        valfun[0,i]=np.amax(chi2[i,]) #update valfun
        g2[0,i]=np.argmax(chi2[i,])
    progress=abs(np.amax(valfun-valfun_old)) #check progress
    iter=iter+1

#Here value function iteration ends

#Now we compute steady state level of capital

maxiter=1000
conv=0.001
progress=1
k_init=25 #set the initial value of capital for the convergence path computation
k_path2=np.zeros((1, maxiter)) #initialize array that stores capital path
k_path2[0, 0]=k_init
iter=1 #number of started iterations
while iter<maxiter:# and progress>conv: #this is to FIND steady state, for simulating capital path delete second condition!
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    k_path2[0,iter]=k_grid[l] #next element in k_path is element of k_grid whose index is stored in g[j]
    progress=abs(k_path2[0,iter]-k_path2[0,iter-1])
    k_init=np.copy(k_path2[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1
    
k_ss2=k_path2[0, iter-1]
y_ss2=k_ss2**(1-theta)*(z*h)**theta
i_ss2=delta*k_ss2    
c_ss2=y_ss2-i_ss2




#Now the true simulation starts

c_path=np.zeros((1, maxiter))
i_path=np.zeros((1, maxiter))
y_path=np.zeros((1, maxiter))
iter=1
k_init=25

#compute capital path
while iter<11: 
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g2[0,j])
    k_path[0,iter]=k_grid[l] #next element in k_path is element of k_grid whose index is stored in g[j]
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1
    
while iter>10 and iter<maxiter: 
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g1[0,j])
    k_path[0,iter]=k_grid[l] #next element in k_path is element of k_grid whose index is stored in g[j]
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

mpl.pyplot.scatter(plot_grid, k_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('capital')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 25])
mpl.pyplot.title(r"Capital path induced by a productivity shock at $t=10$")
mpl.pyplot.show()

#compute consumption path
iter=1
k_init=4
while iter<maxiter-1: 
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    c_path[0,iter]=k_path[0,iter]**(1-theta)*((z*h)**theta)-k_path[0,iter+1]+(1-delta)*k_path[0,iter]
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

mpl.pyplot.scatter(plot_grid, c_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('consumption')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 5])
mpl.pyplot.title(r"Consumption path induced by a productivity shock at $t=10$")
mpl.pyplot.show()

#compute output path
iter=1
k_init=4
while iter<maxiter-1: 
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    y_path[0,iter]=k_path[0,iter]**(1-theta)*((z*h)**theta)
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

mpl.pyplot.scatter(plot_grid, y_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('output')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 5])
mpl.pyplot.title(r"Output path induced by a productivity shock at $t=10$")
mpl.pyplot.show()

#compute savings=investments path
iter=1
k_init=4
while iter<maxiter-1: 
    i=np.where(k_grid==k_init) #find index of k_init in k_grid
    j=int(i[0])
    l=int(g[0,j])
    i_path[0,iter]=y_path[0,iter]-c_path[0,iter]
    k_init=np.copy(k_path[0,iter]) #it's clumsy, but quick to treat k_init as an operative variable
    iter=iter+1

mpl.pyplot.scatter(plot_grid, i_path, s=1, color='blue')
mpl.pyplot.xlabel('time')
mpl.pyplot.ylabel('savings')
mpl.pyplot.legend()
mpl.pyplot.xlim([0, 200])
mpl.pyplot.ylim([0, 5])
mpl.pyplot.title(r"Savings path induced by a productivity shock at $t=10$")
mpl.pyplot.show()