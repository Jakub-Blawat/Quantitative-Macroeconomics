# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:21:35 2019

@author: Jakub Blawat
"""
import sympy as sp
import numpy as np
import matplotlib as mpl
import scipy as scp
import quantecon as qe
import numpy.polynomial.chebyshev as cheby

#Set parameters:
beta=0.988 #standard US rate of impatience
theta=0.679 #labour share
delta=0.013 #standard US depreciation rate
#h=1 #fixed labour supply
kappa=0
nu=2

qe.tic()

#Algorithm taken from Makoto Nakamura notes: https://sites.google.com/site/makotonakajima/notes.

#1. Set the order of polynomials used for approximation n.
n = 2

#2. Set the number of collocation points used m. It should be the case m > n + 1 for Chebyshev regression.
m = 5

#3. Set a tolerance parameter.
epsilon = 0.01

#4. Set upperbound and lowerbound of the state space.
k_ss=(1/(1-theta)*((1/beta)+delta-1))**(-1/theta)
k_low = 0.1
k_up  = 1.2*k_ss
k=np.linspace(k_low,k_up,100)

#5. Compute the collocation points.
coll_nodes=[]

for i in range(100):
    q=np.cos((((2*i)-1)/200)*np.pi)
    coll_nodes.append(q)
coll_nodes=np.asarray(coll_nodes)

coll_nod=[]
for i in range(100):
    q=((coll_nodes[i])*((k_up - k[0])/2))+ (((k_up + k[0])/2))
    coll_nod.append(q)
coll_nod=np.asarray(coll_nod)
               
#6. Guess for the level of the value function at the points.
y_0=np.ones(100)

#7. Get the Chebyshev coefficient.
coeff=np.polynomial.chebyshev.chebfit(coll_nod,y_0,n)

#8. Set an initial guess.
V_0=cheby.chebval(k,coeff)

#9. Get the feasible return matrix.

Z_1,Z_2=np.meshgrid(k,k)

def feasibility(z_1,z_2):
    return z_1**(1-theta)+(1-delta)*z_1-z_2

N = feasibility(Z_1,Z_2)

def utility(z_1,z_2):
    for i,j in zip(range(0,100),range(0,100)):
        if N[i,j]>=0:
            return np.log10(z_1**(1-theta)+(1-delta)*z_1-z_2)-(kappa/(1+(1/nu)))
        else:
            return -1000

M = utility(Z_1,Z_2)

#10. For each i in (1,m) solve the maximization problem:
X_= np.empty(shape=[100, 100])
G_0 = np.empty(shape=[100,1])
for i,j in zip(range(0,100),range(0,100)):
    X_[i,j]=M[i,j]+(beta*V_0[j])
    X_[np.isnan(X_)] = -1000    
for i in range(0,100):
    G_0[i]=np.argmax(X_[i,:])
    
#11. Obtain updated guess for the coeffcients
y_1=np.empty(shape=[100,1])
for i in range(0,100):
    y_1[i]=utility(k[i],G_0[i])+(beta*V_0[i])
y_1[np.isnan(y_1)] = 0
y_1=np.reshape(y_1, (100,))

#12. Compare coefficients:
coeff1=cheby.chebfit(coll_nod,y_1,n)

check=np.amax(abs(coeff - coeff1))

#13. Otherwise, update the value function:
count=0

while check>epsilon and count<1000:
    V_=cheby.chebval(k,coeff1) #VF guess
    for i in range(100):
        for j in range(100):
            X_[i,j]=M[i,j]+(beta*V_[j])
            X_[np.isnan(X_)] = -1000     
        for i in range(0,100): #Policy function
            G_0[i]=np.argmax(X_[i,:]) 
        for i in range(0,100): #Update the value function
            y_1[i]=utility(k[i],G_0[i])+(beta*V_[i])
            y_1[np.isnan(y_1)] = 0
            y_1=np.reshape(y_1, (100,))        
        coeff1=cheby.chebfit(coll_nod,y_1,n)
        check=np.amax(abs(coeff - coeff1))
        count +=1
        
#14. After we obtain convergence (according to our predetermined criteria), we should check the validity of the settings.
#First of all, check if the bounds on the state space are not binding.
print('Validity check:')
if k_low < min(G_0):
    print('K lowerbound - passed')
else:
    print('K lowerbound - there must be an error somewhere ;C')

if k_up > max(G_0):
    print('K upperbound - passed')
else:
    print('K upperbound - there must be an error somewhere ;C')

#15. We also increase n, increase m, and reduce epsilon and make sure that the results are not sensitive to the changes.
print('Number of iterations:',count)
qe.toc()

mpl.pyplot.plot(k,V_)
#mpl.pyplot.xlim([0,40])
mpl.pyplot.title('Chebyshev approximation')
