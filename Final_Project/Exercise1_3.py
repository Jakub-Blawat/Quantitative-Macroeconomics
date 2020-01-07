# -*- coding: utf-8 -*-
"""
@author: Jakub Bławat
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import quantecon as qe
from sklearn.linear_model import LinearRegression

#--------------------------Random seed generation------------------------------
np.random.seed(seed=885743368)

#--------------------------Parametrization-------------------------------------
alpha = 0.3
beta  = pow(0.99, 40)
tau   = 0.0
lambd = 0.5
g     = 0.0 
exp_zeta = 1.0
exp_rho  = 1.0
exp_eta  = 1.0
periods  = 50000
std_log_zeta = 0.13
std_log_rho  = 0.50
std_log_eta  = 0.95
nodes = 11

#-------------------------------------Shocks-----------------------------------

# State drawing with equal probabilities.  If =1 then boom, if=-1 then recession  
z=np.random.choice([-1,1], periods, [0.5, 0.5])
# Zeta equal to std if boom and -std when recession
log_zeta= np.array(z)*std_log_zeta
# Rho equal to std if boom and -std when recession
log_rho = np.array(z)*std_log_rho
# And eta (using quantecon.quad.qnwnorm)
eta_nodes = qe.quad.qnwnorm(nodes, 0, std_log_eta**2)
eta = np.random.choice(eta_nodes[0], periods, 1/11)
expctd_eta = np.mean(eta)

#----- a) Compute the theoretical values of  psi, which follow from (1).-------

# Phi from the HL paper
phi_ss = (1)/(1+((1-alpha)*lambd*exp_eta)/(alpha*(1+lambd)*exp_rho))

# Saving rate:
sav_rate_ss = (beta*phi_ss)/(1 + (beta*phi_ss))

# Computing psi zero
psi_0 = math.log(sav_rate_ss) + math.log(1-tau) + math.log((1-alpha)/(1-lambd)) + math.log(exp_zeta)

# Computing psi one
psi_1 = alpha

print('Subpoint a)')
print('Theoretical value of psi zero is:', psi_0)
print('Theoretical value of psi one is:', psi_1)

#-----b) Carry out the algorithm to convergence--------------------------------
#--------i) In iteration m, solve the household problem------------------------

#Defining capital path
def capital_path(prev_cap):
    ln_cap = psi_0 + psi_1*prev_cap
    return ln_cap

#Setting initial values
ln_k = [1]
iter = 0
t = [0]

# Iteration for capital path
while iter<periods:
    ln_k.append(capital_path(ln_k[iter]))
    iter+=1
    t.append(iter)
# ^It converges in 31 iterations^


# Steady state (we calculate it to compare and set the grid
ln_k_ss = (psi_0)/(1-alpha)
k_ss = math.exp(ln_k_ss)

print('Steady state k:', k_ss, 'and log(k):', ln_k_ss) 

# Setting the grid
k_grid = np.linspace(start=0.5*k_ss, stop=1.5*k_ss, num=5)

# Wages (Equation 3b from the HL paper)
def wage(ka, zeta1):
    w = (1-alpha)*(1+g)*pow(ka, alpha)*zeta1
    return w

# Interest rate (from Equation 3a)
def rate(ka, zeta1, rho1):
    r = alpha*pow(ka, alpha-1)*zeta1*rho1
    return r

# Assets
def assets(k1):
    a = k1*(1+lambd)
    return a

# Equation 2a
def c_young(ka, k1, zeta1):
    c1 = (1-tau)*wage(ka, zeta1) - assets(k1) 
    return c1

# Equation 2b
def c_old(k1, zeta1):
    c2 = assets(k1)*rate(k1, exp_zeta, exp_rho)+ lambd*zeta1*wage(k1, zeta1)*(1-tau)
    return c2
    
# Inititial values:
k = np.exp(ln_k)    
c1 = (1-tau)*wage(k[0], np.exp(log_zeta[0])) - assets(k[1])
a1 = assets(k[1])
s1 = (a1)/((1-tau)*wage(k[0], np.exp(log_zeta[0])))

#def Euler():
 #   s = 

# Setting the first value of the list
sav_rate = [s1]

#Iteration:
iter = 0
while iter < periods:
    sav_rate.append((assets(k[iter+1]))/(((1-tau)*wage(k[iter], np.exp(log_zeta[iter])))))
    iter+=1

# The soluton is the list ------>sav_rate

#Plots
plt.plot(t, ln_k, label='Exercise 1.3 - Capital path')
plt.title('Capital over time (log) (T=50)')
plt.xlabel('Periods')
plt.ylabel('Capital stock (in logs)')
plt.xlim(0,50)
plt.show()

plt.plot(t, sav_rate, label='Exercise 1.3 - Saving rate')
plt.title('Saving rate over time for T=100')
plt.xlabel('Periods')
plt.ylabel('Capital stock (in logs)')
plt.xlim(0,100)
plt.ylim(0.7,1)
plt.show()


#--------------------------------ii) Simulation--------------------------------
#Now we simulate for the Equation (1) from the assignment
def capital_path1(iter1, shock1, prev_cap):
    ln_cap =np.log(sav_rate[iter1]) + math.log(1-tau) + math.log((1-alpha)/(1-lambd)) + shock1 + alpha*prev_cap
    return ln_cap

#Initial ln_k
ln_k1=[ln_k_ss]

# Iteration
iter = 0
while iter < periods:
    ln_k1.append(capital_path1(iter, log_zeta[iter], ln_k1[iter]))
    iter+=1
    
# Plot
plt.plot(t, ln_k1, label='Exercise 1.3b.ii - Capital path')
plt.title('Capital over time (log) (T=50)')
plt.xlabel('Periods')
plt.ylabel('Capital stock (in logs)')
plt.xlim(0,100)
plt.show()


#------------------------------iii) Regression---------------------------------
# I've used simple linear regression from sklearn.linear_model package

# Discarding first 500  periods 
ln_k_short = np.array(ln_k1[500:])
t_short = np.array(t[500:]).reshape(-1, 1)

# Regression
model = LinearRegression()
model.fit(t_short, ln_k_short)

# predict y from the data
x_new = np.linspace(0, 50000, 50000)
y_new = model.predict(x_new[:, np.newaxis])

# plot the results
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.scatter(t_short, ln_k_short)
ax.plot(x_new, y_new)
ax.set_xlabel('t')
ax.set_ylabel('ln(k1)')
ax.axis('tight')
plt.xlim(500,600)
plt.show()


#------------------------------c) Comparison-----------------------------------

# In .pdf file

#-----------------------------d) For tau=0.1-----------------------------------
tau1 = 0.1

phi_ss_new = (1)/(1+((1-alpha)*lambd*exp_eta*tau*(1+lambd*(1-exp_eta)))/(alpha*(1+lambd)*exp_rho))

# Saving rate:
sav_rate_ss_new = (beta*phi_ss_new)/(1 + (beta*phi_ss_new))

# Computing psi zero
psi_0_new = math.log(sav_rate_ss_new) + math.log(1-tau1) + math.log((1-alpha)/(1-lambd)) + math.log(exp_zeta)

# Computing psi one
psi_1_new = alpha

print('Subpoint d):')
print('Theoretical value of psi 0 and tau=0.1 is:', psi_0_new)
print('Theoretical value of psi 1 and tau=0.1 is:', psi_1_new)


def capital_path_new(prev_cap_new):
    ln_cap_new = psi_0_new + psi_1_new*prev_cap_new
    return ln_cap_new

#Setting initial values
ln_k_new = [1]
iter = 0
t_new = [0]

# Iteration for capital path
while iter<periods:
    ln_k_new.append(capital_path_new(ln_k_new[iter]))
    iter+=1
    t_new.append(iter)
# ^It converges in 31 iterations^


# Steady state (we calculate it to compare and set the grid
ln_k_ss_new = (psi_0_new)/(1-alpha)
k_ss_new = math.exp(ln_k_ss_new)

print('New steady state is given by k:', k_ss_new, 'and log(k):', ln_k_ss_new) 


















