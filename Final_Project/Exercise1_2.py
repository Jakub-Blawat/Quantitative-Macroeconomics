# -*- coding: utf-8 -*-
"""
@author: Jakub Bławat
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import quantecon as qe

#--------------------------Random seed generation------------------------------
np.random.seed(seed=885743368)
#--------------------------Parametrization-------------------------------------
alpha = 0.3
beta  = pow(0.99, 40)
tau   = 0.0
lambd = 0.5
g     = 0.0 
periods  = 50000
exp_zeta = 1.0
exp_rho  = 1.0
exp_eta  = 1.0
std_log_zeta = 0.13
std_log_rho  = 0.50
std_log_eta  = 0.95
nodes  = 11

#---------------------------Steady state---------------------------------------
# Phi as in the paper (for tau=0 and mean values of shocks).
phi_ss = (1)/(1+((1-alpha)*(lambd*exp_eta))/(alpha*(1+lambd)*exp_rho))

# Saving rate:
sav_rate_ss = (beta*phi_ss)/(1 + (beta*phi_ss))

# Steady state capital stock (for ln(zeta)=0): 
ln_k_ss = (math.log(sav_rate_ss) + math.log(1-tau) + math.log((1-alpha)/(1-lambd)))/(1-alpha)
k_ss = math.exp(ln_k_ss)
print('Steady state log of capital stock:', ln_k_ss)
print('Steady state capital stock:', k_ss)

#-----------------------------1st simulation-----------------------------------
# For tau=0, E(zeta)=E(eta)=E(rho)=1. 
# Phi and saving rate are the same as in the steady state.  

#Random drawing of the shocks from log-normal distribution
zeta  = np.random.lognormal(mean=math.log(exp_zeta), sigma=std_log_zeta, size=periods)
rho   = np.random.lognormal(mean=math.log(exp_rho), sigma=std_log_rho, size=periods)
eta   = np.random.lognormal(mean=math.log(exp_eta), sigma=std_log_eta, size=periods)

# Defining function for capital path.
def capital_path(shock, prev_cap):
    ln_cap = math.log(sav_rate_ss) + math.log(1-tau) + math.log((1-alpha)/(1-lambd)) + math.log(shock) + alpha*prev_cap
    return ln_cap

# Defining initial capital stock 
ln_k=[ln_k_ss]
t=[0]
iter = 0

# Iteration
while iter<periods:
    ln_k.append(capital_path(zeta[iter], ln_k[iter]))
    iter+=1
    t.append(iter)

#--------------------------------Plotting--------------------------------------
plt.plot(t, ln_k, label='Exercise 1.2 - 1st simualtion')
plt.title('Capital path over time (log)')
plt.xlabel('Periods')
plt.ylabel('Capital stock (in logs)')
plt.show()

plt.plot(t, ln_k, label='Exercise 1.2 - 1st simulation')
plt.title('Capital path over time (in log) for T=100')
plt.xlabel('Periods')
plt.ylabel('Capital stock (in logs)')
plt.xlim(0,100)
plt.show()

#---------------2nd simulation - with discrete shocks--------------------------

# State drawing with equal probabilities.  If =1 then boom, if=-1 then recession  
z=np.random.choice([-1,1], periods, [0.5, 0.5])
# Zeta equal to std if boom and -std when recession
log_dis_zeta= np.array(z)*std_log_zeta
# Rho equal to std if boom and -std when recession
log_dis_rho = np.array(z)*std_log_rho
# And eta (using quantecon.quad.qnwlogn)
eta_nodes = qe.quad.qnwlogn(nodes, 0, std_log_eta**2)
log_dis_eta = np.random.choice(eta_nodes[0], periods, 1/11)
expctd_eta = np.mean(log_dis_eta)

# Phi function:
def phi_func(rho1, eta1):
    phi_val = (1)/(1+((1-alpha)*lambd*expctd_eta)/(alpha*(1+lambd)*np.exp(rho1)))
    return phi_val

# Saving rate function:
def sav_rate_func(rho2, eta2):
    sav_rate_val = (beta*phi_func(rho2, eta2))/(1 + (beta*phi_func(rho2, eta2)))
    return sav_rate_val

# Capital path:
def capital_path1(shock1, shock2, shock3, prev_cap):
    ln_cap =np.log(sav_rate_func(shock2, shock3)) + math.log(1-tau) + math.log((1-alpha)/(1-lambd)) + shock1 + alpha*prev_cap
    return ln_cap

# Defining initial capital stock 
ln_k1=[ln_k_ss]
log_dis_test = log_dis_eta[2]

# Iteration
iter = 0
while iter < periods:
    ln_k1.append(capital_path1(log_dis_zeta[iter], log_dis_rho[iter], log_dis_eta[iter], ln_k1[iter]))
    iter+=1
    
# From logs to capital
k =  np.exp(ln_k1)       
    
#--------------------------------Plotting--------------------------------------
plt.scatter(t, k, s=0.2, label='Exercise 1.2-Discrete')
plt.title('Scatter plot for discretized shocks')
plt.xlabel('Periods')
plt.ylabel('Capital stock')
plt.ylim(0, 0.05)
plt.show()

plt.plot(t, k, label='Exercise 1.2 - 2nd simulation')
plt.title('Capital path over time  for T=100')
plt.xlabel('Periods')
plt.ylabel('Capital stock')
plt.ylim(0, 0.05)
plt.xlim(1,100)
plt.show()