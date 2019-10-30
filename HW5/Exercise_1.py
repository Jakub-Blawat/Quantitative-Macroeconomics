# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:43:47 2019

@author: Jakub Bławat
"""

import numpy as np
import matplotlib as mpl
import random

random.seed(42693)

# Question 1 

# 1)

######################################################################
## We draw 10^7 observations from joint normal distribution and use 
# numpy.random.multivariate_normal(mean, cov[, size, check_valid, tol])
# to generate variables
######################################################################

mean = [-0.5, -0.5]
cov = [[1, 0], [0, 1]] #we need to have identity matrix as a variance-covariance matrix to assure no correlation
log_k, log_z = np.random.multivariate_normal(mean, cov, 10**7).T



# Plotting k,z in logs

mpl.pyplot.scatter(log_k, log_z, s=1, color='blue')
mpl.pyplot.show()


k = np.exp(log_k)
z = np.exp(log_z)

print("Average k:", k.mean())
print("Avergae z:", z.mean())

# Plotting k,z in levels

mpl.pyplot.scatter(k, z, s=1, color='blue')
mpl.pyplot.show()

# 2)

gamma = 0.6

def y(cap,s):
    output = pow(s,1-gamma)*pow(cap,gamma)
    return output

prod = y(k, z)

agg_prod = sum(prod)
print("Aggregate production:", agg_prod)
print("Average production:", prod.mean())

# 3)
k_opt = np.array(sorted(k, reverse = True))
z_opt = np.array(sorted(z, reverse = True))

prod_opt = y(k_opt, z_opt)
agg_prod_opt = sum(prod_opt)
capital_opt = sum(k_opt)

print("Aggregate optimal production:", agg_prod_opt)
print("Average optimal production:", prod_opt.mean())


#4)
# Computing optimal allociations against the data
k_atd = k_opt - k

#5)
# Reallocation problem
agg_prod = sum(prod)
prod_gain = agg_prod_opt - agg_prod
prod_gain_percent = (agg_prod_opt/agg_prod - 1) * 100
prod_gain_pc = prod_gain/(10**7)

print("Production gain nominally:", prod_gain)
print("Production gain percentage:", prod_gain_percent)
print("Production gain per capita:", prod_gain_pc)

#6)
# Same code but with different correlations:

########################### For correaltion equal to 0.5:######################
mean_a = [-0.5, -0.5]
cov_a = [[1, 0.5], [0.5, 1]] 
log_k_a, log_z_a = np.random.multivariate_normal(mean_a, cov_a, 10**7).T


mpl.pyplot.scatter(log_k_a, log_z_a, s=1, color='blue')
mpl.pyplot.show()


k_a = np.exp(log_k_a)
z_a = np.exp(log_z_a)

print("Average k:", k_a.mean())
print("Avergae z:", z_a.mean())

# Plotting k,z in levels

mpl.pyplot.scatter(k_a, z_a, s=1, color='blue')
mpl.pyplot.show()

# 6.1.2)
def y(cap,s):
    output = pow(s,1-gamma)*pow(cap,gamma)
    return output

prod_a = y(k_a, z_a)

agg_prod_a = sum(prod_a)
print("Aggregate production:", agg_prod_a)
print("Average production:", prod_a.mean())

# 6.1.3)
k_a_opt = np.array(sorted(k_a, reverse = True))
z_a_opt = np.array(sorted(z_a, reverse = True))

prod_opt_a = y(k_a_opt, z_a_opt)
agg_prod_opt_a = sum(prod_opt_a)
capital_opt_a = sum(k_a_opt)

print("Aggregate optimal production:", agg_prod_opt_a)
print("Average optimal production:", prod_opt_a.mean())


#6.1.4)
# Computing optimal allociations against the data
k_a_atd = k_a_opt - k_a

#6.1.5)
# Reallocation problem
agg_prod_a = sum(prod_a)
prod_gain_a = agg_prod_opt_a - agg_prod_a
prod_gain_percent_a = (agg_prod_opt_a/agg_prod_a - 1) * 100
prod_gain_pc_a = prod_gain_a/(10**7)

print("Production gain nominally:", prod_gain_a)
print("Production gain percentage:", prod_gain_percent_a)
print("Production gain per capita:", prod_gain_pc_a)


############################### For correaltion equal to -0.5:#################

mean_b = [-0.5, -0.5]
cov_b = [[1, -0.5], [-0.5, 1]] 
log_k_b, log_z_b = np.random.multivariate_normal(mean_b, cov_b, 10**7).T


mpl.pyplot.scatter(log_k_b, log_z_b, s=1, color='blue')
mpl.pyplot.show()


k_b = np.exp(log_k_b)
z_b = np.exp(log_z_b)

print("Average k:", k_b.mean())
print("Avergae z:", z_b.mean())

# Plotting k,z in levels

mpl.pyplot.scatter(k_b, z_b, s=1, color='blue')
mpl.pyplot.show()

# 6.2.2)
def y(cap,s):
    output = pow(s,1-gamma)*pow(cap,gamma)
    return output

prod_b = y(k_b, z_b)

agg_prod_b = sum(prod_b)
print("Aggregate production:", agg_prod_b)
print("Average production:", prod_b.mean())

# 6.2.3)
k_b_opt = np.array(sorted(k_b, reverse = True))
z_b_opt = np.array(sorted(z_b, reverse = True))

prod_opt_b = y(k_b_opt, z_b_opt)
agg_prod_opt_b = sum(prod_opt_b)
capital_opt_b = sum(k_b_opt)

print("Aggregate optimal production:", agg_prod_opt_b)
print("Average optimal production:", prod_opt_b.mean())


#6.2.4)
# Computing optimal allociations against the data
k_b_btd = k_b_opt - k_b

#6.2.5)
# Reallocation problem

agg_prod_b = sum(prod_b)
prod_gain_b = agg_prod_opt_b - agg_prod_b
prod_gain_percent_b = (agg_prod_opt_b/agg_prod_b - 1) * 100
prod_gain_pc_b = prod_gain_b/(10**7)

print("Production gain nominally:", prod_gain_b)
print("Production gain percentage:", prod_gain_percent_b)
print("Production gain per capita:", prod_gain_pc_b)