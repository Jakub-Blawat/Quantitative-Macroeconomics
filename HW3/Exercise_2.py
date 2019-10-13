# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:58:12 2019

@author: kuba
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 22:26:17 2019

@author: kuba
"""

#import sympy as sy
#import numpy as ny
#import matplotlib as mpl
from scipy.optimize import fsolve

#Parameters:
kappa = 5
nu    = 1
sigma = 0.8
eta_l_A = 0.5
eta_l_B = 2.5
eta_h_A = 5.5
eta_h_B = 3.5
zeta  = 1
theta = 0.6
cap_up = 2
lambda_A = 0.95
lambda_B = 0.84
phi   = 0.2

cap_l_A = 1
cap_h_A = 1
cap_l_B = 1
cap_h_B = 1

# Solving our set of equations:
 
#Country A

def f(x):
    # From firms problem:
    f1 = (1-theta)*(zeta)*pow(cap_up,-theta)*pow(x[2]*eta_l_A+x[3]*eta_h_A, theta)-x[0]
    f2 = (theta)*(zeta)*pow(cap_up,1-theta)*pow(x[2]*eta_l_A+x[3]*eta_h_A, theta-1)-x[1]
    # From HHs problem:
    f3 = (lambda_A)*(1-phi)*pow(x[1]*(eta_h_A),1-phi)*pow(x[5],-sigma)-kappa*pow(x[3],(eta_h_A) + (1/nu))
    f4 = (lambda_A)*(1-phi)*pow(x[1]*(eta_l_A),1-phi)*pow(x[4],-sigma)-kappa*pow(x[2],(eta_l_A) + (1/nu))
    # Budget constraints
    f5 =  (lambda_A)*pow(x[1]*x[3]*eta_h_A,1-phi)+x[0]*pow(cap_h_A,eta_h_A)-x[5]
    f6 =  (lambda_A)*pow(x[1]*x[2]*eta_l_A,1-phi)+x[0]*pow(cap_l_A,eta_l_A)-x[4]
    return[f1,f2,f3,f4,f5,f6]

equil_A = fsolve(f, [1,1,1,1,1,1])

#Equilibrium for country A
print('A-Rate of return:', round(equil_A[0],2))
print('A-Wages:', round(equil_A[1],2))  
print('A-Labor supply (low type)', round(equil_A[2],2))
print('A-Labor supply (high type):', round(equil_A[3],2))
print('A-Consumption (low types):', round(equil_A[4],2))
print('A-Consumption (high type):', round(equil_A[5],2))

# Country B

def g(x):
    # From firms problem:
    g1 = (1-theta)*(zeta)*pow(cap_up,-theta)*pow(x[2]*eta_l_B+x[3]*eta_h_B, theta)-x[0]
    g2 = (theta)*(zeta)*pow(cap_up,1-theta)*pow(x[2]*eta_l_B+x[3]*eta_h_B, theta-1)-x[1]
    # From HHs problem:
    g3 = (lambda_B)*(1-phi)*pow(x[1]*(eta_h_B),1-phi)*pow(x[5],-sigma)-kappa*pow(x[3],(eta_h_B) + (1/nu))
    g4 = (lambda_B)*(1-phi)*pow(x[1]*(eta_l_B),1-phi)*pow(x[4],-sigma)-kappa*pow(x[2],(eta_l_B) + (1/nu))
    # Budget constraints
    g5 =  (lambda_A)*pow(x[1]*x[3]*eta_h_B,1-phi)+x[0]*pow(cap_h_B,eta_h_B)-x[5]
    g6 =  (lambda_A)*pow(x[1]*x[2]*eta_l_B,1-phi)+x[0]*pow(cap_l_B,eta_l_B)-x[4]
    return[g1,g2,g3,g4,g5,g6]

equil_B = fsolve(g, [1,1,1,1,1,1])

#Equilibrium for country B
print('B-Rate of return:', round(equil_B[0],2))
print('B-Wages:', round(equil_B[1],2))  
print('B-Labor supply (low type)', round(equil_B[2],2))
print('B-Labor supply (high type):', round(equil_B[3],2))
print('B-Consumption (low types):', round(equil_B[4],2))
print('B-Consumption (high type):', round(equil_B[5],2))
