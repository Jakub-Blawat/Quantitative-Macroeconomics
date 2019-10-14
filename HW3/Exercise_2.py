# -*- coding: utf-8 -*-
"""
@author: Jakub Blawat
"""

import numpy as np
from scipy.optimize import fsolve

#Parameters:
name=['Country A', 'Country B']
kappa = 5
nu   = 1
sigma = 0.8
eta_ll = [0.5, 2.5]
eta_hh = [5.5, 3.5]
zeta  = 1
theta = 0.6
cap_up = 2
lamm = [0.95, 0.84]
phi   = 0.2

cap_ll = [1, 1]
cap_hh = [1,1]

# Solving our set of equations:

#0 - r, 1-w, 2->hl, 3->hH, 4->cL, 5->cH,
def f(x, eta_l, eta_h, lam, cap_l, cap_h):
    # From firms problem:
    f1 = (1-theta)*(zeta)*pow(cap_up,-theta)*pow(x[2]*eta_l+x[3]*eta_h, theta)-x[0]
    f2 = (theta)*(zeta)*pow(cap_up,1-theta)*pow(x[2]*eta_l+x[3]*eta_h, theta-1)-x[1]
    # From HHs problem:
    f3 = (lam)*(1-phi)*pow(x[1]*(eta_h),1-phi)*pow(x[5],-sigma)-kappa*pow(x[3],(phi) + (1/nu))
    f4 = (lam)*(1-phi)*pow(x[1]*(eta_l),1-phi)*pow(x[4],-sigma)-kappa*pow(x[2],(phi) + (1/nu))
    # Budget constraints
    f5 =  (lam)*pow(x[1]*x[3]*eta_h,1-phi)+x[0]*pow(cap_h, eta_h)-x[5]
    f6 =  (lam)*pow(x[1]*x[2]*eta_l,1-phi)+x[0]*pow(cap_l, eta_l)-x[4]
    return[f1,f2,f3,f4,f5,f6]
    
equil = np.empty([len(name), 6], dtype=object)
for i in range(len(name)):
    f_i = lambda x : f(x, eta_ll[i], eta_hh[i], lamm[i], cap_ll[i], cap_hh[i])
    equil[i] = (fsolve(f_i, [1,1,1,1,1,1]))

#Equilibrium:
    print(str(name[i]),'- Rate of return:', round(equil[i,0],2))
    print(str(name[i]),'- Wages:', round(equil[i,1],2))  
    print(str(name[i]),'- Labor supply (low type)', round(equil[i,2],2))
    print(str(name[i]),'- Labor supply (high type):', round(equil[i,3],2))
    print(str(name[i]),'- Consumption (low types):', round(equil[i,4],2))
    print(str(name[i]),'- Consumption (high type):', round(equil[i,5],2))

# Ex 2.2 Open Economy

kappa = 5
nu   = 1
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

#0 - rA, 1-wA, 2->hlA, 3->hHA, 4->cLA, 5->cHA, 6-kLsA, 7kHsA,
#8- rB, 9-wB, 10->hlB, 11->hHB, 12->cLB, 13->cHB, 14-kLsB, 15-kHsB
def g(x):
    # From firms problem:
    ##rates
    g1 = (1-theta)*(zeta)*pow(x[6]+x[7]+cap_up-x[14]-x[15],-theta)*pow(x[2]*eta_l_A+x[3]*eta_h_A, theta)-x[0]
    g2 = (1-theta)*(zeta)*pow(x[14]+x[15]+cap_up-x[6]-x[7],-theta)*pow(x[10]*eta_l_B+x[11]*eta_h_B, theta)-x[8]
    ##wages
    g3 = (theta)*(zeta)*pow(x[6]+x[7]+cap_up-x[14]-x[15],1-theta)*pow(x[2]*eta_l_A+x[3]*eta_h_A, theta-1)-x[1]
    g4 = (theta)*(zeta)*pow(x[14]+x[15]+cap_up-x[6]-x[7],1-theta)*pow(x[10]*eta_l_B+x[11]*eta_h_B, theta-1)-x[9]
    # From HHs problem:
    ##high_type
    g5 = (lambda_A)*(1-phi)*pow(x[1]*(eta_h_A),1-phi)*pow(x[5],-sigma)-kappa*pow(x[3],phi + (1/nu))
    g6 = (lambda_B)*(1-phi)*pow(x[9]*(eta_h_B),1-phi)*pow(x[13],-sigma)-kappa*pow(x[11],phi + (1/nu))
    ##low type
    g7 = (lambda_A)*(1-phi)*pow(x[1]*(eta_l_A),1-phi)*pow(x[4],-sigma)-kappa*pow(x[2],phi + (1/nu))
    g8 = (lambda_B)*(1-phi)*pow(x[9]*(eta_l_A),1-phi)*pow(x[12],-sigma)-kappa*pow(x[10],phi + (1/nu))
    #capital optimum
    ##high type
    g9 = eta_h_A*x[0]*pow(x[7], eta_h_A-1)-x[8]
    g10 = eta_h_B*x[8]*pow(x[15], eta_h_B-1)-x[0]
    ##low type
    g11 = eta_l_A*x[0]*pow(x[6], eta_l_A-1)-x[8]
    g12 = eta_l_B*x[8]*pow(x[14], eta_l_B-1)-x[0]
    # Budget constraints
    ##high type
    g13 =  (lambda_A)*pow(x[1]*x[3]*eta_h_A,1-phi)+x[0]*pow(x[7], eta_h_A)+x[8]*(cap_h_A-x[7])-x[5]
    g14 =  (lambda_B)*pow(x[9]*x[11]*eta_h_B,1-phi)+x[8]*pow(x[15], eta_h_B)+x[0]*(cap_h_B-x[7])-x[13]
    ##low type    
    g15 =  (lambda_A)*pow(x[1]*x[2]*eta_l_A,1-phi)+x[0]*pow(x[6], eta_l_A)+x[8]*(cap_l_A-x[6])-x[4]
    g16 =  (lambda_B)*pow(x[9]*x[10]*eta_l_B,1-phi)+x[8]*pow(x[14], eta_l_B)+x[0]*(cap_l_B-x[6])-x[12]
    
    return[g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16]

equil_open = fsolve(g, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

#Equilibrium 
print('Open Economy:')
print('Equilibrium in capital union:')
print('A-Rate of return:', round(equil_open[0],2))
print('A-Wages:', round(equil_open[1],2))  
print('A-Labor supply (low type)', round(equil_open[2],2))
print('A-Labor supply (high type):', round(equil_open[3],2))
print('A-Consumption (low types):', round(equil_open[4],2))
print('A-Consumption (high type):', round(equil_open[5],2))
print('A-domestic capital supply (low types):', round(equil_open[6],2))
print('A-domestic capital supply (high type):', round(equil_open[7],2))
print('B-Rate of return:', round(equil_open[8],2))
print('B-Wages:', round(equil_open[9],2))  
print('B-Labor supply (low type)', round(equil_open[10],2))
print('B-Labor supply (high type):', round(equil_open[11],2))
print('B-Consumption (low types):', round(equil_open[12],2))
print('B-Consumption (high type):', round(equil_open[13],2))
print('B-domestic capital supply (low types):', round(equil_open[14],2))
print('B-domestic capital supply (high type):', round(equil_open[15],2))