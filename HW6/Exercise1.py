# -*- coding: utf-8 -*-
"""
@author: Jakub Blawat
"""

import matplotlib.pyplot as plt
import numpy as np

# We will assume standard values for beta and inte
beta  = 0.98
R     = 1.05
theta = 0.50

# Define help variables
mT_1     = (pow(beta,-1/theta)*pow(R,(theta-1)/theta))/(1+(pow(beta,-1/theta)*pow(R,(theta-1)/theta)))
gammaT_1 = pow(mT_1,1-theta) + beta*pow((1-mT_1)*R,1-theta)
mT_2     = (pow(R,(theta-1)/theta)*pow(beta*gammaT_1,-1/theta))/(1+pow(R,(theta-1)/theta)*pow(beta*gammaT_1,-1/theta))

# Grid for wealth:
wT   = np.linspace(0,100,101)
wT_1 = np.linspace(0,100,101)
wT_2 = np.linspace(0,100,101)

#Value Functions:
V_T  = (pow(wT,1-theta))/(1-theta)
V_T1 = (pow(wT_1,1-theta)*gammaT_1)/(1-theta)
V_T2 = (pow(wT_2,1-theta)*(1+beta*pow((1-mT_2)*R,1-theta)*gammaT_1))/(1-theta)

# Consumption policy functions:
cT  = wT
cT_1 = mT_1*wT_1
cT_2 = mT_2*wT_2

#Plots (value functions):
plt.plot(wT, V_T, label=r'$V_T$')
plt.plot(wT_1, V_T1, label=r'$V_{T-1}$')
plt.plot(wT_2, V_T2, label=r'$V_{T-2}$')
plt.legend(loc="lower right")
plt.title('Value functions')
plt.show()

#Plots (consumption policy):
plt.plot(wT, cT, label=r'$c_T$')
plt.plot(wT_1, cT_1, label=r'$c_{T-1}$')
plt.plot(wT_2, cT_2, label=r'$c_{T-2}$')
plt.legend(loc="lower right")
plt.title('Consumption policy functions')
plt.show()

#Plots (consumption policy as function of cash on hand):
