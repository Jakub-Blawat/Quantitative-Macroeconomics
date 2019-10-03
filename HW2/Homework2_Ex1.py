# -*- coding: utf-8 -*-
"""
@author: Jakub Bławat
"""

import sympy as sy
import numpy as ny
import matplotlib as mpl
import numpy.polynomial.polynomial as poly
from numpy import inf
import numpy.polynomial.chebyshev as cheby

#mpl.pyplot.style.use("ggplot")

# Exercise 1.1
# Define the variable and the function that we are going to approximate
x = sy.Symbol('x')
f = x**(0.321)

# Define factorial function
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n*factorial(n-1)

# Define Taylor approximation function
def taylor(function,x0,n):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x,i).subs(x,x0))/(factorial(i))*(x-x0)**i
        i += 1
    return p

# Plot
def plot():
    x_grid = [0,4]
    x1 = ny.linspace(x_grid[0],x_grid[1],500)
    y1 = [] 
    order = [1,2,5,20]
    for j in order:
        func = taylor(f,1,j)
        for k in x1:
            y1.append(func.subs(x,k))
        mpl.pyplot.plot(x1,y1,label='Order '+str(j))
        y1 = []
    mpl.pyplot.plot(x1,x1**(0.321),label=r'$x^{0.321}$')
    mpl.pyplot.xlim(x_grid)
    mpl.pyplot.ylim([-10,10])
    mpl.pyplot.xlabel('x')
    mpl.pyplot.ylabel('y')
    mpl.pyplot.legend()
    mpl.pyplot.title('Taylor approximation')
    mpl.pyplot.show()
plot()

# Exercise 1.2. Taylor approximation of the ramp function.
x = sy.Symbol('x', real=True)
f = (x + abs(x))*0.5

#Creating the new plot function similar to previous one.
def plot2():
    x_grid = [-2,6]
    x1 = ny.linspace(x_grid[0],x_grid[1],500)
    y1 = []
    order = [1,2,5,20]
    for j in order:
        func = taylor(f,2,j)
        for k in x1:
            y1.append(func.subs(x,k))
        mpl.pyplot.plot(x1,y1,label='Order '+str(j))
        y1 = []
    mpl.pyplot.plot(x1, (x1 + abs(x1))*0.5,label=r'$\frac{x+\left|x\right|}{2}$')
    mpl.pyplot.xlim(x_grid)
    mpl.pyplot.ylim([-2,6])
    mpl.pyplot.xlabel('x')
    mpl.pyplot.ylabel('y')
    mpl.pyplot.legend()
    mpl.pyplot.title('Taylor approximation of the ramp function')
    mpl.pyplot.show()
plot2()


# Exercise 1.3.1

#---------------------Runge function------------------------------------------
def f(x):
	return 1.0 / (1.0 + 25.0*x**2)

# Evenly spaced interpolation nodes:
x_grid=[-1,1]
x1 = ny.linspace(x_grid[0],x_grid[1],11) 
y = f(x1)
xs = ny.linspace(x_grid[0],x_grid[1],101)

#Order3
runge3_coef = poly.polyfit(x1,y,3)
runge3_f = poly.polyval(xs,runge3_coef)

#Order 5
runge5_coef = poly.polyfit(x1,y,5)
runge5_f = poly.polyval(xs,runge5_coef) 

#Order 10
runge10_coef = poly.polyfit(x1,y,10)
runge10_f = poly.polyval(xs,runge10_coef)

#------------------------Exponential Function----------------------------------
def g(x):
	with ny.errstate(divide='ignore', invalid='ignore'):
		return ny.exp(1/x)

yg = g(x1)
yg[yg == inf] = 9999

#Order3
exp3_coef = poly.polyfit(x1,yg,3)
exp3_g = poly.polyval(xs,exp3_coef)

#Order5
exp5_coef = poly.polyfit(x1,yg,5)
exp5_g = poly.polyval(xs,exp5_coef)

#Order10
exp10_coef = poly.polyfit(x1,yg,10)
exp10_g = poly.polyval(xs,exp10_coef)

#-----------------------Ramp Function------------------------------------------
def h(x):
	return (x+abs(x))*0.5
yh = h(x1)

ramp3_coef = poly.polyfit(x1,yh,3)
ramp3_h = poly.polyval(xs,ramp3_coef)

ramp5_coef = poly.polyfit(x1,yh,5)
ramp5_h = poly.polyval(xs,ramp5_coef)

ramp10_coef = poly.polyfit(x1,yh,10)
ramp10_h = poly.polyval(xs,ramp10_coef)

#---------------------------------Plots----------------------------------------
fig = mpl.pyplot.figure(figsize=(15,10))
fig.suptitle('Evenly-Spaced nodes and monomials')

axis1 = mpl.pyplot.subplot(321)
mpl.pyplot.plot(xs,f(xs), label='Exact function')
mpl.pyplot.plot(xs,runge3_f,label='Cubic Polynomials') 
mpl.pyplot.plot(xs,runge5_f,label='Monomial of order 5')
mpl.pyplot.plot(xs,runge10_f,label='Monomial of order 10')
mpl.pyplot.legend(bbox_to_anchor=(0.25,0), loc="lower center", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.ylim(-2,2)
mpl.pyplot.title(r"Runge Function: $\frac{1}{1 + 25x^{2}}$")

mpl.pyplot.subplot(322, sharex=axis1)
mpl.pyplot.plot(xs,f(xs)-runge3_f,label='Cubic error') 
mpl.pyplot.plot(xs,f(xs)-runge5_f,label='Order 5 error')
mpl.pyplot.plot(xs,f(xs)-runge10_f,label='Order 10 error')
mpl.pyplot.ylim(-0.6,0.6)
mpl.pyplot.legend(bbox_to_anchor=(0.75,0), loc="lower center", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.title('Runge Function Approximation Errors')

mpl.pyplot.subplot(323, sharex=axis1)
mpl.pyplot.plot(xs,g(xs))
mpl.pyplot.plot(xs,exp3_g) 
mpl.pyplot.plot(xs,exp5_g)
mpl.pyplot.plot(xs,exp10_g)
mpl.pyplot.ylim(0,10000)
mpl.pyplot.title(r"Exponential Function: $e^{\frac{1}{x}}$")

mpl.pyplot.subplot(324, sharex=axis1)
mpl.pyplot.plot(xs,g(xs)-exp3_g)
mpl.pyplot.plot(xs,g(xs)-exp5_g)
mpl.pyplot.plot(xs,g(xs)-exp10_g)
mpl.pyplot.ylim(0,10000)
mpl.pyplot.title('Exponential Function Approximation Errors')

mpl.pyplot.subplot(325, sharex=axis1)
mpl.pyplot.plot(xs,h(xs))
mpl.pyplot.plot(xs,ramp3_h) 
mpl.pyplot.plot(xs,ramp5_h)
mpl.pyplot.plot(xs,ramp10_h)
mpl.pyplot.ylim(-0.5,2.5)
mpl.pyplot.title(r"Ramp Function: $\frac{x+|x|}{2}$")

mpl.pyplot.subplot(326, sharex=axis1)
mpl.pyplot.plot(xs,h(xs)-ramp3_h) 
mpl.pyplot.plot(xs,h(xs)-ramp5_h)
mpl.pyplot.plot(xs,h(xs)-ramp10_h)
mpl.pyplot.ylim(-1,1)
mpl.pyplot.title('Ramp Function Approximation Errors')

mpl.pyplot.xlim(x_grid)
mpl.pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
mpl.pyplot.show()

# Exercise 1.3.2
def f(x):
	return 1.0 / (1.0 + 25.0 * x**2)
x_grid=[-1,1]
x1 = ny.linspace(x_grid[0],x_grid[1],11) 
y = f(x1)
N = 11

xs = ny.linspace(x_grid[0],x_grid[1],101)
cs = poly.polyfit(x1,y,3)
fit3 = poly.polyval(xs,cs)
x1 = ny.cos((2 * ny.arange(1, N + 1) - 1) / (2 * N) * ny.pi)
x1 = sorted(x1)
x1 = ny.asarray(x1)
# Runge's function in [-1,1]
def f(x):
	return 1.0 / (1.0 + 25.0 * ny.power(x, 2))
y = f(x1)

cs = poly.polyfit(x1, y,3)
ffit3 = poly.polyval(xs,cs)
xs = ny.linspace(x_grid[0], x_grid[1], 101)

# Monomials of order 5and 10:
coef = poly.polyfit(x1, y, 5)
fit = poly.polyval(xs, coef)
coef10 = poly.polyfit(x1, y, 10)
fit10 = poly.polyval(xs, coef10)

# Exponential Function
def g(x):
	with ny.errstate(divide='ignore', invalid='ignore'):
		return ny.exp(1 / x)

yg = g(x1)
yg[yg == inf] = 9999

cs_g = poly.polyfit(x1, yg,3)
exp3_g = poly.polyval(xs,cs_g)
coef_g = poly.polyfit(x1, yg, 5)
exp5_g = poly.polyval(xs, coef_g) 
exp10_coef = poly.polyfit(x1, yg, 10)
exp10_g = poly.polyval(xs, exp10_coef)


def h(x):
	return .5 * (x + abs(x))
yh = h(x1)

csh = poly.polyfit(x1, yh,3)
ramp3_h = poly.polyval(xs,csh)

# Monomials of order 5 and 10:
ramp_coef = poly.polyfit(x1, yh, 5)
ramp5_h = poly.polyval(xs, ramp_coef)
ramp10_coef = poly.polyfit(x1, yh, 10)
ramp10_h = poly.polyval(xs, ramp10_coef)

#--------------------------------Plots-----------------------------------------
fig = mpl.pyplot.figure(figsize=(15, 10))
fig.suptitle('Czebyszow nodes and monomials')

ax1 = mpl.pyplot.subplot(321)
mpl.pyplot.plot(xs, f(xs), label='Exact Function')
mpl.pyplot.plot(xs, runge3_f, label='Cubic Polynomials')
mpl.pyplot.plot(xs, runge5_f, label='Monomial of order 5')
mpl.pyplot.plot(xs, runge10_f, label='Monomial of order 10')
mpl.pyplot.legend(bbox_to_anchor=(0.25, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.ylim(-1, 1)
mpl.pyplot.title(r"Runge Function: $\frac{1}{1 + 25x^{2}}$")

mpl.pyplot.subplot(322, sharex=ax1)
mpl.pyplot.plot(xs, f(xs) - runge3_f, label='Cubic error')
mpl.pyplot.plot(xs, f(xs) - runge5_f, label='Monomial 5 error')
mpl.pyplot.plot(xs, f(xs) - runge10_f, label='Monomial 10 error')
mpl.pyplot.ylim(-1, 1)
mpl.pyplot.legend(bbox_to_anchor=(0.75, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.title('Runge function approximation errors')

mpl.pyplot.subplot(323, sharex=ax1)
mpl.pyplot.plot(xs, g(xs))
mpl.pyplot.plot(xs, exp3_g)
mpl.pyplot.plot(xs, exp5_g)
mpl.pyplot.plot(xs, exp10_g)
mpl.pyplot.ylim(0, 10000)
mpl.pyplot.title(r"Exponential Function: $e^{\frac{1}{x}}$")

mpl.pyplot.subplot(324, sharex=ax1)
mpl.pyplot.plot(xs, g(xs) - exp3_g)
mpl.pyplot.plot(xs, g(xs) - exp5_g)
mpl.pyplot.plot(xs, g(xs) - exp10_g)
mpl.pyplot.ylim(0, 10000)
mpl.pyplot.title('Exponential function approximation errors')

mpl.pyplot.subplot(325, sharex=ax1)
mpl.pyplot.plot(xs, h(xs))
mpl.pyplot.plot(xs, ramp3_h)
mpl.pyplot.plot(xs, ramp5_h)
mpl.pyplot.plot(xs, ramp10_h)
mpl.pyplot.ylim(-1, 1)
mpl.pyplot.title(r"Ramp Function: $\frac{x+|x|}{2}$")

mpl.pyplot.subplot(326, sharex=ax1)
mpl.pyplot.plot(xs, h(xs) - ramp3_h)
mpl.pyplot.plot(xs, h(xs) - ramp5_h)
mpl.pyplot.plot(xs, h(xs) - ramp10_h)
mpl.pyplot.ylim(-0.2, 0.2)
mpl.pyplot.title('Ramp function approximation errors')

mpl.pyplot.xlim(x_grid)
mpl.pyplot.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.25, wspace=0.35)
mpl.pyplot.show()

# Exercise 1.3.3

x_grid=[-1,1]
N= 11
x1 = ny.cos((2*ny.arange(1,N+1)-1)/(2*N)*ny.pi) 
x1 = sorted(x1)
x1 = ny.asarray(x1)
xs = ny.linspace(x_grid[0],x_grid[1],101)


#------------------Runge function---------------------------------------------
def f(x):
	return 1.0 / (1.0 + 25.0 * ny.power(x,2))
y = f(x1)

# Chebyshev polynomials of order 3, 5, 10:
runge3_coef = cheby.chebfit(x1,y,3) 
runge3_f = cheby.chebval(xs,runge3_coef)

runge5_coef = cheby.chebfit(x1,y,5) 
runge5_f  = cheby.chebval(xs,runge5_coef)

runge10_coef = cheby.chebfit(x1,y,10) 
runge10_f  = cheby.chebval(xs,runge10_coef)

#------------------Exponential Function---------------------------------------
def g(x):
	with ny.errstate(divide='ignore', invalid='ignore'):
		return ny.exp(1/x)

yg = g(x1)
yg[yg == inf] = 9999

exp3_coef = cheby.chebfit(x1,yg,3) 
exp3_g  = cheby.chebval(xs,exp3_coef)

exp5_coef = cheby.chebfit(x1,yg,5) 
exp5_g  = cheby.chebval(xs,exp5_coef)

exp10_coef = cheby.chebfit(x1,yg,10) 
exp10_g  = cheby.chebval(xs,exp10_coef)

#------------------Ramp Function-----------------------------------------------
def h(x):
	return (x+abs(x))*0.5
yh = h(x1)

ramp3_coef = cheby.chebfit(x1,yh,3) 
ramp3_h  = cheby.chebval(xs,ramp3_coef)

ramp5_coef = cheby.chebfit(x1,yh,5) 
ramp5_h  = cheby.chebval(xs,ramp5_coef)

ramp10_coef = cheby.chebfit(x1,yh,10) 
ramp10_h  = cheby.chebval(xs,ramp10_coef)

#----------------------Plot----------------------------------------------------
fig = mpl.pyplot.figure(figsize=(15,10))
fig.suptitle('Czebyszow nodes and Czebyszow polynomials')

axis2 = mpl.pyplot.subplot(321)
mpl.pyplot.plot(xs,f(xs), label='Exact Function')
mpl.pyplot.plot(xs,runge3_f,label='Czebyszow:3') 
mpl.pyplot.plot(xs,runge5_f,label='Czebyszow:5')
mpl.pyplot.plot(xs,runge10_f,label='Czebyszow:10')
mpl.pyplot.legend(bbox_to_anchor=(0.25,0), loc="lower center", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.ylim(-0.2,1)
mpl.pyplot.title(r"Runge Function: $\frac{1}{1 + 25x^{2}}$")

mpl.pyplot.subplot(322, sharex=axis2)
mpl.pyplot.plot(xs,f(xs)-runge3_f,label='Czebyszew:3 error') 
mpl.pyplot.plot(xs,f(xs)-runge5_f,label='Czebyszow:5 error')
mpl.pyplot.plot(xs,f(xs)-runge10_f,label='Czebyszow:10 error')
mpl.pyplot.ylim(-0.25,1)
mpl.pyplot.legend(bbox_to_anchor=(0.75,0), loc="lower center", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.title('Runge Function Approximation Errors')


mpl.pyplot.subplot(323, sharex=axis2)
mpl.pyplot.plot(xs, g(xs))
mpl.pyplot.plot(xs, exp3_g)
mpl.pyplot.plot(xs, exp5_g)
mpl.pyplot.plot(xs, exp10_g)
mpl.pyplot.ylim(-100, 10000)
mpl.pyplot.title(r"Exponential Function: $e^{\frac{1}{x}}$")

mpl.pyplot.subplot(324, sharex=axis2)
mpl.pyplot.plot(xs, g(xs) - exp3_g)
mpl.pyplot.plot(xs, g(xs) - exp5_g)
mpl.pyplot.plot(xs, g(xs) - exp10_g)
mpl.pyplot.ylim(-5000, 5000)
mpl.pyplot.title('Exponential Function Approximation Errors')

mpl.pyplot.subplot(325, sharex=axis2)
mpl.pyplot.plot(xs, h(xs))
mpl.pyplot.plot(xs, ramp3_h)
mpl.pyplot.plot(xs, ramp5_h)
mpl.pyplot.plot(xs, ramp10_h)
mpl.pyplot.ylim(-0.25, 1.25)
mpl.pyplot.title(r"Ramp Function: $\frac{x+|x|}{2}$")

mpl.pyplot.subplot(326, sharex=axis2)
mpl.pyplot.plot(xs, h(xs) - ramp3_h)
mpl.pyplot.plot(xs, h(xs) - ramp5_h)
mpl.pyplot.plot(xs, h(xs) - ramp10_h)
mpl.pyplot.ylim(-0.2, 0.2)
mpl.pyplot.title('Ramp Function Approximation Errors')

mpl.pyplot.xlim(x_grid)
mpl.pyplot.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.90, hspace=0.25,wspace=0.35)
mpl.pyplot.show()

# Exercise 2
#TBA
