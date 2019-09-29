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

mpl.pyplot.style.use("ggplot")

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

# Taylor approximation at x0 of the function
def taylor(function,x0,n):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x,i).subs(x,x0))/(factorial(i))*(x-x0)**i
        i += 1
    return p

# Plot results
def plot():
    x_lims = [0,4]
    x1 = ny.linspace(x_lims[0],x_lims[1],500)
    y1 = [] 
    order = [1,2,5,20]
    for j in order:
        func = taylor(f,1,j)
        for k in x1:
            y1.append(func.subs(x,k))
        mpl.pyplot.plot(x1,y1,label='Order '+str(j))
        y1 = []
    mpl.pyplot.plot(x1,x1**(0.321),label=r'$x^{0.321}$')
    mpl.pyplot.xlim(x_lims)
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
    x_lims = [-2,6]
    x1 = ny.linspace(x_lims[0],x_lims[1],500)
    y1 = []
    order = [1,2,5,20]
    for j in order:
        func = taylor(f,2,j)
        for k in x1:
            y1.append(func.subs(x,k))
        mpl.pyplot.plot(x1,y1,label='Order '+str(j))
        y1 = []
    mpl.pyplot.plot(x1, (x1 + abs(x1))*0.5,label=r'$\frac{x+\left|x\right|}{2}$')
    mpl.pyplot.xlim(x_lims)
    mpl.pyplot.ylim([-2,6])
    mpl.pyplot.xlabel('x')
    mpl.pyplot.ylabel('y')
    mpl.pyplot.legend()
    mpl.pyplot.title('Taylor approximation of the ramp function')
    mpl.pyplot.show()
plot2()


# Exercise 1.3.1

# Runge's function in [-1,1]
def f(x):
	return 1.0 / (1.0 + 25.0 * x**2)

# Evenly spaced interpolation nodes:
x_lims=[-1,1]
x1 = ny.linspace(x_lims[0],x_lims[1],11) 
y = f(x1)
xs = ny.linspace(x_lims[0],x_lims[1],101)
cs = poly.polyfit(x1,y,3)
ffit3 = poly.polyval(xs,cs)

# Monomials of order 5 and 10:
coeffs = poly.polyfit(x1,y,5)
ffit = poly.polyval(xs,coeffs) 
coeffs10 = poly.polyfit(x1,y,10)
ffit10 = poly.polyval(xs,coeffs10)

# Exponential Function
def g(x):
	with ny.errstate(divide='ignore', invalid='ignore'):
		return ny.exp(1/x)

yg = g(x1)
yg[yg == inf] = 9999


csg = poly.polyfit(x1,yg,3)
coeffg = poly.polyfit(x1,yg,5)

ffitg = poly.polyval(xs,coeffg)
ffit3g = poly.polyval(xs,csg)

coeffs10g = poly.polyfit(x1,yg,10)
ffit10g = poly.polyval(xs,coeffs10g)

def h(x):
	return .5*(x+abs(x))
yh = h(x1)

csh = poly.polyfit(x1,yh,3)
ffit3h = poly.polyval(xs,csh)

# Monomials of order 5 and 10:
coeffsh = poly.polyfit(x1,yh,5)
ffith = poly.polyval(xs,coeffsh)
coeffs10h = poly.polyfit(x1,yh,10)
ffit10h = poly.polyval(xs,coeffs10h)

# Plot
fig = mpl.pyplot.figure(figsize=(10,10))
fig.suptitle('Evenly-Spaced nodes & monomials')

axis1 = mpl.pyplot.subplot(321)
#mpl.pyplot.plot(x1,y,'o',label='Points')
mpl.pyplot.plot(xs,f(xs), label='True function')
mpl.pyplot.plot(xs,ffit3,label='Cubic Polynomials') 
mpl.pyplot.plot(xs,ffit,label='Monomial of order 5')
mpl.pyplot.plot(xs,ffit10,label='Monomial of order 10')
mpl.pyplot.legend(bbox_to_anchor=(0.5,0), loc="best", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.ylim(-2,2)
mpl.pyplot.title(r"Runge Function: $\frac{1}{1 + 25x^{2}}$")

mpl.pyplot.subplot(322, sharex=axis1)
mpl.pyplot.plot(xs,f(xs)-ffit3,label='Cubic error') 
mpl.pyplot.plot(xs,f(xs)-ffit,label='Order 5 error')
mpl.pyplot.plot(xs,f(xs)-ffit10,label='Order 10 error')
mpl.pyplot.ylim(-0.6,0.6)
mpl.pyplot.legend(bbox_to_anchor=(0.9,0), loc="best", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.title('Runge Function Approximation Errors')

mpl.pyplot.subplot(323, sharex=axis1)
mpl.pyplot.plot(x1,yg,'o')
mpl.pyplot.plot(xs,g(xs))
mpl.pyplot.plot(xs,ffit3g) 
mpl.pyplot.plot(xs,ffitg)
mpl.pyplot.plot(xs,ffit10g)
mpl.pyplot.ylim(-100,5000)
mpl.pyplot.title(r"Exponential Function: $e^{\frac{1}{x}}$")

mpl.pyplot.subplot(324, sharex=axis1)
mpl.pyplot.plot(xs,g(xs)-ffit3h)
mpl.pyplot.plot(xs,g(xs)-ffitg)
mpl.pyplot.plot(xs,g(xs)-ffit10g)
mpl.pyplot.ylim(-1000,1000)
mpl.pyplot.title('Exponential Function Approximation Errors')

mpl.pyplot.subplot(325, sharex=axis1)
mpl.pyplot.plot(x1,yh,'o')
mpl.pyplot.plot(xs,h(xs))
mpl.pyplot.plot(xs,ffit3h) 
mpl.pyplot.plot(xs,ffith)
mpl.pyplot.plot(xs,ffit10h)
mpl.pyplot.ylim(-0.5,2.5)
mpl.pyplot.title(r"Ramp Function: $\frac{x+|x|}{2}$")

mpl.pyplot.subplot(326, sharex=axis1)
mpl.pyplot.plot(xs,h(xs)-ffit3h) 
mpl.pyplot.plot(xs,h(xs)-ffith)
mpl.pyplot.plot(xs,h(xs)-ffit10h)
mpl.pyplot.ylim(-1,1)
mpl.pyplot.title('Ramp Function Approximation Errors')

mpl.pyplot.xlim(x_lims)
mpl.pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
mpl.pyplot.show()

# Exercise 1.3.2
def f(x):
	return 1.0 / (1.0 + 25.0 * x**2)
x_lims=[-1,1]
x1 = ny.linspace(x_lims[0],x_lims[1],11) 
y = f(x1)

xs = ny.linspace(x_lims[0],x_lims[1],101)
cs = poly.polyfit(x1,y,3)
ffit3 = poly.polyval(xs,cs)
N = 11
x1 = ny.cos((2 * ny.arange(1, N + 1) - 1) / (2 * N) * ny.pi)
x1 = sorted(x1)
x1 = ny.asarray(x1)
# Runge's function in [-1,1]
def f(x):
	return 1.0 / (1.0 + 25.0 * ny.power(x, 2))
y = f(x1)

cs = poly.polyfit(x1, y,3)
ffit3 = poly.polyval(xs,cs)
xs = ny.linspace(x_lims[0], x_lims[1], 101)

# Monomials of order 5and 10:
coeffs = poly.polyfit(x1, y, 5)
ffit = poly.polyval(xs, coeffs)
coeffs10 = poly.polyfit(x1, y, 10)
ffit10 = poly.polyval(xs, coeffs10)

# Exponential Function
def g(x):
	with ny.errstate(divide='ignore', invalid='ignore'):
		return ny.exp(1 / x)

yg = g(x1)
yg[yg == inf] = 9999

csg = poly.polyfit(x1, yg,3)
ffit3g = poly.polyval(xs,csg)
coeffg = poly.polyfit(x1, yg, 5)
ffitg = poly.polyval(xs, coeffg) 
coeffs10g = poly.polyfit(x1, yg, 10)
ffit10g = poly.polyval(xs, coeffs10g)


def h(x):
	return .5 * (x + abs(x))
yh = h(x1)

csh = poly.polyfit(x1, yh,3)
ffit3h = poly.polyval(xs,csh)

# Monomials of order 5 and 10:
coeffsh = poly.polyfit(x1, yh, 5)
ffith = poly.polyval(xs, coeffsh)
coeffs10h = poly.polyfit(x1, yh, 10)
ffit10h = poly.polyval(xs, coeffs10h)

# Plot
fig = mpl.pyplot.figure(figsize=(12, 12))
fig.suptitle('Czebyszow nodes & monomials')

ax1 = mpl.pyplot.subplot(321)
mpl.pyplot.plot(xs, f(xs), label='Exact Function')
mpl.pyplot.plot(xs, ffit3, label='Cubic Polynomials')
mpl.pyplot.plot(xs, ffit, label='Monomial of order 5')
mpl.pyplot.plot(xs, ffit10, label='Monomial of order 10')
mpl.pyplot.legend(bbox_to_anchor=(0.45, 0), loc="best", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.ylim(-0.2, 1)
mpl.pyplot.title(r"Runge Function: $\frac{1}{1 + 25x^{2}}$")

mpl.pyplot.subplot(322, sharex=ax1)
mpl.pyplot.plot(xs, f(xs) - ffit3, label='Cubic error')
mpl.pyplot.plot(xs, f(xs) - ffit, label='Monomial 5 error')
mpl.pyplot.plot(xs, f(xs) - ffit10, label='Monomial 10 error')
mpl.pyplot.ylim(-1, 1)
mpl.pyplot.legend(bbox_to_anchor=(0.95, 0), loc="best", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.title('Runge function approximation errors')

mpl.pyplot.subplot(323, sharex=ax1)
mpl.pyplot.plot(x1, yg, 'o')
mpl.pyplot.plot(xs, g(xs))
mpl.pyplot.plot(xs, ffit3g)
mpl.pyplot.plot(xs, ffitg)
mpl.pyplot.plot(xs, ffit10g)
mpl.pyplot.ylim(-100, 10000)
mpl.pyplot.title(r"Exponential Function: $e^{\frac{1}{x}}$")

mpl.pyplot.subplot(324, sharex=ax1)
mpl.pyplot.plot(xs, g(xs) - ffit3h)
mpl.pyplot.plot(xs, g(xs) - ffitg)
mpl.pyplot.plot(xs, g(xs) - ffit10g)
mpl.pyplot.ylim(0, 10000)
mpl.pyplot.title('Exponential function approximation errors')

mpl.pyplot.subplot(325, sharex=ax1)
mpl.pyplot.plot(x1, yh, 'o')
mpl.pyplot.plot(xs, h(xs))
mpl.pyplot.plot(xs, ffit3h)
mpl.pyplot.plot(xs, ffith)
mpl.pyplot.plot(xs, ffit10h)
mpl.pyplot.ylim(-2.5, 2.5)
mpl.pyplot.title(r"Ramp Function: $\frac{x+|x|}{2}$")

mpl.pyplot.subplot(326, sharex=ax1)
mpl.pyplot.plot(xs, h(xs) - ffit3h)
mpl.pyplot.plot(xs, h(xs) - ffith)
mpl.pyplot.plot(xs, h(xs) - ffit10h)
mpl.pyplot.ylim(-0.25, 0.25)
mpl.pyplot.title('Ramp function approximation errors')

mpl.pyplot.xlim(x_lims)
mpl.pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
mpl.pyplot.show()

# Exercise 1.3.3

x_lims=[-1,1]
N= 11
x1 = ny.cos((2*ny.arange(1,N+1)-1)/(2*N)*ny.pi) 
x1 = sorted(x1)
x1 = ny.asarray(x1)
xs = ny.linspace(x_lims[0],x_lims[1],101)


# Runge's function in [-1,1]
def f(x):
	return 1.0 / (1.0 + 25.0 * ny.power(x,2))
y = f(x1)

# Chebyshev polynomials of order 3, 5, 10:
coeff3 = cheby.chebfit(x1,y,3) 
ffit3  = cheby.chebval(xs,coeff3)

coeff5 = cheby.chebfit(x1,y,5) 
ffit5  = cheby.chebval(xs,coeff5)

coeff10 = cheby.chebfit(x1,y,10) 
ffit10  = cheby.chebval(xs,coeff10)

# Exponential Function
def g(x):
	with ny.errstate(divide='ignore', invalid='ignore'):
		return ny.exp(1/x)

yg = g(x1)
yg[yg == inf] = 9999

coeffg3 = cheby.chebfit(x1,yg,3) 
ffitg3  = cheby.chebval(xs,coeffg3)

coeffg5 = cheby.chebfit(x1,yg,5) 
ffitg5  = cheby.chebval(xs,coeffg5)

coeffg10 = cheby.chebfit(x1,yg,10) 
ffitg10  = cheby.chebval(xs,coeffg10)

def h(x):
	return .5*(x+abs(x))
yh = h(x1)

coeffh3 = cheby.chebfit(x1,yh,3) 
ffith3  = cheby.chebval(xs,coeffh3)

coeffh5 = cheby.chebfit(x1,yh,5) 
ffith5  = cheby.chebval(xs,coeffh5)

coeffh10 = cheby.chebfit(x1,yh,10) 
ffith10  = cheby.chebval(xs,coeffh10)

# Plot
fig = mpl.pyplot.figure(figsize=(12,12))
fig.suptitle('Czebyszow nodes and Czebyszow polynomials')

axis2 = mpl.pyplot.subplot(321)
mpl.pyplot.plot(xs,f(xs), label='Exact Function')
mpl.pyplot.plot(xs,ffit3,label='Czebyszow:3') 
mpl.pyplot.plot(xs,ffit5,label='Czebyszow:5')
mpl.pyplot.plot(xs,ffit10,label='Czebyszow:10')
mpl.pyplot.legend(bbox_to_anchor=(0.4,0), loc="best", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.ylim(-0.2,1)
mpl.pyplot.title(r"Runge Function: $\frac{1}{1 + 25x^{2}}$")

mpl.pyplot.subplot(322, sharex=axis2)
mpl.pyplot.plot(xs,f(xs)-ffit3,label='Czebyszew:3 error') 
mpl.pyplot.plot(xs,f(xs)-ffit5,label='Czebyszow:5 error')
mpl.pyplot.plot(xs,f(xs)-ffit10,label='Czebyszow:10 error')
mpl.pyplot.ylim(-0.25,1)
mpl.pyplot.legend(bbox_to_anchor=(0.95,0), loc="best", bbox_transform=fig.transFigure, ncol=2)
mpl.pyplot.title('Runge Function Approximation Errors')


mpl.pyplot.subplot(323, sharex=axis2)
mpl.pyplot.plot(x1, yg,'o')
mpl.pyplot.plot(xs, g(xs))
mpl.pyplot.plot(xs, ffitg3)
mpl.pyplot.plot(xs, ffitg5)
mpl.pyplot.plot(xs, ffitg10)
mpl.pyplot.ylim(-100, 10000)
mpl.pyplot.title(r"Exponential Function: $e^{\frac{1}{x}}$")

mpl.pyplot.subplot(324, sharex=axis2)
mpl.pyplot.plot(xs, g(xs) - ffitg3)
mpl.pyplot.plot(xs, g(xs) - ffitg5)
mpl.pyplot.plot(xs, g(xs) - ffitg10)
mpl.pyplot.ylim(-500, 10000)
mpl.pyplot.title('Exponential Function Approximation Errors')

mpl.pyplot.subplot(325, sharex=axis2)
mpl.pyplot.plot(x1, yh,'o')
mpl.pyplot.plot(xs, h(xs))
mpl.pyplot.plot(xs, ffith3)
mpl.pyplot.plot(xs, ffith5)
mpl.pyplot.plot(xs, ffith10)
mpl.pyplot.ylim(-0.25, 1.25)
mpl.pyplot.title(r"Ramp Function: $\frac{x+|x|}{2}$")

mpl.pyplot.subplot(326, sharex=axis2)
mpl.pyplot.plot(xs, h(xs) - ffith3)
mpl.pyplot.plot(xs, h(xs) - ffith5)
mpl.pyplot.plot(xs, h(xs) - ffith10)
mpl.pyplot.ylim(-0.2, 0.2)
mpl.pyplot.title('Ramp Function Approximation Errors')

mpl.pyplot.xlim(x_lims)
mpl.pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
mpl.pyplot.show()

# Exercise 2
TBA
