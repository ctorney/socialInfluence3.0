
#!/usr/bin/python

import sympy as sp
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib as mpl
from scipy import integrate
from scipy import optimize

NA = 128
Ns =180
K = 90       
ws = 0.524
alpha = 0.0

def EXTAPPROX(X,sig):  
    A = np.log(K)/(2.0) - 1/float(sig)
    B = np.log(K)
    VR = 0.5*B**2/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR*X  +  ((VR)*(alpha*  (Ns-1) - 1.0)) * X**2)
def X1APPROX(X,sig):  
    A = np.log(K)/(2.0) - 1/float(sig)
    B = np.log(K)
    return (1.0+1/float(sig))*(K**-0.5)*0.5*(1.0+B*X)
    return m.exp(-A)*0.5*(1.0+B*X)
def mfX1(sig):  
# ax^2 + bx + C = 0
    A = np.log(K)/(2.0)
    C =  m.exp(-1.0/float(sig))
    C =  1.0 -1.0/float(sig)
    B = np.log(K)
    A = np.log(K)/(2.0) - 1.0/float(sig)
#    return 1.0/(2*m.exp(A)-B)
    return 1.0/(2*K**0.5*C - np.log(K))
def mfX2(sig):  
# ax^2 + bx + C = 0
    A = np.log(K)/(2.0) - 1.0/float(sig)
    B = np.log(K)
    return (1.0-A)/(2.0-B)
def fullX(X,sig):  
# ax^2 + bx + C = 0
    A = np.log(K)/(2.0) - 1.0/float(sig)
    B = np.log(K)
    return X*(1- ((B**2*X)/(2.0*(B*X-1)))*(alpha*X + (1-X)/Ns ))
    return (1.0-A)/(2.0-B)

dx = 0.01
xs = np.arange(0,1.0,dx)


sigma = 4;
ex = [EXTAPPROX(x,sigma) for x in xs]
ex1 = [X1APPROX(x,sigma) for x in xs]
#for p in pot: print p
x2 = mfX2(sigma)
x1 = mfX1(sigma)
x2 = fullX(x2,sigma)
plt.plot(xs,ex,linestyle='-')
plt.plot(xs,(ex1),linestyle='-')
plt.plot(x2, x2,'ro')
plt.plot(x1, x1,'ro')
plt.plot(xs,xs,linestyle='-')
plt.axis([0, 0.5, 0, 0.5])
plt.show()
#atimes = (np.log(atimes))
#plt.plot(sigs, atimes)
#plt.plot(sigs, atimes2)
#plt.show()
