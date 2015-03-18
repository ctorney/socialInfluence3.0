
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
Ns =800
K = 90       
ws = 0.524

def tswup( xx, sig ):
    gc = np.log(K)*sig*(1.0-2.0*xx)/(2.0)
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    if gc>1:

        return 0.5*m.exp(-(A-B*xx))
    else:
        return 1.0 - 0.5*m.exp((A-B*xx))

def EXTSWUP(X,sig):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns),sig) for j in xrange(0,Ns+1))
def tup(X,sig):
    return (1-X)*(EXTSWUP(X,sig))
def tdown(X,sig): 
    return (X)*(1.0-EXTSWUP(X,sig))
def EXTAPPROX(X,sig):  
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    VR = 0.5*B**2/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR*X  +  ((VR)*(alpha*  (Ns-1) - 1.0)) * X**2)
def EXTAPPROX1(X,sig):  
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    VR = 0.5*B**2/float(Ns)
    Y =  (A-m.log(B/2))/B
 #   Y=A/B
    return m.exp(-A)*0.5*m.exp((B*Y))*(1.0 +  B*(X-Y)  +  0.5*B**2 * (X-Y)**2)*(1.0 +  VR*X  +  ((VR)*(alpha*  (Ns-1) - 1.0)) * X**2)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR*X  +  ((VR)*(alpha*  (Ns-1) - 1.0)) * X**2)
numA = 15
sigs = np.zeros(numA)
atimes = np.zeros(numA)
atimes2 = np.zeros(numA)
alpha = 0.0

dx = 0.01
xs = np.arange(0,0.6,dx)


sigma = 4;
ex = [EXTAPPROX(x,sigma) for x in xs]
ex1 = [EXTAPPROX1(x,sigma) for x in xs]
#for p in pot: print p
plt.plot(xs,ex,linestyle='-')
plt.plot(xs,ex1,linestyle='-')
plt.plot(xs,xs,linestyle='-')
plt.axis([0, 0.6, 0, 0.60])
plt.show()
#atimes = (np.log(atimes))
#plt.plot(sigs, atimes)
#plt.plot(sigs, atimes2)
#plt.show()
