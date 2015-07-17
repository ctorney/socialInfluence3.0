
#!/usr/bin/python

import sympy as sp
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib as mpl
from scipy import integrate
from scipy import optimize

NA = 64
Ns = 20
K = 50       
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
def EXTAPPROX(X,sig,a):  
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR +  ((VR/(1.0+VR))*a * B * X * (Ns-1)/float(Ns)))
def EXTAPPROX0(X,sig,a):  
    A = np.log(K)/(2.0)# - 1/sig
    B = np.log(K)
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR +  ((VR/(1.0+VR))*a * B * X * (Ns-1)/float(Ns)))
def atup(X,sig,a):
 #   return (EXTAPPROX(X,sig))
    return (1-X)*(EXTAPPROX(X,sig, a))
def atdown(X,sig,a): 
    return (X)*(1.0-EXTAPPROX(X,sig,a))

    
sigma  = 14.0

alpha = 0.00
A = np.log(K)/(2.0) - 1/sigma
B = np.log(K)

#for x in range(0,NA):
#    print x/float(NA), atup(x/float(NA),sigma), atdown(x/float(NA),sigma)
##
xGrid=np.arange(0,1,0.01)
gridUp = [atup(x+0.01,sigma,alpha) for x in xGrid]
gridDown = [atdown(x+0.01,sigma,alpha) for x in xGrid]

pot=np.log(np.divide(gridDown,gridUp))
pot=np.cumsum(pot)
plt.plot(xGrid,pot)
alpha = 0.35
xGrid=np.arange(0,1,0.01)
gridUp = [atup(x+0.01,sigma,alpha) for x in xGrid]
gridDown = [atdown(x+0.01,sigma,alpha) for x in xGrid]
pot=np.log(np.divide(gridDown,gridUp))
pot=np.cumsum(pot)
plt.plot(xGrid,pot)
plt.show()
   
#atimes = (np.log(atimes))
#plt.plot(sigs, atimes)
#plt.plot(sigs, atimes2)
#plt.show()
