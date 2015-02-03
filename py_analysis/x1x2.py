
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
Ns =80
K = 900       
ws = 0.524

def detODE(X):
    return np.array([sp.N(EXTAPPROX(X,sigma)-X)],dtype=float)
def EXTAPPROX(X,sig):  
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))#*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
def EXTAPPROX2(X,sig):  
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))#*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
def EXTAPPROX3(X,sig):  
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return 0.5*(1 + (B*X-A) +  0.0*(B*X-A)**2)# + 0.5/3*(B*X-A)**3 + 1/24.0*(B*X-A)**4)#*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
    return 0.5*m.exp(1-A)*(1 + (B*X-1) +  0.5*(B*X-1)**2 + 0.5/3*(B*X-1)**3 + 1/24.0*(B*X-1)**4)#*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
def EXTAPPROX4(X,sig):  
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return 0.5*m.exp(1-A)*(1 + (B*X-1) +  0.5*(B*X-1)**2)# + 0.5/3*(B*X-1)**3 + 1/24.0*(B*X-1)**4)#*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
    return (1-X)*m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
    return 0.5*(1.0+ ((B*X)-A) + 0.5*((B*X)-A)**2)*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
def X1APPROX(sig):  
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    return (1)/(2*m.exp(A)-B)
def X2APPROX(sig):  
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    return (1-A)/(2-B)
    X = m.exp(-A)*0.5;
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))

numA = 25
sigs = np.zeros(numA)
atimes = np.zeros(numA)
atimes2 = np.zeros(numA)
alpha = 0
for acount in range(numA):
    sigma  = 0.5+0.5*float(acount)
    sigs[acount]=sigma#np.sqrt(sigma)
    
    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    N2 = int(NA*(A/B) -1)
    

    xs = [x for x in np.arange(0.01,0.5,0.01)]
  #  ets = [EXTAPPROX(x,sigma)-x for x in xs]
  #  ets2 = [EXTAPPROX2(x,sigma) for x in xs]
  #  ets3 = [EXTAPPROX3(x,sigma) for x in xs]
  #  ets4 = [EXTAPPROX4(x,sigma) for x in xs]
    shiftx=0.5
    x1=optimize.fsolve(detODE,0)[0]
    x3=optimize.fsolve(detODE,x1+shiftx)[0]
    print sigma, x1, X1APPROX(sigma),x3, X2APPROX(sigma)
#atimes = (np.log(atimes))
    #plt.plot(xs, np.log( ets))
 #   plt.plot(xs, ets2)
 #   plt.plot(xs, ets3)
 #   plt.plot(xs, ets4)
 #   plt.plot(xs, xs)#np.zeros_like(xs))
    #plt.xscale('log')
#plt.plot(sigs, atimes2)
    plt.show()
