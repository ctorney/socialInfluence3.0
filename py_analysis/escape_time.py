
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
Ns =18
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
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
def EXTAPPROX0(X,sig):  
    A = np.log(K)/(2.0)# - 1/sig
    B = np.log(K)
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
def atup(X,sig):
 #   return (EXTAPPROX(X,sig))
    return (1-X)*(EXTAPPROX(X,sig))
def atdown(X,sig): 
    return (X)*(1.0-EXTAPPROX(X,sig))
numA = 15
sigs = np.zeros(numA)
atimes = np.zeros(numA)
atimes2 = np.zeros(numA)
alpha = 0.30
for acount in range(numA):
    
    sigma  = 2.5+0.5*float(acount)
    sigs[acount]=sigma#np.sqrt(sigma)
    
    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    N2 = int(NA*(A/B) -1)
    
    Q = np.zeros((N2,N2))

    Q[0,0]=1.0-tup(0,sigma)-tdown(0,sigma)
    Q[0,1]=tup(0,sigma)+tdown(0,sigma)

    Q[N2-1,N2-2]=tdown(N2/float(NA),sigma)
    Q[N2-1,N2-1]=1.0-tup(N2/float(NA),sigma)-tdown(N2/float(NA),sigma)
    for x in range(1,N2-1):
        Q[x,x-1] = tdown(x/float(NA),sigma)
        Q[x,x] = 1.0 - tup(x/float(NA),sigma) - tdown(x/float(NA),sigma)
        Q[x,x+1] = tup(x/float(NA),sigma)
    
 

    bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
    
    times = bb*np.matrix(np.ones((N2,1)))

 #   for x in range(0,NA):
 #       print x, EXTAPPROX(x/float(NA), sigma)
 #   print '=============='
   
    #print sigma, times[0]
    atimes[acount] = times[0]
    Q = np.zeros((N2,N2))

    Q[0,0]=1.0-atup(0,sigma)-atdown(0,sigma)
    Q[0,1]=atup(0,sigma)+atdown(0,sigma)

    Q[N2-1,N2-2]=atdown(N2/float(NA),sigma)
    Q[N2-1,N2-1]=1.0-atup(N2/float(NA),sigma)-atdown(N2/float(NA),sigma)
    for x in range(1,N2-1):
        Q[x,x-1] = atdown(x/float(NA),sigma)
        Q[x,x] = 1.0 - atup(x/float(NA),sigma) - atdown(x/float(NA),sigma)
        Q[x,x+1] = atup(x/float(NA),sigma)
    
 

    bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
    
    times = bb*np.matrix(np.ones((N2,1)))

   
    #print sigma, times[0]
    atimes2[acount] = times[0]

for i in range(numA):
    print sigs[i],atimes[i],atimes2[i]
#atimes = (np.log(atimes))
#plt.plot(sigs, atimes)
#plt.plot(sigs, atimes2)
#plt.show()
