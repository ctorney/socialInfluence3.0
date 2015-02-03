
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
Ns = 8
K = 90       
ws = 0.514

def tswup( xx ):
    gc = np.log(K)*sigma*(1.0-2.0*xx)/(2.0)
    if gc>1:
        return 0.5*m.exp(-(gc-1.0)/sigma)
    else:
        return 1.0 - 0.5*m.exp((gc-1.0)/sigma)
def tswup1( xx, sig ):
    gc = np.log(K)*sig*(1.0-2.0*xx)/(2.0)
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    if gc>1:
     #   return 0.5*m.exp(-(gc-1.0)/sig)
        return 0.5*m.exp(-(A-B*xx))
    else:
        return 1.0 - 0.5*m.exp((A-B*xx))#0.5*m.exp((gc-1.0)/sig)

def detODE(X):
    return np.array([sp.N(ETSWUPZAPROX(X)-X)],dtype=float)
def ETSWUPZALPHA(X):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup(j/float(Ns)) for j in xrange(0,Ns+1))
def ETSWUPZALPHA2(X,sig):  
    return sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* tswup1(j/float(Ns),sig) for j in xrange(0,Ns+1))
def ETSWUPZAPROX(X):  
    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
def X1():  
    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    return 0.5*( m.exp(-(A))  ) / ( 1.0 -  0.5*( m.exp(-(A))  ) *(B +  0.5*B**2/float(Ns)) ) 
def X2(X):  
#rh = RHO(X)
    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    VR = B**2*X*(1-X)/float(Ns)
    rh =  alpha * ( (B)* X  )/ (float(Ns) + 0.5*B**2 * X * (1-X) )
    KK =   0.5*B**2 * X * (1-X) / float(Ns)
    sigma_2 = rh + (1.0-rh)/float(Ns)
    return 0.5*m.exp(-(A-B*X))*(1.0 +  VR + ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
    return 0.5*m.exp(-(A-B*X))*(1.0 +  (0.5*B**2 * X * (1-X) / float(Ns))*(1.0 + ((alpha*B*X/(1+ (0.5*B**2 * X * (1-X) / float(Ns)))) * (1.0- (1.0/float(Ns))))))

    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    rh =  alpha * ( (B)* X  )/ (float(Ns) + 0.5*B**2 * X * (1-X) )#sum(sp.binomial(Ns,j) * X**j* (1-X)**(Ns-j)* (j/float(Ns)-X)**2 for j in xrange(0,Ns+1))
    sigma_2 = rh + (1.0-rh)/float(Ns)
    return 0.5*(1+(B*X-A) )*(1.0 + 0.5*B**2* X * (1-X) * sigma_2)
def diffdown(x):
    h=0.0001
    return (tdown2(x+0.5*h)-tdown2(x-0.5*h))/h

def diffup(x):
    h=0.0001
    return (m.exp(-0)*tup2(x+0.5*h)-m.exp(-0)*tup2(x-0.5*h))/h
    
def diff2phi(x):
    return -((1.0/(m.exp(-0)*tup2(x)))*diffup(x)-(1.0/tdown2(x))*diffdown(x))
   
def tup(X):
    return (1-X)*(ETSWUPZALPHA(X))
def tdown(X): 
    return (X)*(1.0-ETSWUPZALPHA(X))
def tup2(X):
    return ETSWUPZAPROX(X)  
def tdown2(X): 
    return (X)
    
def phi(x): 
    N21 = 0.5/int(NA)
    #intexp = lambda y: -m.log(m.exp(-A)*tup2(y)/tdown2(y))
    intexp = lambda y: -m.log(tup2(y)/tdown2(y))
    if x<N21:
        return 0
    ph, err = integrate.quad(intexp,0,x)
    return ph
def timeSigma(s):
    A = np.log(K)/(2.0) - 1/s
    B = np.log(K)
    N2 = int(NA*(A/B) -1)
#    print "====================="
#    print alpha
#    print "====================="
#    for x in range(0,NA):
#        print ETSWUP2(x/float(NA)), X2(x/float(NA))
    
    Q = np.zeros((N2,N2))

    Q[0,0]=1.0-tup3(0,s)-tdown3(0,s)
    Q[0,1]=tup3(0,s)+tdown3(0,s)

    Q[N2-1,N2-2]=tdown3(N2/float(NA),s)
    Q[N2-1,N2-1]=1.0-tup3(N2/float(NA),s)-tdown3(N2/float(NA),s)
    for x in range(1,N2-1):
        Q[x,x-1] = tdown3(x/float(NA),s)
        Q[x,x] = 1.0 - tup3(x/float(NA),s) - tdown3(x/float(NA),s)
        Q[x,x+1] = tup3(x/float(NA),s)
    
 

    bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
    
    times = bb*np.matrix(np.ones((N2,1)))

   
    return np.log(times[0])
def dSigma(sig):
    h=0.0001
    return (timeSigma(sig+0.5*h)-timeSigma(sig-0.5*h))/h
numA = 15
sigs = np.zeros(numA)
dsigs = np.zeros(numA)
atimes = np.zeros(numA)
atimes2 = np.zeros(numA)


alpha = 0.0# + acount/float(numA-1)
for acount in range(numA):
    sigma  = 1/(0.1 + 0.01 * float(acount))
    sigs[acount]=sigma
    dsigs[acount]=dSigma(sigma)
    
    A = np.log(K)/(2.0) - 1/sigma
    B = np.log(K)
    N2 = int(NA*(A/B) -1)
#    print "====================="
#    print alpha
#    print "====================="
#    for x in range(0,NA):
#        print ETSWUP2(x/float(NA)), X2(x/float(NA))
    
    Q = np.zeros((N2,N2))

    Q[0,0]=1.0-tup(0)-tdown(0)
    Q[0,1]=tup(0)+tdown(0)

    Q[N2-1,N2-2]=tdown(N2/float(NA))
    Q[N2-1,N2-1]=1.0-tup(N2/float(NA))-tdown(N2/float(NA))
    for x in range(1,N2-1):
        Q[x,x-1] = tdown(x/float(NA))
        Q[x,x] = 1.0 - tup(x/float(NA)) - tdown(x/float(NA))
        Q[x,x+1] = tup(x/float(NA))
    
 

    bb = np.matrix(np.linalg.inv(np.identity(N2) - Q))
    
    times = bb*np.matrix(np.ones((N2,1)))

   
    #print sigma, times[0]
    atimes[acount] = times[0]

    x1=X1()
    shiftx=0.25
    
    print x1
    x1=optimize.fsolve(detODE,0)[0]
    print x1
    x3=optimize.fsolve(detODE,x1+shiftx)[0]
    dx12 =diff2phi(x1)
    dx32 =diff2phi(x3)
    atimes2[acount] = NA*((tup2(x1))**-1*(abs(dx12*dx32))**-0.5)*2.0*3.142*m.exp(A + NA*(A*(x3-x1)) + (NA*( phi(x3)-phi(x1))))
    atimes2[acount] = (sigma**-2)*(1.0 + NA*((x3-x1)))# + (NA*( phi(x3)-phi(x1))))
    atimes2[acount] = dx12#NA*((m.exp(-0))**-1*(abs(dx12*dx32))**-0.5)
#print ETSWUP2(0.1)

#aa= np.arange(65)/float(2*NA)
#for a in range(np.size(atimes)): print alphas[a], atimes[a]
#plt.plot(aa, [RHO(i) for i in aa])
#atimes = (np.log(atimes))
atimes = (np.log(atimes))
#atimes2 = (np.log(atimes2))
plt.plot(sigs, atimes)
plt.plot(sigs, atimes2)
#plt.yscale('log')
#plt.show()
for i in atimes2: print i
