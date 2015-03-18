
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
def EXTAPPROX2(X,sig):  
    return 2*(2*X-1.0)/(2*X+1.0)
def EXTAPPROX1(X,sig):  
    A = np.log(K)/(2.0) - 1/float(sig)
    B = np.log(K)
    VR = 0.5*B**2/float(Ns)
    C = 0.5*B**2/float(Ns)
    D = C*  (alpha*  (Ns-1) - 1.0)

    Y =  (A-m.log(B/2))/B
#   Y= 0.1#A/B
#  return m.exp(-A)*0.5*m.exp((B*Y))*(((1.0  - B*Y  +  0.5*B**2 * ( Y**2))  ) + X*(( B  +  0.5*B**2 * (-2*Y )) + (C   - C*B*Y  +  0.5*B**2 *C*( Y**2)) ) + X**2*((  0.5*B**2 ) + ( C*B  +  0.5*B**2 *C*(-2*Y ))  + (1.0   - B*Y  +  0.5*B**2 * (Y**2))*( D )))
#   return m.exp(-A)*0.5*m.exp((B*Y))*((1.0 +  B*X - B*Y  +  0.5*B**2 * (X**2-2*X*Y + Y**2)) + (C*X+  C*B*X**2 - C*B*Y*X  +  0.5*B**2 *C*X*(X**2-2*X*Y + Y**2))  + (1.0   - B*Y  +  0.5*B**2 * (Y**2))*( D * X**2))
    return -A + ((B*X))#*(1.0 +  B*(X-Y)  +  0.5*B**2 * (X-Y)**2)*(1.0 +  VR*X  +  ((VR)*(alpha*  (Ns-1) - 1.0)) * X**2)
    return m.exp(-A)*0.5*m.exp((B*X))#*(1.0 +  B*(X-Y)  +  0.5*B**2 * (X-Y)**2)*(1.0 +  VR*X  +  ((VR)*(alpha*  (Ns-1) - 1.0)) * X**2)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR*X  +  ((VR)*(alpha*  (Ns-1) - 1.0)) * X**2)
def SOLVEQUAD1(sig):  
# ax^2 + bx + C = 0
    A = np.log(K)/(2.0) - 1.0/float(sig)
    B = np.log(K)
    return 1.0/(2*m.exp(A) - B)
    VR = 0.5*B**2/float(Ns)
    C = 0.5*B**2/float(Ns)
    D = C*  (alpha*  (Ns-1) - 1.0)

    AB =   (1.0/(2.0*float(sig)*np.log(K)))#0.5 - 1.0/(sig*np.log(K))
    AB1 =   - (1.0/(float(sig)*np.log(K)))#0.5 - 1.0/(sig*np.log(K))
    Y =  (A-m.log(B/2))/B
    qc = 1.0+  (1.0/(2.0*float(sig))) - 0.25*np.log(K)#1/B-0.5*(A/B)#((1.0  - B*Y  +  0.5*B**2 * ( Y**2))  )
    qb =  (1.0/float(sig) - 2.0)#0.5 - AB - 2/B#(( B  +  0.5*B**2 * (-2*Y )) + (C   - C*B*Y  +  0.5*B**2 *C*( Y**2)) )-m.exp(A)*2.0*m.exp((-B*Y))
    qa = np.log(K)#2*B#((  0.5*B**2 ) + ( C*B  +  0.5*B**2 *C*(-2*Y ))  + (1.0   - B*Y  +  0.5*B**2 * (Y**2))*( D ))
    return (-qb - m.sqrt(qb**2-4*qa*qc))/(2*qa)
def SOLVEQUAD(sig):  
# ax^2 + bx + C = 0
    A = np.log(K)/(2.0) - 1.0/float(sig)
    B = np.log(K)
    return (1.0-A)/(2.0-B)
    VR = 0.5*B**2/float(Ns)
    C = 0.5*B**2/float(Ns)
    D = C*  (alpha*  (Ns-1) - 1.0)

    AB =   (1.0/(2.0*float(sig)*np.log(K)))#0.5 - 1.0/(sig*np.log(K))
    AB1 =   - (1.0/(float(sig)*np.log(K)))#0.5 - 1.0/(sig*np.log(K))
    Y =  (A-m.log(B/2))/B
    qc = 1.0+  (1.0/(2.0*float(sig))) - 0.25*np.log(K)#1/B-0.5*(A/B)#((1.0  - B*Y  +  0.5*B**2 * ( Y**2))  )
    qb =  (1.0/float(sig) - 2.0)#0.5 - AB - 2/B#(( B  +  0.5*B**2 * (-2*Y )) + (C   - C*B*Y  +  0.5*B**2 *C*( Y**2)) )-m.exp(A)*2.0*m.exp((-B*Y))
    qa = np.log(K)#2*B#((  0.5*B**2 ) + ( C*B  +  0.5*B**2 *C*(-2*Y ))  + (1.0   - B*Y  +  0.5*B**2 * (Y**2))*( D ))
    return (-qb - m.sqrt(qb**2-4*qa*qc))/(2*qa)
 #   Y=A/B
    return m.exp(-A)*0.5*m.exp((B*Y))*(((1.0  - B*Y  +  0.5*B**2 * ( Y**2))  ) + X*(( B  +  0.5*B**2 * (-2*Y )) + (C   - C*B*Y  +  0.5*B**2 *C*( Y**2)) ) + X**2*((  0.5*B**2 ) + ( C*B  +  0.5*B**2 *C*(-2*Y ))  + (1.0   - B*Y  +  0.5*B**2 * (Y**2))*( D )))
    return m.exp(-A)*0.5*m.exp((B*Y))*((1.0 +  B*X - B*Y  +  0.5*B**2 * (X**2-2*X*Y + Y**2)) + (C*X+  C*B*X**2 - C*B*Y*X  +  0.5*B**2 *C*X*(-2*X*Y + Y**2))  + (1.0   - B*Y  +  0.5*B**2 * (Y**2))*( D * X**2))
    return m.exp(-A)*0.5*m.exp((B*Y))*(1.0 +  B*(X-Y)  +  0.5*B**2 * (X-Y)**2)*(1.0 +  VR*X  +  ((VR)*(alpha*  (Ns-1) - 1.0)) * X**2)
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR*X  +  ((VR)*(alpha*  (Ns-1) - 1.0)) * X**2)
numA = 15
sigs = np.zeros(numA)
atimes = np.zeros(numA)
atimes2 = np.zeros(numA)
alpha = 0.0

dx = 0.01
xs = np.arange(0,1.0,dx)


sigma = 4;
ex = [EXTAPPROX(x,sigma) for x in xs]
ex1 = [EXTAPPROX1(x,sigma) for x in xs]
ex2 = [EXTAPPROX2(x,sigma) for x in xs]
#for p in pot: print p
x2 = SOLVEQUAD1(sigma)
#plt.plot(xs,ex,linestyle='-')
plt.plot(xs,(ex1),linestyle='-')
plt.plot(xs,(ex2),linestyle='-')
plt.plot(x2, np.log(2*x2),'ro')
plt.plot(xs,np.log(2.0*xs),linestyle='-')
plt.axis([0, 1.0, -5.0, 1])
plt.show()
#atimes = (np.log(atimes))
#plt.plot(sigs, atimes)
#plt.plot(sigs, atimes2)
#plt.show()
