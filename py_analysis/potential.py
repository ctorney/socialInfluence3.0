
#!/usr/bin/python

import sympy as sp
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib as mpl
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from matplotlib import colors

from numpy import *

fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

NA = 128
Ns = 8
K = 50       
ws = 0.524
# Set the default color cycle
mpl.rcParams['axes.color_cycle'] = ['7A68A6', 'A60628', '348ABD', 'd8950c',   '467821', 'E24A33']
font = {'family' : 'normal',        'weight' : 'normal',        'size'   : 18}
mpl.rc('font', **font)

cls = ["#7A68A6", "#A60628", "#348ABD", "#d8950c",   "#467821", "#E24A33"]
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

def EXTAPPROX(X,sigma):  
    sig = float(sigma)
    A = np.log(K)/(2.0) - 1/sig
    B = np.log(K)
    gc = np.log(K)*sig*(1.0-2.0*X)/(2.0)
    VR = 0.5*B**2*X*(1-X)/float(Ns)
    if gc<1:
        return 1.0 - (m.exp(A)*0.5*m.exp((-B*X)))*(1.0 +  VR -  ((VR/(2*m.exp(-(A-B*X))-1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
    return m.exp(-A)*0.5*m.exp((B*X))*(1.0 +  VR +  ((VR/(1.0+VR))*alpha * B * X * (Ns-1)/float(Ns)))
def atup(X,sig):
    return (1-X)*(EXTSWUP(X,sig))
    return (1-X)*(EXTAPPROX(X,sig))
def atdown(X,sig): 
    return (X)*(1.0-EXTSWUP(X,sig))
    return (X)*(1.0-EXTAPPROX(X,sig))
numA = 15
sigs = np.zeros(numA)
atimes = np.zeros(numA)
atimes2 = np.zeros(numA)
alpha = 0.0
sigma = 20.0
dx = 1/128.0
ups = [atup(x+dx,sigma) for x in arange(0,1,dx)]
downs = [atdown(x+dx,sigma) for x in arange(0,1,dx)]
pot=-np.log(np.divide(ups,downs).astype(float32))
pot=np.cumsum(pot)
ax0.plot(arange(0,1,dx),pot ,   color = cls[1],linestyle='-')
for p in pot: print p


alpha = 0.22
ups = [atup(x+dx,sigma) for x in arange(0,1,dx)]
downs = [atdown(x+dx,sigma) for x in arange(0,1,dx)]
pot=-np.log(np.divide(ups,downs).astype(float32))
pot=np.cumsum(pot)
for p in pot: print p
ax0.plot(arange(0,1,dx),pot ,   color = cls[1],linestyle='-')
    
        
            #if (j%4==0):
                    
#for jj in range(6):
       
           #grid = np.load('output/an-time-' + str(j) + '.npy')
               
                   #if (j%4==0):
                           #ax0.plot(grid[:,0],grid[:,1] ,label = str(j),   linestyle='--', color = cls[jj])

#ax0.set_ylabel('Time before transition')    
l=ax0.legend(loc=1,     ncol=1, borderaxespad=0.5, prop={'size':12})
#l.set_title('P', prop={'size':12}) 
#l.get_title().set_fontproperties({'size':12}) 
ax0.axis([0.0,1.0, -300, 8])
ax1.axis([0.0, 1.0, -300, 8])
#ax0.set_yscale('log')
ax1.get_yaxis().set_visible(False)
ax0.set_title('small network')
ax1.set_title('large network')
ax0.set_ylabel('potential')  

fig.text(0.5, 0.04, 'fraction of population correct', ha='center', va='center')
plt.savefig('fig-potential2.png', format='png', dpi=100)
