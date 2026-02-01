#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:01:04 2021

@author: jiachengxu
"""

import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve
start = timeit.default_timer()
sns.set(font_scale=2)
plt.rcParams.update({
    'font.size': 13,           # Base font size
    'axes.titlesize': 13,      # Title
    'axes.labelsize': 13,      # X/Y labels
    'xtick.labelsize': 13,     # X tick labels
    'ytick.labelsize': 13,     # Y tick labels
    'legend.fontsize': 13,     # Legend
})
plt.rcParams["figure.figsize"] = (4,2.75)
sns.set_style("ticks")


def OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise):
    SynActi += (Rate - SynActi) #* dt/50 optional to make SynActi slower, no qualtative changes
    EInputRec = weight * SynActi
    
    Prestate = np.copy(state)
    #if input more than Tu, dendrite actives to up
    EThUp = x*0 + Tu
    state = (EInputRec  > EThUp.reshape(-1,1))  
    
    #if input more than Td, dendrite can be maintained in up state
    #this defines maximal sustainable dendrites
    EThDown = x*0 + Td
    Maxstate = (EInputRec  >  EThDown.reshape(-1,1)) 
    
    #actual activated dendrites need (0. previously in up state) 1. activated by crossing Tu; 2. sustainable 
    state = (Prestate | state) & Maxstate
    
    Vden = np.sum( state, axis = 1) * continousfactor * DenVol
    VTotalNow = inputrecord + Vden - inhpara * np.sum(Rate)/cont + Noise
    
    VTotal += (VTotalNow-VTotal)/Cm
    Rate = np.copy(VTotal)
    Rate[VTotal < 0 ] = 0
    
    return SynActi,state,Vden,VTotal,Rate

#dendritic paras===============================================================
Tu  = 9
Td  = 2
DenVol = 1
inhpara = 1/360

#input paras==================================================================
dt = 0.01
Cm = 50/dt
EncodingTime = int(1000/dt)
DelayTime    = int(1000/dt)
A2           = 80# input gaussian amp
B2           = 180 #location of input gaussian
SD           = 10 # input gaussian variance. note here I set c = 2\sigma^2 in standard gaussian

#simulation constants==========================================================
cont = 1
continousfactor = 1.0/cont
x = np.arange(0, 360.0, continousfactor)#must be 1.0 so not set to integer data
z = np.arange(0, 360.0, continousfactor)

Choice=180 #the neuron to generate multistability band. 
#by default it is the peak neuron, choose other neurons give qualitatively similar band shape.

TuX = []
TuY = []
DelayX = []
DelayY = []
TdX = []
TdY = []

#para used for approx linear input memory relation
A= 0.5 #linear memory slope
B= -0.6 #linear memory intercept
kappa = 10*np.sqrt(2*np.pi) #calculated kappa constant by integration with current para set

#weight========================================================================
weight = np.zeros(( len(x), len(x) ))
for xx in range(len(x)):
    dist = np.minimum( abs(x[xx]-z), 360-abs(x[xx]-z)  )
    weight[ xx, :] = 18/(( dist+2)**2)


#initialize====================================================================
SynActi = 0 * x
state   = np.zeros((len(x), len(x)), dtype=bool)
Vden    = 0 * x
VTotal  = 0 * x
Rate    = 0 * x
Noise   = 0 * x
ChangeInh = 0

InhX = []
InhY = []
SynActiList = []

#input=========================================================================
inputrecord = A2*np.exp( -np.minimum( abs(z - B2), 360-abs(z - B2))**2 /(2*SD**2))

#encoding=========================================================================
for t in range(EncodingTime): 
    SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise)
    
    #tracing out Tu
    TuX.append(SynActi[Choice*cont])
    TuY.append(Vden[Choice*cont]- inhpara * np.sum(Rate)/cont)

#memory=========================================================================
inputrecord = 0
for t in range(DelayTime): 
    SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise)
    
#record the memory point, which is connected to the origin 
DelayX.append(SynActi[Choice*cont])
DelayY.append(Vden[Choice*cont]- inhpara * np.sum(Rate)/cont)

plt.figure(2)
plt.plot(Rate)
    
#erasure=========================================================================
inputrecord = -10 # inh should be large enough to erase memory, otherwise Td curve does not show up
for t in range(DelayTime*1): 
    SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise)
    
    #tracing out Td
    TdX.append(SynActi[Choice*cont])
    TdY.append(Vden[Choice*cont]- inhpara * np.sum(Rate)/cont)
    
#plot=========================================================================
plt.figure(1)
def W(x):
    W = np.exp( -x**2 /(2*SD**2)) * 18/(( x+2)**2)
    return W
def Winv(y, x0=1.0):
    sol = y*0
    for i in range(len(y)):
        # Solve W(x) = y starting from initial guess x0
        sol[i] = fsolve(lambda x: W(x) - y[i], x0)
    return sol
E_A = np.arange(0.1,60,0.05)
TuCurve =2*DenVol*Winv(Tu/E_A) -E_A * inhpara*kappa
TdCurve =2*DenVol*Winv(Td/E_A) -E_A * inhpara*kappa
plt.plot(E_A, TuCurve, '-', linewidth = 2, color='gray')
plt.plot(E_A, TdCurve,  '-', linewidth = 2, color = 'gray')

plt.plot([0, DelayX[0]], [0,DelayY[0]], 'black', linestyle=(0, (10, 10)))
plt.plot(TuX,TuY, '-', linewidth = 2, color='black')
plt.plot(TdX, TdY,  '-', linewidth = 2, color = 'black')

axes = plt.gca()
axes.set_xlim([0,40])
axes.set_ylim([-0.5,20])
plt.xlabel('Peak firing rate', labelpad=0)
plt.ylabel('Total recurrent feedback (solid) \n& activity decay (dashed)', labelpad=0)
#axes.set_aspect('equal')
#plt.axis('off')
sns.despine()



plt.show()
stop = timeit.default_timer()
print('Time: ', stop - start)
import os
os.system('say "your program has finished"')

'''
For Fig. 3B, the discrete part of autapse multistable band is generated by
with following changes to the default code here:
1. set 'inhpara = 0'
2. set 'inputrecord = A2*np.exp( -np.minimum( abs(z - B2), 360-abs(z - B2))**2 /(2*SD**2))'
to 'inputrecord = 14'
3. replace weight section to
for xx in range(len(x)):
    for yy in range(len(x)):
        #note that to map a ring model network (used here) to an autapse model, 
        #weight function differs by a factor of 2, as the following lines for.
        if xx-yy>=0: 
            distance = xx-yy
        elif xx-yy<0:
            distance = xx-yy+360
        weight[xx,yy]=4/(( distance  + 1)**1)
4. some optional but relevant code to plot discrete autapse curves and continuum limit (see Fig3autapse.py) together
plt.rcParams["figure.figsize"] = (4*1.33,2.75*1.33)
plt.figure('TCurve')
plt.plot(TuEff, label*beta, linewidth = 2, color = 'black') #label*beta makes y dendritic rate
plt.plot(TdEff, label*beta, linewidth =2, color = 'black')
plt.plot(TuX,TuY, '-', linewidth = 2, color='black', alpha = 0.5)
plt.plot(TdX, TdY,  '-', linewidth = 2, color = 'black', alpha=0.5)

#plt.plot(label, label/beta)
axes = plt.gca()
axes.set_xlim([0,23])
axes.set_ylim([0,11])
plt.yticks(np.arange(0, 11, 2))
axes.set_aspect('equal')
#plt.axis('off')
sns.despine()
plt.rcParams["figure.figsize"] = (4,2.75)

#========================================================================
For Fig. 4F and 6A
1. change the weight section to
weight = np.zeros(( len(x), len(x) ))
for xx in range(len(x)):
    dist = np.minimum( abs(x[xx]-z), 360-abs(x[xx]-z)  )
    dist[dist>9] = 9.1
    wtemp = np.exp(dist**2/200)*Tu*(1+inhpara*kappa)*A/(2*DenVol*dist*A  + 2*DenVol*dist/(1+inhpara*kappa)-B)
    wtemp[wtemp==np.min(wtemp)] = 0. 
    #wtemp may not be monotonically decreasing, as mentioned in materials and methods, it was cut off
    #after it stops monotonically decreasing. Here, it happens if the distance is beyond 9.
    weight[ xx, :] = wtemp
2. for function W(x), redefine as 
def W(x):
    W = np.exp( -x**2 /(2*SD**2)) * np.exp(x**2/200)*Tu*(1+inhpara*kappa)*A/(2*DenVol*x*A  + 2*DenVol*x/(1+inhpara*kappa)-B)
    return W
#========================================================================
For Fig. 5
5A: set A2  = 40 or A2  = 20


5B: set A2  = 20
change memory section to
inputrecord = 0
for t in range(DelayTime): 
    SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise)

inputrecord = 80*np.exp( -np.minimum( abs(z - B2), 360-abs(z - B2))**2 /(2*SD**2))
for t in range(EncodingTime): 
    SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise)
    #tracing out Tu
    TuX.append(SynActi[Choice*cont])
    TuY.append(Vden[Choice*cont]- inhpara * np.sum(Rate)/cont)

inputrecord = 0
for t in range(DelayTime): 
    SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise)


'''