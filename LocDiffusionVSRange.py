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
sns.set_context("notebook", font_scale=2)
plt.rcParams.update({
    'font.size': 13,           # Base font size
    'axes.titlesize': 13,      # Title
    'axes.labelsize': 13,      # X/Y labels
    'xtick.labelsize': 13,     # X tick labels
    'ytick.labelsize': 13,     # Y tick labels
    'legend.fontsize': 13,     # Legend
})
plt.rcParams["figure.figsize"] = (4,2.75)
start = timeit.default_timer()
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
DenVol = 1
Cm = 50
taunmda = 100
inhpara = 1/360

#input paras==================================================================
EncodingTime = 1000
DelayTime    = 10000
B2           = 180 #location of input gaussian
SD           = 10 # input gaussian variance. note here I set c = 2\sigma^2 in standard gaussian

#simulation constants==========================================================
cont = 1
continousfactor = 1.0/cont
x = np.arange(0, 360.0, continousfactor)#must be 1.0 so not set to integer data
z = np.arange(0, 360.0, continousfactor)


#weight========================================================================
weight = np.zeros(( len(x), len(x) ))
for xx in range(len(x)):
    dist = np.minimum( abs(x[xx]-z), 360-abs(x[xx]-z)  )
    weight[ xx, :] = 18/(( dist+2)**2)

#simulation para========================================================================
Simulations = 400
condition = 11
TotalTime = EncodingTime*2 + DelayTime

FitAll = np.zeros((4, condition, DelayTime, Simulations))

NoiseStdList = []
BiRange = []

for ss in range(condition):
    #condition========================================
    #change bistable range
    TuF  = 3
    TdF  = 3
    
    TuC  = TuF + 0.2*ss
    TdC  = TdF - 0.2*ss
    A2  = 15
    NoiseStd = 10
    BiRange.append(0.2*ss*2)
    #==========================================
    
    NoiseStdList.append(NoiseStd)
    print(TuC, end='')
    for trial in range(Simulations):
        #initialize====================================================================
        SynActi = 0 * x
        state   = np.zeros((len(x), len(x)), dtype=bool)
        Vden    = 0 * x
        VTotal  = 0 * x
        Rate    = 0 * x
        
        #input=========================================================================
        Tu  = np.copy(TuF)
        Td  = np.copy(TdF)
        inputrecord = A2*np.exp( -np.minimum( abs(z - B2), 360-abs(z - B2))**2 /(2*SD**2))
        for t in range(EncodingTime): 
            NoiseForNow = np.random.normal(0, 1, cont * 360) * 0
            SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, NoiseForNow)
        
        #this is the initial delay time to stabilize noise free activity
        inputrecord = 0
        for t in range(EncodingTime): 
            NoiseForNow = np.random.normal(0, 1, cont * 360) * 0
            SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, NoiseForNow)
        
        Tu  = np.copy(TuC)
        Td  = np.copy(TdC)
        #this is the the delay time to observe noise effect
        inputrecord = 0
        for t in range(DelayTime): 
            NoiseForNow = np.random.normal(0, 1, cont * 360) * NoiseStd
            SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, NoiseForNow)
            
            #if determine memory amp by taking largest N firing rate
            FitAll[0, ss, t,trial] = np.mean( np.sort(Rate)[350:] ) 
            RateNorm = Rate - 2.5
            RateNorm[RateNorm<0] = 0 
            FitAll[2, ss, t,trial] = np.sum( RateNorm * x )/np.sum(RateNorm)


#colors = ['blue', 'orange','orange']
for ss in range(condition): 
    plt.figure('Amp')
    plt.plot(np.arange(0, DelayTime-300, 1),  FitAll[0, ss, 300:,:], '-', alpha = 1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Memory amplitude')
    plt.gca().set_xlim([0,DelayTime-300])
    plt.gca().set_ylim([0,22])
    sns.despine()

    plt.figure('LocationVar')
    plt.plot(np.arange(0, DelayTime-300, 1),  np.var(FitAll[2, ss, 300:,:], axis=1), '-', alpha =1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Location variance')
    plt.gca().set_xlim([0,DelayTime-300])
    plt.gca().set_ylim([0,53])
    sns.despine()

    
plt.rcParams["figure.figsize"] = (3.25,2.75)

plt.figure('Range')
plt.plot(BiRange,  np.var(FitAll[2, :, -1,:], axis=1)  , '.', color = 'black', alpha =1, clip_on=False, zorder=3)
plt.plot(BiRange[0],  np.var(FitAll[2, :, -1,:], axis=1)[0]  , '.', color = 'blue', alpha =1, clip_on=False, zorder=3)
plt.plot(BiRange[4],  np.var(FitAll[2, :, -1,:], axis=1)[4]  , '.', color = 'orange', alpha =1, clip_on=False, zorder=3)
plt.plot(BiRange[8],  np.var(FitAll[2, :, -1,:], axis=1)[8]  , '.', color = 'green', alpha =1, clip_on=False, zorder=3)



plt.xlabel('Bistable range')
plt.ylabel('Final location variance')
plt.gca().set_xlim([0,np.max(BiRange)+0.1])
plt.gca().set_ylim([0,53])
sns.despine()
plt.show()

plt.rcParams["figure.figsize"] = (4,2.75)

stop = timeit.default_timer()
print('Time: ', stop - start)


import os
os.system('say "program finished"')



colors = ['blue', 'orange','green']
for ss in range(3): 

    plt.figure(4)
    plt.plot(np.arange(0, DelayTime-300, 1),  np.var(FitAll[2, ss*4, 300:,:], axis=1), '-', color = colors[ss], alpha =1 , clip_on=False, zorder=3)
    plt.xlabel('Time (ms)')
    plt.ylabel('Location variance')
    plt.gca().set_xlim([0,DelayTime-300])
    plt.gca().set_ylim([0,53])
    sns.despine()

plt.show()


'''
The default code generates 7A,B. To get 7C:
1. change the weight section to
weight = np.zeros(( len(x), len(x) ))
for xx in range(len(x)):

    signed_dist = (z - x[xx] + 180) % 360 - 180
    dist = np.abs(signed_dist)

    base_weight = 18 / ((dist + 2) ** 2)

    # directional modulation
    scale = np.ones_like(base_weight)
    scale[signed_dist < 0] *= 1.05   # left side stronger
    scale[signed_dist > 0] *= 0.95   # right side weaker

    weight[xx, :] = base_weight * scale

2. change 'condition = 11' to 'condition = 2'.

3. change 'TuC  = TuF + 0.2*ss' and 'TdC  = TdF - 0.2*ss' to 'TuC  = TuF + 1.6*ss' and 'TdC  = TdF - 1.6*ss'

4. run the following code to get 7C. (Note that the location plotted is substracted by a constant offset.)
plt.rcParams["figure.figsize"] = (4,2.75)
plt.figure('Location')
plt.plot(np.arange(0, DelayTime-300, 1),  (np.mean(FitAll[2, 0, 300:,:], axis=1)-183), '-', color = 'blue', alpha =1)
plt.plot(np.arange(0, DelayTime-300, 1),  (np.mean(FitAll[2, 1, 300:,:], axis=1)-182), '-', color = 'green', alpha =1)
plt.xlabel('Time (ms)')
plt.ylabel('Location deviation')
plt.gca().set_xlim([0,DelayTime-300])
plt.gca().set_ylim([0,40])
sns.despine()
'''