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
from scipy.optimize import curve_fit

start = timeit.default_timer()
sns.set(font_scale=2)
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

Simulations = 400
condition = 3
TotalTime = EncodingTime + DelayTime

#DriftingMatrix =      np.zeros((len(x), TotalTime, Simulations, condition))
FitAll = np.zeros((4, condition, DelayTime, Simulations))

#Noise=========================================================================
NoiseStdList = []

for ss in range(condition):
    #changing AMP case
    Tu  = 4.4
    Td  = 1.6
    ampinc = [3, 8, 16]
    A2  = ampinc[ss]
    NoiseStd = 10
    

    
    NoiseStdList.append(NoiseStd)
    print(Tu, end='')
    for trial in range(Simulations):
        #print(trial, end='')
        #initialize====================================================================
        SynActi = 0 * x
        state   = np.zeros((len(x), len(x)), dtype=bool)
        Vden    = 0 * x
        VTotal  = 0 * x
        Rate    = 0 * x
        
        inputrecord = A2*np.exp( -np.minimum( abs(z - B2), 360-abs(z - B2))**2 /(2*SD**2))
        for t in range(EncodingTime): 
            NoiseForNow = np.random.normal(0, 1, cont * 360) * NoiseStd
            SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, NoiseForNow)
            #DriftingMatrix[:, t, trial, ss] = Rate
        
        inputrecord = 0
        for t in range(DelayTime): 
            NoiseForNow = np.random.normal(0, 1, cont * 360) * NoiseStd
            SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, NoiseForNow)
            #DriftingMatrix[:, t+EncodingTime, trial, ss] = Rate
            #if determine memory amp by taking largest N firing rate
            FitAll[0, ss, t,trial] = np.mean( np.sort(Rate)[350:] ) 
            RateNorm = Rate - 2.5
            RateNorm[RateNorm<0] = 0 
            FitAll[2, ss, t,trial] = np.sum( RateNorm * x )/np.sum(RateNorm)

        
        plt.figure(333)
        plt.plot(Rate, color = 'g', alpha = 0.1)
        sns.despine()

#=========================================================================
clist = ['lightgrey', 'grey', 'black']
#plt.rcParams["figure.figsize"] = (8,2.5)
plt.figure('Amp')
plt.plot(np.arange(0, DelayTime-300, 1),  FitAll[0, 0, 300:,:], '-', color = clist[0], alpha = 1)
plt.plot(np.arange(0, DelayTime-300, 1),  FitAll[0, 1, 300:,:], '-', color = clist[1], alpha = 1)
plt.plot(np.arange(0, DelayTime-300, 1),  FitAll[0, 2, 300:,:], '-', color = clist[2], alpha = 1)
plt.xlabel('Time (ms)')
plt.ylabel('Memory Amplitude')
plt.gca().set_xlim([0,DelayTime-300])
plt.gca().set_ylim([0,22])
sns.despine()

plt.figure('Location')
plt.plot(np.arange(0, DelayTime-300, 1),  np.var(FitAll[2, 0, 300:,:], axis=1)-np.min(np.var(FitAll[2, 0, 300:,:], axis=1)) + np.min(np.var(FitAll[2, 2, 300:,:], axis=1)), '-', color = clist[0], alpha =1)
plt.plot(np.arange(0, DelayTime-300, 1),  np.var(FitAll[2, 1, 300:,:], axis=1)-np.min(np.var(FitAll[2, 1, 300:,:], axis=1)) + np.min(np.var(FitAll[2, 2, 300:,:], axis=1)), '-', color = clist[1], alpha =1)
plt.plot(np.arange(0, DelayTime-300, 1),  np.var(FitAll[2, 2, 300:,:], axis=1), '-', color = clist[2], alpha =1)
plt.xlabel('Time (ms)')
plt.ylabel('Location Variance')
plt.gca().set_xlim([0,DelayTime-300])
plt.gca().set_ylim([0,7])
sns.despine()

    
plt.rcParams["figure.figsize"] = (8,5.5)


stop = timeit.default_timer()
print('Time: ', stop - start)

plt.show()
import os
os.system('say "program finished"')