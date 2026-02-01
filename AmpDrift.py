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
Tu  = 9
Td  = 2
DenVol = 1
Cm = 50
taunmda = 100
inhpara = 1/360

#input paras==================================================================
EncodingTime = 1000
DelayTime    = 10000
A2           = 15# input gaussian amp
B2           = 180 #location of input gaussian
SD           = 10 # input gaussian variance. note here I set c = 2\sigma^2 in standard gaussian

#simulation constants==========================================================
cont = 1
continousfactor = 1.0/cont
x = np.arange(0, 360.0, continousfactor)#must be 1.0 so not set to integer data
z = np.arange(0, 360.0, continousfactor)

#para used for approx linear input memory relation
A= 0.5 #linear memory slope
B= -0.6 #linear memory intercept
kappa = 10*np.sqrt(2*np.pi) #calculated kappa constant by integration with current para set

#weight========================================================================
weight = np.zeros(( len(x), len(x) ))
for xx in range(len(x)):
    dist = np.minimum( abs(x[xx]-z), 360-abs(x[xx]-z)  )
    dist[dist>9] = 9.1
    wtemp = np.exp(dist**2/200)*Tu*(1+inhpara*kappa)*A/(2*DenVol*dist*A  + 2*DenVol*dist/(1+inhpara*kappa)-B)
    wtemp[wtemp==np.min(wtemp)] = 0.
    #wtemp may not be monotonically decreasing, as mentioned in materials and methods, it is cut off
    #after it stops monotonically decreasing. Here, it happens if the distance is beyond 9.
    weight[ xx, :] = wtemp

Simulations = 10
condition = 3
TotalTime = EncodingTime + DelayTime

DriftingMatrix =      np.zeros((len(x), TotalTime, Simulations, condition))


#Noise=========================================================================
NoiseRaw = np.random.normal(0, 1, [TotalTime,Simulations, cont * 360])
NoiseStdList = []

for ss in range(condition):
    
    listofamp = [10,15,20]
    A2 = listofamp[ss]
    NoiseStd = 3 #3 6 9
    NoiseStdList.append(NoiseStd)
    Noise = NoiseRaw * NoiseStd
    
    for trial in range(Simulations):
        print(trial, end='')
        #initialize====================================================================
        SynActi = 0 * x
        state   = np.zeros((len(x), len(x)), dtype=bool)
        Vden    = 0 * x
        VTotal  = 0 * x
        Rate    = 0 * x
        
        #input=========================================================================
        inputrecord = A2*np.exp( -np.minimum( abs(z - B2), 360-abs(z - B2))**2 /(2*SD**2))
        for t in range(EncodingTime): 
            NoiseForNow = Noise[t, trial, :]
            SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, NoiseForNow)
            DriftingMatrix[:, t, trial, ss] = Rate
        
        inputrecord = 0
        for t in range(DelayTime): 
            NoiseForNow = Noise[t+EncodingTime, trial, :]
            SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, NoiseForNow)
            DriftingMatrix[:, t+EncodingTime, trial, ss] = Rate
            
        plt.figure(333)
        plt.plot(Rate)
        sns.despine()


#Fitting=========================================================================
FitAll = np.zeros((4, condition, TotalTime, Simulations))
for ss in range(condition):
    for t in range(TotalTime):
        for trial in range(Simulations):
            RateNow = DriftingMatrix[:, t, trial, ss]
            if np.max(RateNow) > np.mean(RateNow)*2:
                #if determine memory amp by taking largest N firing rate
                FitAll[0, ss, t,trial] = np.mean( np.sort(RateNow)[350:] ) 
                FitAll[2, ss, t,trial] = np.sum( RateNow * x )/np.sum(RateNow)
                
print( np.std(FitAll[2, :, -1,:], axis = 1) )


colors = ['red', 'blue','orange', 'magenta', 'cyan', 'maroon', 'grey']
plt.rcParams["figure.figsize"] = (4,1.25)
for ss in range(condition): 
    plt.figure('Amp')
    plt.plot(np.arange(0, DelayTime-300, 1),  FitAll[0, ss, EncodingTime+300:,:], '-', color = colors[ss], alpha = 0.3)
    plt.xlabel('Time (ms)' , labelpad=0)
    #plt.ylabel('Memory amplitude')
    plt.gca().set_xlim([0,DelayTime-300])
    plt.gca().set_ylim([0,13])
    sns.despine()


plt.rcParams["figure.figsize"] = (4,2.75)
    


stop = timeit.default_timer()
print('Time: ', stop - start)

plt.show()
import os
os.system('say "program finished"')