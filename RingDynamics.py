import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
Tu  = 9 
Td  = 2
DenVol = 1
Cm = 50
inhpara = 1/360

#discreteness paras===============================================================
#this controls discreteness of ring model. cont=1 for 360 neurons of the network. 
cont = 1
continousfactor = 1.0/cont

#other paras==================================================================
EncodingTime = 1000 #in ms
DelayTime    = 1000
B2           = 180 #location of input Gaussian
SD           = 10 # input Gaussian SD. 

#initialization==========================================================
x = np.arange(0, 360.0, continousfactor)
z = np.arange(0, 360.0, continousfactor)

#para used for approx linear input memory relation
A= 0.5 #linear memory slope
B= -0.6 #linear memory intercept
kappa = 10*np.sqrt(2*np.pi) #calculated kappa constant by integration with current para set

#weight========================================================================
weight = np.zeros(( len(x), len(x) ))
for xx in range(len(x)):
    dist = np.minimum( abs(x[xx]-z), 360-abs(x[xx]-z)  )
    weight[ xx, :] = 18/(( dist+2)**2)


#============================================================================
InputRecord = []
MemoryRecord = []
TrialRatePlot ='ON'

for i in range(4):
    #initialize====================================================================
    SynActi = 0 * x
    state   = np.zeros((len(x), len(x)), dtype=bool)
    Vden    = 0 * x
    VTotal  = 0 * x
    Rate    = 0 * x
    Noise   = 0 * x
    
    #input=========================================================================
    A2 = 8+8*i
    InputRecord.append(A2)
    
    inputrecord = A2*np.exp( -np.minimum( abs(z - B2), 360-abs(z - B2))**2 /(2*SD**2))
    for t in range(EncodingTime): 
        SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise)
    
    if TrialRatePlot =='ON':
        plt.figure(1)
        plt.plot(Rate, color = 'maroon', label=A2, alpha = i*0.25+0.25)
        plt.xlim([0,360])
        plt.ylim([0,44])
        plt.xlabel('Neural Label (degrees)')
        plt.ylabel('Firing Rate')
        plt.legend(title='Input Amp')
        sns.despine()
        
    
    inputrecord = 0 #memory period
    for t in range(DelayTime): 
        SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise)
    MemoryRecord.append(Rate[180*cont])
    
    if TrialRatePlot =='ON':
        plt.figure(2)
        plt.plot(x, Rate, color = 'green', alpha = i*0.25+0.25)
        plt.xlim([0,360])
        plt.ylim([0,16])
        plt.xlabel('Neural Label (degrees)')
        plt.ylabel('Firing Rate')
        sns.despine()
        
        plt.figure(3)
        plt.plot(x, Rate, color = 'green', linewidth = 5,alpha = i*0.25+0.25)
        plt.xlim([150,210])
        plt.ylim([0,16])
        #plt.xlabel('Neural Label (degrees)')
        #plt.ylabel('Firing Rate')
        sns.despine()
    
    


plt.rcParams["figure.figsize"] = (8,5.5)
plt.figure(4)
plt.plot(InputRecord, MemoryRecord, '-', color = 'blue')
plt.xlim([0,60])
plt.ylim([0,16])
plt.xlabel('Input Amplitude')
plt.ylabel('Memory Amplitude')
sns.despine()



plt.show()
stop = timeit.default_timer()
print('Time: ', stop - start)

'''
#=========================================================================
Following changes are made to the default code to generate various plots
For Fig. 4D, 
1. TrialRatePlot ='OFF'
2. change the loop for i to be 'for i in range(800):'
3. set A2 = 0.1*i #basically it makes input amplitude continously increasing from 0 to 80.


#=========================================================================
For Fig. 4F,G, to get simulation result for the linear input-memory mapping:
1. set: EncodingTime = 2000 and DelayTime    = 2000
2. set cont=20 (larger value makes the simulation goes to the continuum limit 
3. replace weight section to
weight = np.zeros(( len(x), len(x) ))
for xx in range(len(x)):
    dist = np.minimum( abs(x[xx]-z), 360-abs(x[xx]-z)  )
    dist[dist>9] = 9.1
    wtemp = np.exp(dist**2/200)*Tu*(1+inhpara*kappa)*A/(2*DenVol*dist*A  + 2*DenVol*dist/(1+inhpara*kappa)-B)
    wtemp[wtemp==np.min(wtemp)] = 0.
    #wtemp may not be monotonically decreasing, as mentioned in materials and methods, it is cut off
    #after it stops monotonically decreasing. Here, it happens if the distance is beyond 9.
    weight[ xx, :] = wtemp
4. Set 'A2 = 5*i+5'. 
5. Run code to get Fig. 4G inset. Then set TrialRatePlot ='OFF'
6. set the loop for i to be 'for i in range(36):'
7. set 'A2 = i'
8. run following code to get Fig. 4G.
plt.rcParams["figure.figsize"] = (8,5.5)
plt.figure(4)
plt.plot(InputRecord, MemoryRecord, 'o', color = 'blue')
plt.plot(np.arange(0, 60), A*np.arange(0, 60)+B, '--', color = 'red') #for a linear line
plt.xlim([0,35])
plt.ylim([0,18])
plt.xlabel('Input Amplitude')
plt.ylabel('Memory Amplitude')
sns.despine()
'''