import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from brokenaxes import brokenaxes
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



plt.figure(1)
bax1 = brokenaxes(xlims=((0, 10), (150, 210), (350, 360)), hspace=.05)
bax1.set_ylim(0,44)
bax1.set_xlabel("Neural label (degrees)", labelpad=22)
bax1.set_ylabel('Firing rate', labelpad=24)

plt.figure(3)
bax = brokenaxes(xlims=((0, 10), (150, 210), (350, 360)), hspace=.05)
bax.set_ylim(0,16)
bax.set_xlabel("Neural label (degrees)", labelpad=22)
bax.set_ylabel('Firing rate', labelpad=24)
gradedlist = [0.3, 0.5, 0.7, 1]

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
        bax1.plot(Rate, color = 'red', label=A2,linewidth = 2, alpha = gradedlist[i], zorder=10)
        bax1.legend(title='Stim. amp.', fontsize=13, title_fontsize=13, bbox_to_anchor=(1.05, 1))
        
        
    inputrecord = 0 #memory period
    for t in range(DelayTime): 
        SynActi,state,Vden,VTotal,Rate = OneTimeStep(SynActi,state,Vden,VTotal,Rate, Noise)
    MemoryRecord.append(Rate[180*cont])
    
    if TrialRatePlot =='ON':
        plt.figure(3)
        bax.plot(x, Rate, color = 'blue', linewidth = 2, alpha = gradedlist[i], zorder=10)
        
        
for ax in bax.axs[1:]:
    ax.spines['left'].set_visible(False)


plt.figure(4)

M = np.arange(0.1,30,0.05)
gamma = (1+inhpara*kappa)
def finv(M):
    I = gamma* ( Tu/(np.exp( -(gamma*M/(2*DenVol))**2 /(2*SD**2)) * 18/(( gamma*M/(2*DenVol) +2)**2))  -M)
    return I
plt.plot(finv(M),M, '-', linewidth = 2, color='gray', alpha=1)


plt.plot(InputRecord, MemoryRecord, '-', color = 'blue')
plt.xlim([0,60])
plt.ylim([0,16])
plt.xlabel('Stimulus amplitude', labelpad=0)
plt.ylabel('Memory amplitude', labelpad=0)

sns.despine()

plt.show()
stop = timeit.default_timer()
print('Time: ', stop - start)

'''
#=========================================================================
Following changes are made to the default code to generate various plots
For Fig. 4D, 
1. TrialRatePlot ='OFF'
2. change the loop for i to be 'for i in range(601):'
3. set A2 = 0.1*i 

#=========================================================================
For Fig. 4G, to get simulation result for the linear input-memory mapping:
1. replace weight section to
weight = np.zeros(( len(x), len(x) ))
for xx in range(len(x)):
    dist = np.minimum( abs(x[xx]-z), 360-abs(x[xx]-z)  )
    dist[dist>9] = 9.1
    wtemp = np.exp(dist**2/200)*Tu*(1+inhpara*kappa)*A/(2*DenVol*dist*A  + 2*DenVol*dist/(1+inhpara*kappa)-B)
    wtemp[wtemp==np.min(wtemp)] = 0.
    #wtemp may not be monotonically decreasing, as mentioned in materials and methods, it is cut off
    #after it stops monotonically decreasing. Here, it happens if the distance is beyond 9.
    weight[ xx, :] = wtemp

2. Then set TrialRatePlot ='OFF'
3. set the loop for i to be 'for i in range(36):'
4. set 'A2 = i'
5. run following code to get Fig. 4G.
plt.rcParams["figure.figsize"] = (4,2.75)
plt.figure(4)
plt.xlim([0,36])
plt.ylim([0,18])
plt.xlabel('Stimulus amplitude', labelpad=0)
plt.ylabel('Memory amplitude', labelpad=0)
plt.plot(np.arange(0, 60), A*np.arange(0, 60)+B, '-', linewidth = 2, color='gray', alpha=1) #for a linear line
plt.plot(InputRecord, MemoryRecord, '.',  color = 'blue', clip_on=False, zorder=3)
sns.despine()
'''