import numpy as np
import matplotlib.pyplot as plt
import timeit
import seaborn as sns
start = timeit.default_timer()
sns.set(font_scale=2)
sns.set_style("ticks")

x =  [0.5, 0.5, 0.007, 0.014, 17.2, 0., 0.38, 48,    5.5, 0.54,  1., 8.6,  5.2, 20,  5, 0]

tau_AMPA	=2		# time constant of AMPA channel closing in ms
tau_NMDA	=100	#	"	of NMDA				"			"
alpha	    =0.5	# ratio of unactivated receptors activated by spike
Cm		    = 10    #Membrane Capacitance 		in nF/mm2
Mg	        = 0.5	#[Mg2+] in mM

dt          = 0.01  #time in ms

VTh         = -50   #firing threshold of the soma
spike_duration = int(3/dt) #spike duration
peak = 30           #peak of spike

times = int(1400/dt)      #time in ms
EncodingTime = int(1400/dt)
NoOfDendrites = 10

g_L_s       =x[0]    #Leak Conductance of soma			in uS/mm2	
g_L_d       =x[1]    #Leak Conductance of dendrite
g_DtoS      =x[2]    #Soma-Dendrite Coupling Conductance 
g_StoD      =x[3]    #Dendrite-Soma Coupling Conductance
Tonic_s     =x[4]    


#initialize
sAMPA  =np.zeros( NoOfDendrites )
sNMDA  =np.zeros( NoOfDendrites )
sAMPA_soma =0
V_d    = np.zeros( NoOfDendrites ) - 70
V_s    = -70

SpikeRecord = np.zeros( times , dtype=bool) 
V_d_Record = np.zeros( [NoOfDendrites, times] ) 
V_s_Record = np.zeros( times ) 

InputList = []
V_dList = []

Range  = 71
VIndenpendentSim = np.zeros( [Range*2, NoOfDendrites] ) 

#assign conductance
weight_AMPA  = np.zeros( NoOfDendrites )+x[6] 
weight_NMDA  = np.zeros( NoOfDendrites )
for i in range(NoOfDendrites):
    weight_NMDA[i] = x[7]/((i+x[8])**x[9])
weight_KIR = x[11] 
weight_AMPA_soma = x[12]

StiDen = [0,1,2,3,4,5,6,7,8,9] #dendrites stimulated by input 

#rate increase
for step in range(Range):
    #generate external input signal
    ExcInput = np.zeros( [1,times], dtype=bool) 
    SignalRate = (step+1)
    ITI = 1000/SignalRate/dt
    for t in range(int(times//ITI)+1):
        try:
            ExcInput[0, int( t*ITI )] = True
        except:
            IndexError
    
    
    for i in range(times):
        sAMPA  *= np.exp(-dt/tau_AMPA) # synaptic activations
        sNMDA  *= np.exp(-dt/tau_NMDA) 
        sAMPA_soma  *= np.exp(-dt/tau_AMPA)

        if ExcInput[0, i] == 1 :  
            sAMPA[StiDen] += alpha * (1 - sAMPA[StiDen]) 
            sNMDA[StiDen] += alpha * (1 - sNMDA[StiDen]) 
        
        #for dendritic terms
        g_AMPA   =weight_AMPA  * sAMPA
        g_NMDA   =weight_NMDA  * sNMDA /(1 + 0.3*Mg*np.exp(0.08*((0)-V_d))) 
        g_KIR  =weight_KIR /(1 + np.exp(-0.1*((-90)-V_d-10)))                            
        
        #for external input to the soma
        g_AMPA_soma   =weight_AMPA_soma  * sAMPA_soma
        
        V_d0 = np.copy(V_d)
        V_s0 = np.copy(V_s)
        
        #dendritic voltage update
        dV_d = (
            g_AMPA*(0 - V_d)+
            g_NMDA*(0 - V_d)+
            g_KIR*((-90) - V_d)+
            g_L_d*((-80) - V_d)+
            g_StoD*(V_s0 - V_d))*dt/Cm
        V_d = V_d + dV_d
        
        #somatic voltage update
        dV_s = (
            g_AMPA_soma*(0 - V_s)+
            g_L_s*((-80) - V_s)+
            (g_DtoS*(V_d0 - V_s)).sum()
            +Tonic_s)*dt/Cm
        V_s = V_s + dV_s
        
        SpikeRecord[i]= (V_s >= VTh) & (V_s0 < VTh)  #record new spike triggered
        
        #a fired spike lasts for a while before it resets
        spikebuffer = np.sum( SpikeRecord[i-spike_duration+1:i+1]) 
        if spikebuffer > 0:
            V_s = peak 
        if SpikeRecord[i-spike_duration] & (spikebuffer < 1):
            V_s = -80  
        
        #record data
        V_s_Record[i] = V_s
        V_d_Record[:,i] = V_d
    
    InputList.append(SignalRate)
    V_dList.append(np.mean(V_d_Record[:, int(400/dt):], axis = 1))

#==============================================================================
#same idea, but with rate decrease
for step in range(Range):
    ExcInput = np.zeros( [1,times], dtype=bool) 
    SignalRate = Range-step
    ITI = 1000/SignalRate/dt
    for t in range(int(times//ITI)+1):
        try:
            ExcInput[0, int( t*ITI )] = True
        except:
            IndexError
    
    for i in range(times):
        sAMPA *= np.exp(-dt/tau_AMPA) 
        sNMDA *= np.exp(-dt/tau_NMDA) 
        sAMPA_soma *= np.exp(-dt/tau_AMPA)
        
        if ExcInput[0, i] == 1 :  
            sAMPA[StiDen] += alpha * (1 - sAMPA[StiDen])
            sNMDA[StiDen] += alpha * (1 - sNMDA[StiDen])
        
        g_AMPA   =weight_AMPA  * sAMPA
        g_NMDA   =weight_NMDA  * sNMDA /(1 + 0.3*Mg*np.exp(0.08*((0)-V_d))) 
        g_KIR  =weight_KIR /(1 + np.exp(-0.1*((-90)-V_d-10)))
        
        
        g_AMPA_soma   =weight_AMPA_soma  * sAMPA_soma
        
        V_d0 = np.copy(V_d)
        V_s0 = np.copy(V_s)
        
        
        dV_d = (
            g_AMPA*(0 - V_d)+
            g_NMDA*(0 - V_d)+
            g_KIR*((-90) - V_d)+
            g_L_d*((-80) - V_d)+
            g_StoD*(V_s0 - V_d))*dt/Cm
        V_d = V_d + dV_d
        
        dV_s = (
            g_AMPA_soma*(0 - V_s)+
            g_L_s*((-80) - V_s)+
            (g_DtoS*(V_d0 - V_s)).sum()
            +Tonic_s)*dt/Cm
        V_s = V_s + dV_s
        
        
        SpikeRecord[i]= (V_s >= VTh) & (V_s0 < VTh)
        
        spikebuffer = np.sum( SpikeRecord[i-spike_duration+1:i+1]) 
        if spikebuffer > 0:
            V_s = peak 
        if SpikeRecord[i-spike_duration] & (spikebuffer < 1):
            V_s = -80  
        
        V_s_Record[i] = V_s
        V_d_Record[:,i] = V_d
    
    InputList.append(SignalRate)
    V_dList.append(np.mean(V_d_Record[:, int(400/dt):], axis = 1))


V_dList = np.array(V_dList)
VIndenpendentSim[:, StiDen] = V_dList[:, StiDen]



plt.rcParams["figure.figsize"] = (4,8)
i=0
colors = plt.rcParams["axes.prop_cycle"]()
fig, axn = plt.subplots( NoOfDendrites, 1, sharex=True,sharey=True)
for ax in axn.flat:
    c = next(colors)["color"]
    ax.plot(InputList, VIndenpendentSim[:,NoOfDendrites -1-i], 
            color = c)
    ax.set_xlim(left=0, right=70)
    ax.set_ylim(bottom=-90, top=5)
    i = i + 1

fig.text(-0.05, 0.5, 'Dendritic voltage (mv)', va='center',  fontsize = 18,rotation='vertical')
plt.xlabel('Dendritic input frequency', fontsize=18, fontweight='light')
fig.tight_layout(pad=0)
sns.despine()
plt.rcParams["figure.figsize"] = (8,5.5)
plt.show()

stop = timeit.default_timer()
print('Time: ', stop - start)
import os
os.system('say "finished"')