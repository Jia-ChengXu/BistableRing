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
sns.set_style("ticks")

tau_AMPA	=2		# time constant of AMPA channel closing in ms
tau_NMDA	=100	#	"	of NMDA				"			"
alpha	    =0.5	# ratio of unactivated receptors activated by spike
Cm		    = 10    #Membrane Capacitance 		in nF/mm2
Mg	        = 0.5	#[Mg2+] in mM

dt          = 0.01  #time in ms

VTh         = -50   #firing threshold of the soma
spike_duration = int(3/dt) #spike duration
peak = 30           #peak of spike

times = int(1200/dt)#each session
EncodingTime = int(400/dt) #encoding time in session
NoOfDendrites = 10

def MultiDen(x):
    g_L_s       =x[0]    #Leak Conductance of soma			in uS/mm2	
    g_L_d       =x[1]    #Leak Conductance of dendrite
    g_DtoS      =x[2]    #Soma-Dendrite Coupling Conductance 
    g_StoD      =x[3]    #Dendrite-Soma Coupling Conductance
    Tonic_s     =x[4]    
    
    weight_AMPA  = np.zeros( NoOfDendrites ) + x[6] 
    
    #assign NMDA conductance
    weight_NMDA  = np.zeros( NoOfDendrites )
    for i in range(NoOfDendrites):
        weight_NMDA[i] = x[7]/((i+x[8])**x[9])
    
    weight_KIR = x[11] 
    weight_AMPA_soma = x[12]
    
    #initialize
    sAMPA  =np.zeros( NoOfDendrites )
    sNMDA  =np.zeros( NoOfDendrites )
    sAMPA_soma =0
    V_d    = np.zeros( NoOfDendrites ) - 70
    V_s    = -70
    
    TotalTime = times*(NoOfDendrites+1)
    SpikeRecord = np.zeros( TotalTime , dtype=bool) 
    V_d_Record = np.zeros( [NoOfDendrites, TotalTime] ) 
    V_s_Record = np.zeros( TotalTime ) 
    ExcInput = np.zeros( [1,TotalTime], dtype=bool) 
    raterecord= np.zeros( TotalTime) 
    
    #generate external input signal
    for inputstep in range(NoOfDendrites+1): 
        if inputstep > 0:
            SignalRate = x[13] + x[14]*(inputstep-1)
            raterecord[inputstep*times:inputstep*times+ EncodingTime ] = SignalRate
            ITI = 1000/SignalRate/dt
            for t in range(int(EncodingTime//ITI)+1):
                try:
                    ExcInput[0, int( t*ITI ) + (inputstep) * times] = True
                except:
                    IndexError
    
    
    for i in range(TotalTime):
        sAMPA  *= np.exp(-dt/tau_AMPA) # synaptic activations
        sNMDA  *= np.exp(-dt/tau_NMDA) 
        sAMPA_soma  *= np.exp(-dt/tau_AMPA)
        
        if SpikeRecord[i-1] == 1 :  #last time spike is recevied by self connection
            sAMPA += alpha * (1 - sAMPA) 
            sNMDA += alpha * (1 - sNMDA) 
        if ExcInput[0,i] == 1: 
            sAMPA_soma += alpha * (1 - sAMPA_soma) 
        
        #for dendritic terms
        g_AMPA   =weight_AMPA  * sAMPA
        g_NMDA   =weight_NMDA  * sNMDA /(1 + 0.3*Mg*np.exp(0.08*((0)-V_d))) 
        g_KIR    =weight_KIR  /(1 + np.exp(-0.1*((-90)-V_d-10)))
        
        #for external input to the soma
        g_AMPA_soma   =weight_AMPA_soma * sAMPA_soma
        
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
        
        #record
        V_s_Record[i] = V_s
        for DenI in range(NoOfDendrites):
            V_d_Record[NoOfDendrites-DenI-1,i] = V_d[DenI]  #this just reverse the recording of V_d_Record[i] 
            #it is used to color match with Fig. 2B
    
    return V_s_Record, V_d_Record, SpikeRecord, ExcInput, raterecord



#==================================================================
#==================================================================
x = [0.5, 0.5, 0.007, 0.014, 17.2, 0., 0.38, 48,    5.5, 0.54,  1., 8.6,  5.2, 20,  5, 0]
#x set is obtained by nlopt optimization.

V_s_Record, V_d_Record, SpikeRecord, ExcInput, raterecord = MultiDen(x)


plt.rcParams["figure.figsize"] = (8,5)

fig, (a0, a1, a2) = plt.subplots(3, 1, sharex=True, height_ratios=[1, 3, 1])

a0.plot(np.arange(0, times*NoOfDendrites)*dt, raterecord[times:], 'black', linewidth = 2)
a0.set_xlim(left=0, right=times*NoOfDendrites*dt)
a0.set_ylim(bottom=0, top=70)

a1.plot(np.arange(0, times*NoOfDendrites)*dt, np.transpose(V_d_Record[:,times:]), linewidth = 2)
a1.set_xlim(left=0, right=times*NoOfDendrites*dt)
a1.set_ylim(bottom=-85, top=0)

a2.plot(np.arange(0, times*NoOfDendrites)*dt, V_s_Record[times:],'black', linewidth = 1, alpha = 0.6 )
a2.set_xlim(left=0, right=times*NoOfDendrites*dt)
a2.set_ylim(bottom=-85, top=45)

'''
fig.text(0.48, -0.006, 'Time (ms)',  fontsize = 24,va='center' )
fig.text(-0.02, 0.12, 'Somatic \n voltage (mv)', va='center',  fontsize = 24,rotation='vertical')
fig.text(-0.01, 0.5, 'Dendritic voltage (mv)', va='center',  fontsize = 24,rotation='vertical')
fig.text(-0.01, 0.9, 'Input (spikes/s)', va='center',  fontsize = 24,rotation='vertical')
'''

fig.tight_layout()
sns.despine()
plt.show()
plt.rcParams["figure.figsize"] = (8,5.5)
