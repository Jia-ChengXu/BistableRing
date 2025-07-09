import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=2)
sns.set_style("ticks")

Tu = 9
Td = 2
label = np.arange(0, 60.0, 0.001)

for it in range(3):
    if it == 0: #for Fig3C
        w = 4/(( label  + 1)**1)
    if it == 1: #for Fig3D
        w = 30/(( label  + 9)**1.4)
    if it == 2: #for Fig3E
        w = 1*np.exp(-label**2/400 )
    
    Input = np.zeros( 1000 )
    Memory = np.zeros( 1000 )
    
    
    TuEff = Tu/w #effective threshold after considering weight
    TdEff = Td/w
    
    beta=1.
    
    Encode = False #encoding not happening
    for i in range(1000):
        I = i*0.04 #Input increases gradually
        Input[i]=I
        
        #find the intersection of input line and Tu curve. Basically solve the stable point by geometry
        idx = np.argwhere(np.diff(np.sign(label*beta + I - TuEff))).flatten()
        if len(idx) == 1: #the first time intersection # is 1 means encoding happens, with I just large enough to activate the first dendrite
            Encode = True #all following input gives memory as well, so encoding happens from now on
        
        if Encode == True: 
            if len(idx) >0: #if len(idx) = 0, all dendrites already fired, such that no stable point found
                idx = idx[0]#ideally, once encoding happens, intersection number len(idx) is always 1, but due to discreteness, it sometimes goes to 2. So only take the first value
                if label[idx]*beta >= TdEff[idx]: #recurrent activity is more than down threshold, sustainable
                    Memory[i]=label[idx]
                else: #if forgetting happens, memory decays to a value that equals the previous memory 
                    Memory[i]=Memory[i-1]
    
    #each multistable band
    plt.figure(it)
    plt.plot(TuEff, label*beta, linewidth = 5, color = 'black') #label*beta makes y dendritic rate
    plt.plot(TdEff, label*beta, linewidth = 5, color = 'black')
    axes = plt.gca()
    axes.set_xlim([0,50])
    axes.set_ylim([0,35])
    axes.set_aspect('equal')
    sns.despine()
    
    #input memory curve
    colorlist = ['blue', 'magenta', 'purple']
    plt.figure(100)
    plt.plot(Input, Memory*beta, linewidth = 3, color = colorlist[it])
    axes = plt.gca()
    axes.set_xlim([0,I])
    axes.set_ylim([0,51])
    sns.despine()

plt.show()