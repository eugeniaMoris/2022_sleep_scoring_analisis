import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#RFS = Reduce Feature Set
#CM = cascade model
#SF = Simpler features
wr_eeg =[0.9331819,  0.94774945, 0.86776687, 0.84786712, 0.90303272, 0.58863636]
s1234_eeg=[0.85010616, 0.85425812, 0.74970344, 0.74542009, 0.80760095, 0.67190332]

wr_eeg_eog =  [0.95359281, 0.95365892, 0.88066876, 0.86873041, 0.92929688, 0.62685432]
s1234_eeg_eog= [0.89189189, 0.86639676, 0.77439747, 0.80059524, 0.8502895,  0.70460705]

wr_eeg_eog_emg = [0.95973279, 0.95497447, 0.88865413, 0.86535433, 0.93462658, 0.64627767]
s1234_eeg_eog_emg= [0.90351267, 0.86583679, 0.79597367, 0.79787234, 0.85858162, 0.74632035]



data_wr= pd.DataFrame(
    {
    'EEG': wr_eeg,
    'EEG+EOG' : wr_eeg_eog,
    'EEG+EOG+EMG' : wr_eeg_eog_emg
    })

data_s1234= pd.DataFrame(
    {
    'EEG': s1234_eeg,
    'EEG+EOG' : s1234_eeg_eog,
    'EEG+EOG+EMG' : s1234_eeg_eog_emg
    })




sns.set_theme(style="whitegrid")
plt.ylim(0, 1.25)

ax= sns.violinplot(data=data_wr,palette='BuPu_r')
ax = sns.swarmplot(data=data_wr, color="white")
plt.title('Wake-Rem-S1')
plt.show()

plt.ylim(0, 1.25)
ax= sns.violinplot(data=data_s1234,palette='YlOrBr')
ax = sns.swarmplot(data=data_s1234, color="white")
plt.title('Stage 2 and SWS')
plt.show()

