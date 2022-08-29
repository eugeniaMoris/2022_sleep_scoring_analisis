import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#RFS = Reduce Feature Set
#CM = cascade model
#SF = Simpler features
wake_eeg =[0.96460761, 0.86269211, 0.86486486, 0.83711507, 0.89381663, 0.63503222]
stage1_eeg=[0.0,         0.0,         0.02020202, 0.0,         0.01098901, 0.0        ]
stage2_eeg=[0.77277599, 0.71004422, 0.69885277, 0.63774048, 0.68179363, 0.66556072]
stageSWS_eeg=[0.71827957, 0.01114206, 0.53623188, 0.28033473, 0.20458554, 0.21568627]
rem_eeg=[0.70655271, 0.51470588, 0.41721854, 0.41697417, 0.07258065, 0.44071856]

wake_eeg_eog = [0.96442688, 0.86268556, 0.86813725, 0.84369885, 0.89700855, 0.62645012]
stage1_eeg_eog=[0.0,         0.0,         0.02020202, 0.03030303, 0.0,         0.0        ]
stage2_eeg_eog=[0.76602564, 0.71671924, 0.69854665, 0.64466801, 0.67781975, 0.66622118]
stageSWS_eeg_eog=[0.66816143, 0.01675978, 0.53527981, 0.26304802, 0.20701754, 0.24761905]
rem_eeg_eog=[0.71086037, 0.48358209, 0.4148398,  0.42490842, 0.11764706, 0.44897959]

wake_eeg_eog_emg =[0.96633663, 0.86196423, 0.86161369, 0.84817607, 0.90167454, 0.63837418]
stage1_eeg_eog_emg=[0.0,         0.0,         0.02040816, 0.0,         0.01098901, 0.0        ]
stage2_eeg_eog_emg=[0.76956056, 0.71455577, 0.69135802, 0.65169457, 0.68672757, 0.67390589]
stageSWS_eeg_eog_emg=[0.69298246, 0.02777778, 0.61609195, 0.32454361, 0.25641026, 0.29493088]
rem_eeg_eog_emg=[0.72167832, 0.49552239, 0.39182283, 0.44,       0.1031746,  0.4313253]

data_wake_19= pd.DataFrame(
    {
    'EEG': [ 0.63503222],
    'EEG+EOG' : [0.62645012],
    'EEG+EOG+EMG' : [0.63837418]
    })

data_s1_19= pd.DataFrame(
    {
    'EEG': [ 0.0],
    'EEG+EOG' : [0.0],
    'EEG+EOG+EMG' : [0.0]
    })

data_s2_19= pd.DataFrame(
    {
    'EEG': [ 0.66556072],
    'EEG+EOG' : [0.66622118],
    'EEG+EOG+EMG' : [0.67390589]
    })

data_sws_19= pd.DataFrame(
    {
    'EEG': [ 0.21568627],
    'EEG+EOG' : [0.24761905],
    'EEG+EOG+EMG' : [0.29493088]
    })

data_rem_19= pd.DataFrame(
    {
    'EEG': [ 0.44071856],
    'EEG+EOG' : [0.44897959],
    'EEG+EOG+EMG' : [0.4313253]
    })

data_wake_17= pd.DataFrame(
    {
    'EEG': [ 0.83711507],
    'EEG+EOG' : [0.84369885],
    'EEG+EOG+EMG' : [0.84817607]
    })

data_s1_17= pd.DataFrame(
    {
    'EEG': [ 0.0],
    'EEG+EOG' : [0.03030303],
    'EEG+EOG+EMG' : [0.0]
    })

data_s2_17= pd.DataFrame(
    {
    'EEG': [ 0.63774048],
    'EEG+EOG' : [0.64466801],
    'EEG+EOG+EMG' : [0.65169457]
    })

data_sws_17= pd.DataFrame(
    {
    'EEG': [ 0.28033473],
    'EEG+EOG' : [0.26304802],
    'EEG+EOG+EMG' : [0.32454361]
    })

data_rem_17= pd.DataFrame(
    {
    'EEG': [ 0.41697417],
    'EEG+EOG' : [0.42490842],
    'EEG+EOG+EMG' : [0.44]
    })


data_wake= pd.DataFrame(
    {
    'EEG': wake_eeg,
    'EEG+EOG' : wake_eeg_eog,
    'EEG+EOG+EMG' : wake_eeg_eog_emg
    })

data_s1= pd.DataFrame(
    {
    'EEG': stage1_eeg,
    'EEG+EOG' : stage1_eeg_eog,
    'EEG+EOG+EMG' : stage1_eeg_eog_emg
    })

data_s2= pd.DataFrame(
    {
    'EEG': stage2_eeg,
    'EEG+EOG' : stage2_eeg_eog,
    'EEG+EOG+EMG' : stage2_eeg_eog_emg
    })

data_sws= pd.DataFrame(
    {
    'EEG': stageSWS_eeg,
    'EEG+EOG' : stageSWS_eeg_eog,
    'EEG+EOG+EMG' : stageSWS_eeg_eog_emg
    })

data_rem= pd.DataFrame(
    {
    'EEG': rem_eeg,
    'EEG+EOG' : rem_eeg_eog,
    'EEG+EOG+EMG' : rem_eeg_eog_emg
    })



sns.set_theme(style="whitegrid")
plt.ylim(-0.5, 1.5)

ax= sns.violinplot(data=data_wake,palette='BuPu_r')
ax = sns.swarmplot(data=data_wake, color="white")
ax = sns.swarmplot(data=data_wake_19, color="red")
ax = sns.swarmplot(data=data_wake_17, color="blue")
plt.title('Wake')
plt.show()

plt.ylim(-0.5, 1.5)
ax= sns.violinplot(data=data_s1,palette='YlOrBr')
ax = sns.swarmplot(data=data_s1, color="white")
ax = sns.swarmplot(data=data_s1_19, color="red")
ax = sns.swarmplot(data=data_s1_17, color="blue")

plt.title('Stage 1')
plt.show()

plt.ylim(-0.5, 1.5)
ax1= sns.violinplot(data=data_s2,palette='ch:start=2,rot=.1')
ax1 = sns.swarmplot(data=data_s2, color="white")
ax1 = sns.swarmplot(data=data_s2_19, color="red")
ax1 = sns.swarmplot(data=data_s2_17, color="blue")

plt.title('Stage 2')
plt.show()

plt.ylim(-0.5, 1.5)
ax2= sns.violinplot(data=data_sws,palette='flare')
ax2 = sns.swarmplot(data=data_sws, color="white")
ax2 = sns.swarmplot(data=data_sws_19, color="red")
ax2 = sns.swarmplot(data=data_sws_17, color="blue")

plt.title('Stage SWS')
plt.show()

plt.ylim(-0.5, 1.5)
ax2= sns.violinplot(data=data_rem,palette='ch:start=.2,rot=-.3')
ax2 = sns.swarmplot(data=data_rem, color="white")
ax2 = sns.swarmplot(data=data_rem_19, color="red")
ax2 = sns.swarmplot(data=data_rem_17, color="blue")

plt.title('REM')
plt.show()

