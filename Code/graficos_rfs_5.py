import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#RFS = Reduce Feature Set
#CM = cascade model
#SF = Simpler features
wake_eeg =[0.85493406, 0.77858985, 0.79178338, 0.72231687, 0.85406859, 0.57079153]
stage1_eeg=[0.04819277, 0.08264463, 0.02857143, 0.0,         0.14414414, 0.02970297]
stage2_eeg=[0.62559242, 0.53900088, 0.54452638, 0.36868687, 0.59239493, 0.68842105]
stageSWS_eeg=[0.74383302, 0.61462206, 0.53864169, 0.49112426, 0.63972286, 0.64606742]
rem_eeg=[0.58002736, 0.31688312, 0.39031339, 0.06113537, 0.31020408, 0.17914439]

wake_eeg_eog = [0.96064073, 0.91730733, 0.84334764, 0.8165596,  0.96248562, 0.42539159]
stage1_eeg_eog=[0.22,       0.31034483, 0.15748031, 0.08695652, 0.12093023, 0.04060914]
stage2_eeg_eog=[0.84075724, 0.80557315, 0.70511628, 0.64769933, 0.75180092, 0.74753086]
stageSWS_eeg_eog=[0.92335116, 0.87007299, 0.80882353, 0.79249707, 0.84258211, 0.57264957]
rem_eeg_eog=[0.72256473, 0.57839721, 0.53194263, 0.16666667, 0.52464789, 0.375     ]

wake_eeg_eog_emg =[0.9594873,  0.91618226, 0.84357392, 0.8132444,  0.96175115, 0.43528442]
stage1_eeg_eog_emg=[0.22,       0.33918129, 0.1300813, 0.08695652, 0.16113744, 0.04060914]
stage2_eeg_eog_emg=[0.83824347, 0.81186869, 0.70144388, 0.64507042, 0.75555556, 0.74799754]
stageSWS_eeg_eog_emg=[0.92142857, 0.87134503, 0.81170018, 0.78463329, 0.83578709, 0.58315335]
rem_eeg_eog_emg=[0.71111111, 0.57209302, 0.54404145, 0.17269076, 0.54355401, 0.38026474]

data_wake_19= pd.DataFrame(
    {
    'EEG': [ 0.57079153],
    'EEG+EOG' : [0.42539159],
    'EEG+EOG+EMG' : [0.43528442]
    })

data_s1_19= pd.DataFrame(
    {
    'EEG': [ 0.02970297],
    'EEG+EOG' : [0.04060914],
    'EEG+EOG+EMG' : [0.04060914]
    })

data_s2_19= pd.DataFrame(
    {
    'EEG': [ 0.68842105],
    'EEG+EOG' : [0.74753086],
    'EEG+EOG+EMG' : [0.74799754]
    })

data_sws_19= pd.DataFrame(
    {
    'EEG': [ 0.64606742],
    'EEG+EOG' : [0.57264957],
    'EEG+EOG+EMG' : [0.58315335]
    })

data_rem_19= pd.DataFrame(
    {
    'EEG': [ 0.17914439],
    'EEG+EOG' : [0.375],
    'EEG+EOG+EMG' : [0.38026474]
    })

data_wake_17= pd.DataFrame(
    {
    'EEG': [ 0.72231687],
    'EEG+EOG' : [0.8165596],
    'EEG+EOG+EMG' : [0.8132444]
    })

data_s1_17= pd.DataFrame(
    {
    'EEG': [ 0.0],
    'EEG+EOG' : [0.08695652],
    'EEG+EOG+EMG' : [0.08695652]
    })

data_s2_17= pd.DataFrame(
    {
    'EEG': [ 0.36868687],
    'EEG+EOG' : [0.64769933],
    'EEG+EOG+EMG' : [0.64507042]
    })

data_sws_17= pd.DataFrame(
    {
    'EEG': [ 0.49112426],
    'EEG+EOG' : [0.79249707],
    'EEG+EOG+EMG' : [0.78463329]
    })

data_rem_17= pd.DataFrame(
    {
    'EEG': [ 0.06113537],
    'EEG+EOG' : [0.16666667],
    'EEG+EOG+EMG' : [0.17269076]
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

