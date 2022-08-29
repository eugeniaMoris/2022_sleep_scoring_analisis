import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#RFS = Reduce Feature Set
#CM = cascade model
#SF = Simpler features
wr_eeg =[0.93163565, 0.93653909, 0.84183007, 0.8403935,  0.92426518, 0.50224618]
s1234_eeg=[0.86221871, 0.84590945, 0.73367572, 0.77174849, 0.87730275, 0.70247046]

wr_eeg_eog =  [0.94167963, 0.94807909, 0.85476619, 0.81833616, 0.92416869, 0.60755814]
s1234_eeg_eog= [0.87971131, 0.86603069, 0.77475593, 0.77185501, 0.85648503, 0.79155944]

wr_eeg_eog_emg = [0.95002894, 0.93623943, 0.86913801, 0.83907092, 0.93087179, 0.55996223]
s1234_eeg_eog_emg= [0.89450102, 0.84219002, 0.79351656, 0.78693026, 0.873451,  0.75678497]



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
plt.title('Wake-Rem')
plt.show()

plt.ylim(0, 1.25)
ax= sns.violinplot(data=data_s1234,palette='YlOrBr')
ax = sns.swarmplot(data=data_s1234, color="white")
plt.title('Stage 1, 2 and SWS')
plt.show()

