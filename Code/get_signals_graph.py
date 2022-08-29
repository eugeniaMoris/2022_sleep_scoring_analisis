import read_EDF_data_per_sj_5
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
import numpy as np
import pywt
import seaborn as sns
import random
from scipy.stats import norm, kurtosis, skew, entropy
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot

mapping = {'EEG Fpz-Cz': 'eeg',
           'EEG Pz-Oz': 'eeg',
           'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'emg',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

annotation_desc_2_event_id = {'Sleep stage W': 1,
                            'Sleep stage 1': 2,
                            'Sleep stage 2': 3,
                            'Sleep stage 3': 4,
                            'Sleep stage 4': 4,
                            'Sleep stage R': 5}
# create a new event_id that unifies stages 3 and 4
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5
            }

path_psg=r'/mnt/Almacenamiento/Doctorado/data_sleep-edf/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf'
path_hypno=r'/mnt/Almacenamiento/Doctorado/data_sleep-edf/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf'

##imprimir desde raw
# raw_train = mne.io.read_raw_edf(path_psg,preload=True)
# annot_train = mne.read_annotations(path_hypno)
# #name = 'SC4071E0'
# sf= raw_train.info['sfreq']

# #El hypnograma de vuelve en anotaciones de las etapas de sueño
# raw_train.set_annotations(annot_train, emit_warning=False)
# raw_train.set_channel_types(mapping)

# #Elimino el Event marker
# #raw_train.drop_channels('Event marker')

# print(raw_train.info)

# #genero los eventos a partir de las anotaciones
# events_train, _ = mne.events_from_annotations(raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

# epochs_train = mne.Epochs(raw=raw_train, events=events_train, event_id=event_id,picks=['EEG Fpz-Cz','EOG horizontal','EMG submental'], tmax=tmax, baseline=None,preload=True)

#raw_train.drop_channels(ch_names=['EEG Pz-Oz','Resp oro-nasal','Temp rectal','Event marker'])

#raw_train.plot(events=events_train)



epochs_all,train_sj = read_EDF_data_per_sj_5.get_eeg_data(0,26,0,'early')

#epochs[0].plot(picks=['EEG Fpz-Cz','EOG horizontal','EMG submental'])
#plt.show()

epochs = epochs_all[0]

w= epochs['Sleep stage W']
s1= epochs['Sleep stage 1']
s2= epochs['Sleep stage 2']
sws= epochs['Sleep stage 3/4']
r= epochs['Sleep stage R']

def get_data_to_graph(signal):
    
    data = signal.get_data(picks=['EEG Fpz-Cz','EOG horizontal','EMG submental'])
    print(data.shape)

    rand = random.randrange(1, 214, 1)
    print('epochs number: ',rand)


    arr = np.array(data[55,0,:])
    arr_eog = np.array(data[55,1,:])
    arr_emg = np.array(data[55,2,:])

    CA_EEG_4, _, _,_,_ = pywt.wavedec(arr,'dmey',level=4)
    CA_EOG_4, _, _,_,_ = pywt.wavedec(arr_eog,'dmey',level=4)
    CA_EMG_4, _, _,_,_ = pywt.wavedec(arr_emg,'dmey',level=4)

    CA_EEG_4 = CA_EEG_4[40:180]
    CA_EOG_4 = CA_EOG_4[40:180]
    CA_EMG_4 = CA_EMG_4[40:180]

    mean_eeg = np.mean(CA_EEG_4)
    median_eeg = np.median(CA_EEG_4)
    std_eeg = np.std(CA_EEG_4)
    kurt_eeg = kurtosis(CA_EEG_4,axis=None)
    sk_eeg = skew(CA_EEG_4,axis=None)

    mean_eog = np.mean(CA_EOG_4)
    median_eog = np.median(CA_EOG_4)
    std_eog = np.std(CA_EOG_4)
    kurt_eog = kurtosis(CA_EOG_4,axis=None)
    sk_eog = skew(CA_EOG_4,axis=None)

    mean_emg = np.mean(CA_EMG_4)
    median_emg = np.median(CA_EMG_4)
    std_emg = np.std(CA_EMG_4)
    kurt_emg = kurtosis(CA_EMG_4,axis=None)
    sk_emg = skew(CA_EMG_4,axis=None)

    return CA_EEG_4, mean_eeg, std_eeg, CA_EOG_4, mean_eog, std_eog, CA_EMG_4, mean_emg, std_emg



w_CA_EEG_4, w_mean_eeg, w_std_eeg, w_CA_EOG_4, w_mean_eog, w_std_eog, w_CA_EMG_4, w_mean_emg, w_std_emg = get_data_to_graph(w)
s1_CA_EEG_4, s1_mean_eeg, s1_std_eeg, s1_CA_EOG_4, s1_mean_eog, s1_std_eog, s1_CA_EMG_4, s1_mean_emg, s1_std_emg = get_data_to_graph(s1)
s2_CA_EEG_4, s2_mean_eeg, s2_std_eeg, s2_CA_EOG_4, s2_mean_eog, s2_std_eog, s2_CA_EMG_4, s2_mean_emg, s2_std_emg = get_data_to_graph(s2)
sws_CA_EEG_4, sws_mean_eeg, sws_std_eeg, sws_CA_EOG_4, sws_mean_eog, sws_std_eog, sws_CA_EMG_4, sws_mean_emg, sws_std_emg = get_data_to_graph(sws)
r_CA_EEG_4, r_mean_eeg, r_std_eeg, r_CA_EOG_4, r_mean_eog, r_std_eog, r_CA_EMG_4, r_mean_emg, r_std_emg = get_data_to_graph(r)


w=w_CA_EMG_4
s1=s1_CA_EMG_4
s2=s2_CA_EMG_4
sws=sws_CA_EMG_4
r=r_CA_EMG_4

fig = plt.figure(figsize=(25, 5))
gs=GridSpec(nrows = 3, ncols=5)

#EEG
# y_min = -0.0003
# y_max= 0.0003

# 0.0001
# 0.0002
# 0.0003

#EOG
# limite_1 = 0.0003
# limite_2 = 0.0006
# limite_3 = 0.0009

# y_min = -0.0009
# y_max= 0.0009

limite_0 = 0.000002
limite_1 = 0.000005
limite_2 = 0.000007
limite_3 = 0.000010
limite_4 = 0.000012
limite_5 = 0.000015

y_min = -0.00000
y_max= 0.000015

ax0 = fig.add_axes([0.0,0.0,0.12,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')



plt.ylim(y_min, y_max)
plt.axis('off')
sns.lineplot(data= w,color='pink')


ax1 = fig.add_axes([0.12,0.0,0.08,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')

plt.ylim(y_min, y_max)
plt.axis('off')
sns.boxplot(data= w,color='pink')

ax1 = fig.add_axes([0.20,0.0,0.12,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')

plt.ylim(y_min,y_max)
plt.axis('off')
sns.lineplot(data= s1,color='skyblue')
ax2 = fig.add_axes([0.32,0.0,0.08,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')

plt.ylim(y_min, y_max)
plt.axis('off')
sns.boxplot(data= s1,color='skyblue')

ax3 = fig.add_axes([0.40,0.0,0.12,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')

plt.ylim(y_min, y_max)
plt.axis('off')
sns.lineplot(data= s2,color='orange')
ax4 = fig.add_axes([0.52,0.0,0.08,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')

plt.ylim(y_min, y_max)
plt.axis('off')
sns.boxplot(data= s2,color='orange')

ax1 = fig.add_axes([0.60,0.0,0.12,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')

plt.ylim(y_min, y_max)
plt.axis('off')
sns.lineplot(data= sws,color='palegreen')
ax2 = fig.add_axes([0.72,0.0,0.08,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')

plt.ylim(y_min, y_max)
plt.axis('off')
sns.boxplot(data= sws,color='palegreen')

ax1 = fig.add_axes([0.80,0.0,0.12,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')

plt.ylim(y_min, y_max)
plt.axis('off')
sns.lineplot(data= r,color='tan')
ax2 = fig.add_axes([0.92,0.0,0.08,1])

plt.axhline(0.0, ls='--',color='grey')
plt.axhline(limite_0, ls='--',color='grey')
plt.axhline(limite_1, ls='--',color='grey')
plt.axhline(limite_2, ls='--',color='grey')
plt.axhline(limite_3, ls='--',color='grey')
plt.axhline(limite_4, ls='--',color='grey')
plt.axhline(limite_5, ls='--',color='grey')

plt.ylim(y_min, y_max)
plt.axis('off')
sns.boxplot(data= r,color='tan')

plt.show()