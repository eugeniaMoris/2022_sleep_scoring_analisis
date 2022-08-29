import mne
import matplotlib.pyplot as plt
import numpy as np

path_S1 = '/media/piddef/Almacenamiento/data_sleep-edf/Stage_1(W)'
path_S2 = '/media/piddef/Almacenamiento/data_sleep-edf/Stage_2'
path_S3 = '/media/piddef/Almacenamiento/data_sleep-edf/Stage_3'
path_S4 = '/media/piddef/Almacenamiento/data_sleep-edf/Stage_4'
path_S5 = '/media/piddef/Almacenamiento/data_sleep-edf/Stage_5(R)'


mapping = {'EEG Fpz-Cz': 'eeg',
           'EEG Pz-Oz': 'eeg',
           'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'emg',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

raw_train = mne.io.read_raw_edf('/media/piddef/Almacenamiento/data_sleep-edf/SC4071E0-PSG.edf')
annot_train = mne.read_annotations('/media/piddef/Almacenamiento/data_sleep-edf/SC4071EC-Hypnogram.edf')
name = 'SC4071E0'

def save_img(data,state, channel, sf,max_cant=1000):
    
    dat_crudo = data.get_data(picks=[channel])
    minimo = min(max_cant, dat_crudo.shape[0])
    print('minimo: ', minimo)

    for i in range(minimo):
        subdata = dat_crudo[i,:,:]
        subdata.reshape(-1)
        #spectograma con un segundo de movimiento
        powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(subdata[0], Fs=sf)
        plt.axis('off')
        if (state== 1):
            plt.savefig(path_S1 + '/' + name + '_' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        elif (state == 2):
            plt.savefig(path_S2 + '/' + name + '_' + str(i) +'.png', bbox_inches='tight', pad_inches=0)
        elif (state== 3):
            plt.savefig(path_S3 + '/' + name + '_' + str(i) +'.png', bbox_inches='tight', pad_inches=0)
        elif (state== 4):
            plt.savefig(path_S4 + '/' + name + '_' + str(i) +'.png', bbox_inches='tight', pad_inches=0)
        elif (state== 5):
            plt.savefig(path_S5 + '/' + name + '_' + str(i) +'.png', bbox_inches='tight', pad_inches=0)
        print('imagen numero: ', i, 'label: ',state)

#el hypnograma de vuelve en anotaciones de las etapas de sueño
raw_train.set_annotations(annot_train, emit_warning=False)
raw_train.set_channel_types(mapping)

# plot some data
#raw_train.plot(duration=60, scalings='auto')


print(raw_train.info)
#data= raw_train.get

#De las anotaciones de etapa de sueño se generan los eventos de las anotaciones que queremos, se ignora el ?, 
#tambien se junta la etapa 3 y 4 
annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

# create a new event_id that unifies stages 3 and 4
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

# plot events
#mne.viz.plot_events(events_train, event_id=event_id,sfreq=raw_train.info['sfreq'])

# keep the color-code for further plotting
#stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#Se crea las epocas en funcion de los eventos
tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                          event_id=event_id, tmin=0., tmax=tmax, baseline=None)

print(epochs_train)
#y_train = epochs_train.events[:, 2]
#print(y_train)

S1_epochs = epochs_train['Sleep stage W']
S2_epochs = epochs_train['Sleep stage 1']
S3_epochs = epochs_train['Sleep stage 2']
S4_epochs = epochs_train['Sleep stage 3/4']
S5_epochs = epochs_train['Sleep stage R']
sf= raw_train.info['sfreq']

save_img(S1_epochs, 1,'EEG Fpz-Cz',sf,300)
save_img(S2_epochs, 2,'EEG Fpz-Cz',sf,300)
save_img(S3_epochs, 3,'EEG Fpz-Cz',sf,300)
save_img(S4_epochs, 4,'EEG Fpz-Cz',sf,300)
save_img(S5_epochs, 5,'EEG Fpz-Cz',sf,300)




