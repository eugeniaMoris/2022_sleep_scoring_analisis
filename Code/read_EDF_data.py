import numpy as np
import mne 
import argparse
import glob
import math
import ntpath
import os
import shutil
import scipy.signal as ssignal
import matplotlib.pyplot as plt


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

def get_epochs(path_psg, path_hypno):
    #Realizar por cada conjunto de archivos
    raw_train = mne.io.read_raw_edf(path_psg,preload=True)
    annot_train = mne.read_annotations(path_hypno)
    raw_train.set_channel_types(mapping)
    sf= raw_train.info['sfreq'] #sfreq=100

    #elimino los primeros y los ultimos 30s
    raw_train.crop(tmin=30*60,tmax=raw_train.times.max()-(300*60))



    #raw_train= raw_train.pick_channels(['EEG Fpz-Cz'])
    #picks_eeg = pick_types(raw_train.info, meg=False, eeg=True)
    #raw_filter=raw_train[picks_eeg]
    
    lowcut = 0.5
    highcut = 25.
    nyquist_freq = 100 / 2.
    high = highcut / nyquist_freq
    low = lowcut / nyquist_freq
    b, a = ssignal.butter(3, [low, high], btype='band')
    rawsignal = ssignal.filtfilt(b, a, raw_train.get_data())

    # Initialize an info structure     
    info = raw_train.info 
    raw_train = mne.io.RawArray(rawsignal, info) 


    #El hypnograma de vuelve en anotaciones de las etapas de sueño
    raw_train.set_annotations(annot_train, emit_warning=False)
    raw_train.set_channel_types(mapping)

    #Elimino el Event marker
    #raw_train.drop_channels('Event marker')

    #print(raw_train.info)

    #genero los eventos a partir de las anotaciones
    events_train, _ = mne.events_from_annotations(raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    #Se crea las epocas en funcion de los eventos
    tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

    #Genero las epocas en funcion de los eventos
    epochs_train = mne.Epochs(raw=raw_train, events=events_train, event_id=event_id,picks='eeg', tmin=0., tmax=tmax, baseline=None)

    #APLICAR FILTTRO
    #lowcut = 0.5
    #highcut = 25.
    #nyquist_freq = 100 / 2.
    #low = lowcut / nyquist_freq
    #high = highcut / nyquist_freq
    #print(lowcut, highcut, low, high)
    #iir_params = dict(order=3, ftype='butter', output='sos') 
    #iir_params = mne.filter.construct_iir_filter(iir_params,f_pass=[low,high],sfreq=sf, btype='bandpass', return_copy=False) #40? 1000?
    #epochs_train.filter(low,high,iir_params=iir_params)

    sw= epochs_train['Sleep stage W']
    
    print(sw)
    return epochs_train

def merge_data():
    #se toma los eeg por tema de memoria
    psg_fnames = glob.glob(os.path.join(r'/media/piddef/Almacenamiento/Materias/Computer_Vision/Sleep-EDF', "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(r'/media/piddef/Almacenamiento/Materias/Computer_Vision/Sleep-EDF', "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)
    #print(psg_fnames)
    #print(ann_fnames)

    epochs=[]

    for i in range(len(psg_fnames)):
        epochs.insert(len(epochs),get_epochs(psg_fnames[i],ann_fnames[i]))

    #print('cant epochs:', len(epochs))
    Epoch_data =mne.concatenate_epochs(epochs) 
    return Epoch_data

psg = '/media/piddef/Almacenamiento/Materias/Computer_Vision/Sleep-EDF/SC4001E0-PSG.edf'
ann = '/media/piddef/Almacenamiento/Materias/Computer_Vision/Sleep-EDF/SC4001EC-Hypnogram.edf'

#epochs = get_epochs(psg,ann)
#print(epochs)
#epochs.plot()
#plt.show()

