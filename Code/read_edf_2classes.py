import numpy as np
import mne 
import argparse
import glob
import math
import ntpath
import os
import shutil
import matplotlib.pyplot as plt

#SE GUARDAN DOS CLASES, SE ENTRENA CLASE 1 VS CLASS 2 Y 3

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
                            'Sleep stage 3': 3,
                            'Sleep stage 4': 3,
                            'Sleep stage R': 4}
# create a new event_id that unifies stages 3 and 4
event_id = {#'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2/3/4': 3,
            #'Sleep stage 3/4': 4
            #'Sleep stage R': 5
            }

def get_epochs(path_psg, path_hypno):
    #Realizar por cada conjunto de archivos
    raw_train = mne.io.read_raw_edf(path_psg)
    annot_train = mne.read_annotations(path_hypno)
    #name = 'SC4071E0'
    sf= raw_train.info['sfreq']

    #El hypnograma de vuelve en anotaciones de las etapas de sueño
    raw_train.set_annotations(annot_train, emit_warning=False)
    raw_train.set_channel_types(mapping)

    #Elimino el Event marker
    #raw_train.drop_channels('Event marker')

    print(raw_train.info)

    #genero los eventos a partir de las anotaciones
    events_train, _ = mne.events_from_annotations(raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    #Se crea las epocas en funcion de los eventos
    tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

    #Genero las epocas en funcion de los eventos
    epochs_train = mne.Epochs(raw=raw_train, events=events_train, event_id=event_id,picks='EEG Fpz-Cz', tmin=0., tmax=tmax, baseline=None)

    return epochs_train

def get_name(path):
    #obtengo el numero de sujeto
    #SC4[sujeto][noche]EO, el numero de sujeto siempre son dos numeros ej 00
    archivo = path.split('/')[-1]
    n_arch = archivo.split('-')[0]
    sujeto= n_arch[3] + n_arch[4]
    return sujeto

def merge_data(epoch, list_epochs, index_name):
    list_aux=[]
    list_aux.insert(0,list_epochs[index_name])
    list_aux.insert(0,epoch)
    result = mne.concatenate_epochs(list_aux)
    return result


def get_eeg_data():
    #se toma los eeg por tema de memoria
    psg_fnames = glob.glob(os.path.join(r'/media/piddef/Almacenamiento/Materias/Computer_Vision/Sleep-EDF', "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(r'/media/piddef/Almacenamiento/Materias/Computer_Vision/Sleep-EDF', "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)


    list_epochs=[]
    subjets= []
    for i in range(len(psg_fnames)):
        name = get_name(psg_fnames[i])
        epoch = get_epochs(psg_fnames[i],ann_fnames[i])
        #epochs.insert(len(epochs),get_epochs(psg_fnames[i],ann_fnames[i]))
        if (name in subjets):
            index_name = subjets.index(name)
            new_epoch = merge_data(epoch, list_epochs, index_name)
            list_epochs.pop(index_name)
            list_epochs.insert(index_name,new_epoch)
        else:
            list_epochs.insert(len(list_epochs),epoch)
            subjets.insert(len(subjets),name)
    
    return list_epochs, subjets

#result, subjets = get_eeg_data()
#print(len(result))
#epoca00=result[0]
#epoca00.plot()
#plt.show()