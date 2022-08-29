import numpy as np
import mne 
import argparse
import glob
import math
import ntpath
import os
import csv
import shutil
import pandas as pd
from pathlib import Path
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
event_id = {#'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5
            }

start = [30000,25000,21000,18000,21000,22000,25000,27000,23000,31000,37000,31000,27000,24000,27000,26000,23000,25000,23000,26000,22000,25000,27000,35000,28000,32000,28000,24000,28000,26000,31000,21000,31000,28000,10000,26000,30000,34000,38000]
end = [54000,58000,53000,53000,51000,51000,51000,53000,60000,67000,57000,69000,51000,55000,55000,56000,56000,56000,57000,59000,54000,57000,54000,58000,59000,61000,58000,54000,56000,54000,61000,55000,60000,57000,61000,54000,56000,79000,75000]

sj_without_sws = [39,40,63,65,66,67,68,109,123,124,134,135,136,137,138,139,140,141,144]


def get_epochs(path_psg, path_hypno,name,pos):
    #Realizar por cada conjunto de archivos
    raw_train = mne.io.read_raw_edf(path_psg,preload=True)
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

    #PARA CUANDO SE AGREGA WAKE
    #raw_train.crop(tmin= start[pos], tmax= end[pos])

    #Genero las epocas en funcion de los eventos
    epochs_train = mne.Epochs(raw=raw_train, events=events_train, event_id=event_id,picks=['EEG Fpz-Cz','EOG horizontal'], tmax=tmax, baseline=None,preload=True)
    print('sujeto: ', name, epochs_train)
    return epochs_train

def get_name(path):
    #obtengo el numero de sujeto
    #SC4[sujeto][noche]EO, el numero de sujeto siempre son dos numeros ej 00
    archivo = path.split('SC')[-1]
    n_arch = archivo.split('-')[0]
    sujeto= n_arch[1] + n_arch[2]
    print('SUJETO. ', sujeto)
    return sujeto

def merge_data(epoch, list_epochs, index_name):
    list_aux=[]
    list_aux.insert(len(list_aux),list_epochs[index_name])
    list_aux.insert(len(list_aux),epoch)
    result = mne.concatenate_epochs(list_aux)
    return result

def get_ages(path):
    df = pd.read_csv(path)
    ages = df['age']
    return ages

def get_eeg_data(min = 0, max = 150):
    #se toma los eeg por tema de memoria

    path = Path(r'/media/euge/Almacenamiento/Doctorado/data_sleep-edf/sleep-edf-database-expanded-1.0.0/sleep-cassette')
    psg_fnames = list(path.glob('*-PSG.edf'))
    ann_fnames = list(path.glob('*-Hypnogram.edf'))

    ages = get_ages(Path(r'/media/euge/Almacenamiento/Doctorado/data_sleep-edf/sleep-edf-database-expanded-1.0.0/SC-subjects.csv'))

    psg_fnames.sort()
    ann_fnames.sort()

    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)
    edades= []

    list_epochs=[]
    subjets= []
    for i in range(len(psg_fnames)):
        if (i not in sj_without_sws) and (ages[i] <= max) and (ages[i] >= min):
            edades.insert(len(edades),ages[i])
            name = get_name(str(psg_fnames[i]))
            epoch = get_epochs(psg_fnames[i],ann_fnames[i],name,i)
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
       
    #return list_epochs, subjets, ages


def get_init(path):
    col_list = ['subject', 'night','age', 'sex (F=1)', 'LightsOff']
    df = pd.read_csv(path, usecols=col_list)
    LightsOff = df['LightsOff']
    for i in range(len(LightsOff)):
        LightsOff[i] = LightsOff[i].replace(":",".")
        print(LightsOff[i])
    return LightsOff

#result, subjets = get_eeg_data()
#print(len(result))
#print(subjets)
#print(result[20])

# csv = '/media/piddef/Almacenamiento/Materias/Computer_Vision/Computer_vision_final/SC-subjects.csv'
# eeg = '/media/piddef/Almacenamiento/Materias/Computer_Vision/Sleep-EDF/SC4192E0-PSG.edf'
# hyp = '/media/piddef/Almacenamiento/Materias/Computer_Vision/Sleep-EDF/SC4192EV-Hypnogram.edf'
# init = get_init(csv)

# print(init)

# epochs = get_epochs(eeg,hyp,'00')
# #epochs.crop(float(init[1])*60) #saca por epoca 
# print(epochs)