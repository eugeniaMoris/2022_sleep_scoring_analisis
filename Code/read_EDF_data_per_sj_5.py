from mne import epochs
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
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5
            }


sj_without_sws = [39,40,63,65,66,67,68,109,123,124,134,135,136,137,138,139,140,141,144]

def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1

def crop_epochs(epochs,earlyLate,half):
    labels = epochs.events[:, -1]
    #print(labels)
    labels=list(labels)

    w= labels.index(1)
    s1 = labels.index(2)
    s2 = labels.index(3)
    sws = labels.index(4)
    r = labels.index(5)

    w_last = listRightIndex(labels,1)
    s1_last = listRightIndex(labels,2)
    s2_last = listRightIndex(labels,3)
    sws_last = listRightIndex(labels,4)
    r_last = listRightIndex(labels,5)

    time_a = (min(s1,s2,sws,r) - 40) #20 min
    time_b = (max(s1_last,s2_last,sws_last,r_last) + 40) #20 min
    #print('most right: ' , max(s1_last,s2_last,sws_last,r_last),s1_last,s2_last,sws_last,r_last)
    #print(epochs)
    

    if earlyLate == 1:
        cut = int((time_b - time_a) / 2)
        #print('Tima a: ', time_a, ' Time b: ', time_b, 'Cut ', cut)
        if half == 'early':
            mx= time_a + cut
            epochs = epochs[time_a:mx]
            #print('entro: EARLY')
        else:
            print('eNTRO EN ELSE')
            mn= time_b - cut
            epochs = epochs[mn:time_b]
            #print('entro: LATE')

    else:
        epochs = epochs[time_a:time_b]

    #print(epochs)
    return(epochs)



def get_epochs(path_psg, path_hypno,name,pos,earlyLate,half):
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
    # plot events
   
    #Se crea las epocas en funcion de los eventos
    tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

    #PARA CUANDO SE AGREGA WAKE
    #raw_train.crop(tmin= start[pos], tmax= end[pos])

    #Genero las epocas en funcion de los eventos
    epochs_train = mne.Epochs(raw=raw_train, events=events_train, event_id=event_id,picks=['EEG Fpz-Cz','EOG horizontal','EMG submental'], tmax=tmax, baseline=None,preload=True)
    print('sujeto: ', name, epochs_train)
   

    epochs_train = crop_epochs(epochs_train,earlyLate=earlyLate, half=half)

    # fig = mne.viz.plot_events(events_train, event_id=event_id,
    #                       sfreq=raw_train.info['sfreq'],
    #                       first_samp=events_train[0, 0])
    # plt.show()
    return epochs_train

def get_name(path):
    #obtengo el numero de sujeto
    #SC4[sujeto][noche]EO, el numero de sujeto siempre son dos numeros ej 00
    archivo = path.split('SC')[-1]
    n_arch = archivo.split('-')[0]
    sujeto= n_arch[1] + n_arch[2]
    #print('SUJETO. ', sujeto)
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

def get_eeg_data(min, max, earlyLate, half):
    #se toma los eeg por tema de memoria

    path = Path(r'/mnt/Almacenamiento/Doctorado/data_sleep-edf/sleep-edf-database-expanded-1.0.0/sleep-cassette')

    #path = Path(r'/home/emoris/wavelets/sleep-cassette')
    psg_fnames = list(path.glob('*-PSG.edf'))
    ann_fnames = list(path.glob('*-Hypnogram.edf'))

    ages = get_ages(Path(r'/mnt/Almacenamiento/Doctorado/data_sleep-edf/sleep-edf-database-expanded-1.0.0/SC-subjects.csv'))
    #ages = get_ages(Path(r'/home/emoris/wavelets/SC-subjects.csv'))


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
            epoch = get_epochs(psg_fnames[i],ann_fnames[i],name,i,earlyLate,half)
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

#result, subjets = get_eeg_data(0,25,1,'late')

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