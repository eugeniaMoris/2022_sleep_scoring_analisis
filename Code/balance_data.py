#import read_EDF_data 
import numpy as np
import mne

def balance_epochs(epochs,percent=0.1 ,same=False, class_rus=3):
    labels = epochs.events[:, -1]
    print(epochs.event_id)
    classes = []
    cant = []
    for elem in epochs.event_id:
        classes.insert(len(classes),epochs[elem])
        cant.insert(len(cant),len(epochs[elem]))
    #print(classes)
    #print(cant)
    new_cant_classes = []
    if same==True:
        min_cant=np.min(cant)
        for clss in range(len(classes)):
            new_cant_classes.insert(len(new_cant_classes),(classes[clss])[:min_cant])
    else:
        max_cant=np.max(cant)
        new_max=int(percent*max_cant/100)
        print('new max: ', new_max)
        for clss in range(len(classes)):
            if (clss == class_rus):
                new_cant_classes.insert(len(new_cant_classes),(classes[clss])[:new_max])
            else:
                new_cant_classes.insert(len(new_cant_classes),(classes[clss]))

    Epoch_data =mne.concatenate_epochs(new_cant_classes) 
    #print('minimo:', Epoch_data)
    return Epoch_data
