#Clasificacion de estapas de sueño utilizando wivelets y el dataset de sleep-edf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import tSNE
import metrics
import read_EDF_data_per_sj_5
import balance_data
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pywt
import pandas as pd
import mne
import random
import seaborn as sns
from pathlib import Path
from scipy.stats import norm, kurtosis, skew, entropy
from scipy.signal import welch
import scipy as sp
import collections
import copy
import joblib

def get_new_data(epochs):
    #tener en cuenta que no van a tener el mismo orden
    #for e in epochs:
    aux = []
    #MODELO 1
    print(epochs)
    w= epochs['Sleep stage W']
    s1= epochs['Sleep stage 1']
    s2= epochs['Sleep stage 2']
    sws= epochs['Sleep stage 3/4']
    r= epochs['Sleep stage R']


    aux.insert(len(aux),w)
    aux.insert(len(aux),s1)
    aux.insert(len(aux),r)
    #print('tipos ', type(s1))
    return mne.concatenate_epochs(aux)

def wavelet_level(señal, level,wavelet, valid, medias, stds):
    train_matrix= []
    level = int(level)
    for e in señal:
        if level == 1:
            CA_EEG, _ = pywt.wavedec(e[0,:],wavelet,level=1)
            CA_EOG, _ = pywt.wavedec(e[1,:],wavelet,level=1)
            CA_EMG, _ = pywt.wavedec(e[2,:],wavelet,level=1)
        if level == 2:
            CA_EEG, _, _ = pywt.wavedec(e[0,:],wavelet,level=2)
            CA_EOG, _, _ = pywt.wavedec(e[1,:],wavelet,level=2)
            CA_EMG, _, _ = pywt.wavedec(e[2,:],wavelet,level=2)
        if level == 3:
            CA_EEG, _, _, _ = pywt.wavedec(e[0:],wavelet,level=3)
            CA_EOG, _, _, _ = pywt.wavedec(e[1:],wavelet,level=3)
            CA_EMG, _, _, _ = pywt.wavedec(e[2:],wavelet,level=3)
        if level == 4:
            CA_EEG, _, _, _, _ = pywt.wavedec(e[0,:],wavelet,level=4)
            CA_EOG, _, _, _, _ = pywt.wavedec(e[1,:],wavelet,level=4)
            CA_EMG, _, _, _, _ = pywt.wavedec(e[2,:],wavelet,level=4)
        if level == 5:
            CA_EEG, _, _, _, _, _ = pywt.wavedec(e[0,:],wavelet,level=5)
            CA_EOG, _, _, _, _, _ = pywt.wavedec(e[1,:],wavelet,level=5)
            CA_EMG, _, _, _, _, _ = pywt.wavedec(e[2,:],wavelet,level=5)
    
        mean_eeg = np.mean(CA_EEG)
        median_eeg = np.median(CA_EEG)
        std_eeg = np.std(CA_EEG)
        kurt_eeg = kurtosis(CA_EEG,axis=None)
        sk_eeg = skew(CA_EEG,axis=None)

        mean_eog = np.mean(CA_EOG)
        median_eog = np.median(CA_EOG)
        std_eog = np.std(CA_EOG)
        kurt_eog = kurtosis(CA_EOG,axis=None)
        sk_eog = skew(CA_EOG,axis=None)

        mean_emg = np.mean(CA_EMG)
        median_emg = np.median(CA_EMG)
        std_emg = np.std(CA_EMG)
        kurt_emg = kurtosis(CA_EMG,axis=None)
        sk_emg = skew(CA_EMG,axis=None)

        Features = np.array([mean_eeg,median_eeg,std_eeg,kurt_eeg,sk_eeg,mean_eog,median_eog,std_eog,kurt_eog,sk_eog,mean_emg,median_emg,std_emg,kurt_emg,sk_emg])
        #Features = np.array([mean_eeg,median_eeg,std_eeg,kurt_eeg,sk_eeg])
        #Features = np.array([mean_eeg,median_eeg,std_eeg,kurt_eeg,sk_eeg,mean_eog,median_eog,std_eog,kurt_eog,sk_eog])



        if valid==0:
            train_matrix.append(Features)
        else:
            for i in range(len(Features)):
                Features[i]= (Features[i]-medias[i])/stds[i]
            train_matrix.insert(len(train_matrix), Features)


    return train_matrix

def normalize_mat(mat):
    print('ENTRO A NORM')
    row,col= mat.shape
    media = np.empty(col)
    std = np.empty(col)
    new_mat = np.empty((row,col))
    #print('columna: ', col)
    for i in range(col):
        media[i] = np.mean(mat[:,i])
        std[i] = np.std(mat[:,i])

        for j in range(row):
            #print(mat[j,i])
            aux = (mat[j,i] - float(media[i]))/float(std[i])
            new_mat[j,i]=aux
    return new_mat, media, std

def train_data(wavelet, rus, level, features, trees, epochs, index):
    
    valid = epochs[index]
    print('SUJETO: ', index)
    epochs.pop(index)

    all_epochs = mne.concatenate_epochs(epochs)
    all_epochs = balance_data.balance_epochs(all_epochs,rus,class_rus=1)

    #me quedo solo con la clases 
    epochs= get_new_data(all_epochs)
    labels = epochs.events[:, -1]

    #WAVELET ENTRENAMIENTO
    train_matrix= wavelet_level(epochs,level,wavelet,0,0,0)
    train_matrix = np.array(train_matrix)

    #ESTANDARIZACION ENTRENAMIEO
    train_norm, medias, stds = normalize_mat(train_matrix)
    
    np.save('medias_ss1.npy', medias)
    np.save('std_ss1.npy', stds)

    #PCA
    if features != -1:
        pca = PCA(n_components=features)
        train_norm = pca.fit_transform(train_norm)

    
    #RANDOM FOREST
    clf1 = RandomForestClassifier(n_estimators=trees ,random_state=random.seed(1234))
    clf1.fit(train_norm,labels)

    #WAVELET VALIDACION
    valid=get_new_data(valid)
    valid_wavelet = wavelet_level(valid,level,wavelet,1,medias,stds)
    y_valid = valid.events[:,-1]
      
    valid_matrix = np.array(valid_wavelet)
    
    if features != -1:
        valid_matrix=pca.transform(valid_matrix)
    
    y_predict = clf1.predict(valid_matrix)

    acc = (accuracy_score(y_valid,y_predict))
    f1 = (f1_score(y_valid,y_predict,average='macro'))

    print('VALID: ', collections.Counter(y_valid))
    print('PRED: ', collections.Counter(y_predict))

    
    W, S1, R = multilabel_confusion_matrix(y_valid, y_predict)

    acc_w, f1_w = metrics.get_metrics(W,1)
    acc_s1, f1_s1 = metrics.get_metrics(S1,1)
    acc_r, f1_r = metrics.get_metrics(R,1)

    
    f = open("Acc_W-S1-R.txt", "w")
    f.write('Data: '+ '\n' + 
        'acc: '+ str(np.mean(acc)) + '\n' +
        'acc_w '+ str(np.mean(acc_w))+ '\n' +
        'acc_s1 '+ str(np.mean(acc_s1)) + '\n' +
        'acc_r '+ str(np.mean(acc_r)))
    f.close()


    #print('acc: ', acc, ' f1: ', f1, ' pres: ', pres, ' rec: ', rec)
    return acc, acc_w, acc_s1, acc_r, f1, f1_w, f1_s1, f1_r

def save_results(f1, f1_w, f1_s1, f1_r, w, r, l, f,t):
    
    df = pd.read_csv("results_cascade_W-R-S1.csv")
    f1= np.mean(f1)

    f1_w = np.mean(f1_w)
    f1_s1= np.mean(f1_s1)
    f1_r= np.mean(f1_r)

    #print(' RESULTADOS OBTENIDOS A GAURDAR ',f1, f1_s1, f1_s2, f1_sws )

    df[w + '_' + str(r) + '_' + str(l) + '_' + str(f)+'_' + str(t)] = [f1, f1_w, f1_s1,f1_r]

    df.to_csv("results_cascade_W-R-S1.csv", index=False)

def train(wavelet, rus, level, features,trees, epochs, test_epochs):
     

    acc = np.empty(len(test_epochs))
    f1 = np.empty(len(test_epochs))

    acc_w = np.empty(len(test_epochs))
    f1_w = np.empty(len(test_epochs))

    acc_s1 = np.empty(len(test_epochs))
    f1_s1 = np.empty(len(test_epochs))

    acc_r = np.empty(len(test_epochs))
    f1_r = np.empty(len(test_epochs))


    epochs = mne.concatenate_epochs(epochs)
    all_epochs = balance_data.balance_epochs(epochs,rus)
    epochs = get_new_data(all_epochs)
    labels = epochs.events[:, -1]    

    #WAVELET ENTRENAMIENTO
    train_matrix= wavelet_level(epochs,level,wavelet,0,0,0)
    train_matrix = np.array(train_matrix)

    #ESTANDARIZACION ENTRENAMIEO
    train_norm, medias, stds = normalize_mat(train_matrix)
    
    np.save('medias_2.npy', medias)
    np.save('std_2.npy', stds)


    #PCA
    pca = PCA(n_components=features)
    if features != -1:
        print('ENTRA A PCA')
        train_norm = pca.fit_transform(train_norm)

    #RANDOM FOREST
    clf1 = RandomForestClassifier(n_estimators=trees ,random_state=random.seed(1234))
    clf1.fit(train_norm,labels)

    #guardo el modelo
    joblib.dump(clf1, "./cascade_2.joblib")



    index = 0
    for t in test_epochs:
        (acc[index], acc_w[index], acc_s1[index], acc_r[index],
        f1[index], f1_w[index], f1_s1[index], f1_r) = predecir_sj(wavelet,rus, level, features, clf1,t, medias,stds,pca)

        index = index +1

    f = open("Acc_cascade_2.txt", "w")
    f.write('Data: '+ 'acc: '+ str(np.mean(acc)) +'acc_w '+ str(np.mean(acc_w))+'acc_s1 '+ str(np.mean(acc_s1))+'acc_r '+ str(np.mean(acc_r)))
    f.close()

    file = open("Fscore_cascade_2.txt", "w")
    file.write('Data: '+ 'fs_w '+ str(f1_w)+'\n'
    + 'fs_s1 '+ str(f1_s1)+'\n'
    + 'fs_r '+ str(f1_r)+'\n')
    file.close()

    save_results(f1, f1_w, f1_s1, f1_r, wavelet, rus, level, features, trees)
  
def predecir_sj(wavelet, rus, level, features, rf, test, medias, stds,pca=10):
    
    test=get_new_data(test)

    test_wavelet = wavelet_level(test,level,wavelet,1,medias,stds)
    y_test = test.events[:,-1]

    test_matrix = np.array(test_wavelet)
    
    if features != -1:
        test_matrix=pca.transform(test_matrix)
    
    y_predict = rf.predict(test_matrix)


    acc = (accuracy_score(y_test,y_predict))
    f1 = (f1_score(y_test,y_predict,average='macro'))
    
    #5 CLASES
    W, S1, R = multilabel_confusion_matrix(y_test, y_predict)

    acc_w, f1_w = metrics.get_metrics(W,1)
    acc_s1, f1_s1 = metrics.get_metrics(S1,1)
    acc_r, f1_r = metrics.get_metrics(R,1)


    #plot_confusion_matrix(rf, test_matrix, y_test)  # doctest: +SKIP
    #plt.show()  # doctest: +SKIP

   
    return acc, acc_w, acc_s1, acc_r, f1, f1_w, f1_s1, f1_r
    #return acc, acc_s1, acc_s2, acc_sws, f1, f1_s1, f1_s2, f1_sws

def main(wavelet,RUS,level,n_features,n_trees):

    epochs,train_sj = read_EDF_data_per_sj_5.get_eeg_data(0,40,1,'late')

    cant_train = 70 * len(epochs) / 100
    cant_train= int(cant_train)
    train_epochs = epochs[:cant_train]
    test_epochs = epochs[cant_train:]

    #PARA 3 CLASES
    data= {'fscore': ['General', 'Wake','Stage 1', 'REM']}

    df = pd.DataFrame(data)
    df.to_csv('results_cascade_W-R-S1.csv')

    acc = np.empty(len(train_epochs))
    f1 = np.empty(len(train_epochs))

    acc_w = np.empty(len(train_epochs))
    f1_w = np.empty(len(train_epochs))

    acc_s1 = np.empty(len(train_epochs))
    f1_s1 = np.empty(len(train_epochs))

    acc_r = np.empty(len(train_epochs))
    f1_r = np.empty(len(train_epochs))

    for w in wavelet:
        for r in RUS:
            for l in level:
                for f in n_features:
                    for t in n_trees:
                    #preparamos archivo csv para almacenar datos
                        train(w, r, l, f,t, train_epochs.copy(),test_epochs)
                        
                        

                        # for i in range(len(train_epochs)):
                        #     (acc[i], acc_w[i], acc_s1[i], acc_r[i],
                        #     f1[i], f1_w[i], f1_s1[i], f1_r[i]) = train_data(w, r, l, f, t, copy.deepcopy(train_epochs),i)             
                        # save_results(f1,f1_w,f1_s1, f1_r, w,r,l,f,t)
    #print('cantidad de sujetos: ', len(train_sj))


if __name__ == '__main__':
    #se agregan todos los parametros que pueden pasarse al software cuando se llama
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--wavelet', required=True, nargs='+', type= str, default = 'dmey',help='Lista de las familias de wavelet a utilizar')
    parser.add_argument('-r', '--RUS', required = True, nargs='+',  type= int, default= 30, help='Porcentaje de reduccion Stage 2')
    parser.add_argument('-l', '--level', required= True, nargs='+',  type= int, default= 1, help='Nivel de profundidad de las wavelet')
    parser.add_argument('-f', '--n_features', required= True, nargs='+', type= int, default= 250, help='Cantidad de features reducidas por PCA')
    parser.add_argument('-t', '--n_trees', required= True, nargs='+', type= int, default= 200, help='Cantidad de features reducidas por PCA')
    
    args = parser.parse_args()

#    os.makedirs('./output/checkpoints/', exist_ok=True)
#
    main(**vars(args))





