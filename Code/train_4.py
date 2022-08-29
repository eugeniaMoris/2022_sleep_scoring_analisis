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
import read_EDF_data_per_sj_4
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


def plot_wavelet(mat, labels):
    labels= list(labels)
    
    w = labels.index(1)
    s1 = labels.index(2)
    s2 = labels.index(3)
    sws = labels.index(4)
    r = labels.index(5)

    plt.figure(1)
    plt.subplot(511)
    plt.plot(mat[w,:])
    plt.subplot(512)
    plt.plot(mat[s1,:])
    plt.subplot(513)
    plt.plot(mat[s2,:])
    plt.subplot(514)
    plt.plot(mat[sws,:])
    plt.subplot(515)
    plt.plot(mat[r,:])

    plt.show()
    
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
    #print('SUJETO: ', subjects[index])
    epochs.pop(index)

    epochs = mne.concatenate_epochs(epochs)

    #BALANCEO DE CLASES - SE REDUCE CLASE 2
    epochs = balance_data.balance_epochs(epochs,rus)
    labels = epochs.events[:, -1]

    #print('epochs: ', epochs)
    pos=0
    train_matrix= []
    for e in epochs:
        if level == 1:
            CA_EEG, _ = pywt.wavedec(e[0,:],wavelet,level=1)
            CA_EOG, _ = pywt.wavedec(e[1,:],wavelet,level=1)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(e[1,:])
            CA = np.array(CA)
        if level == 2:
            CA_EEG, _, _ = pywt.wavedec(e[0,:],wavelet,level=2)
            CA_EOG, _, _ = pywt.wavedec(e[1,:],wavelet,level=2)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(e[1,:])

            CA = np.array(CA)
        if level == 3:
            CA_EEG, _, _, _ = pywt.wavedec(e[0,:],wavelet,level=3)
            CA_EOG, _, _, _ = pywt.wavedec(e[1,:],wavelet,level=3)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(e[1,:])

            CA = np.array(CA)
        if level == 4:
            CA_EEG, _, _, _, _  = pywt.wavedec(e[0,:],wavelet,level=4)
            CA_EOG, _, _, _, _  = pywt.wavedec(e[1,:],wavelet,level=4)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(e[1,:])
            CA = np.array(CA)
        if level == 5:
            CA_EEG, _, _, _, _, _  = pywt.wavedec(e[0,:],wavelet,level=5)
            CA_EOG, _, _, _, _ , _ = pywt.wavedec(e[1,:],wavelet,level=5)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(e[1,:])
            CA = np.array(CA)
        #CA= np.ravel(CA)
        #CA = np.append(CA,CD1)
        train_matrix.append(CA)
        
        # if pos != 0:
        #     prev_label = labels[pos-1]
        # else:
        #     prev_label = labels[pos] 

        # mean = np.mean(CA)
        # median = np.median(CA)
        # std = np.std(CA)
        # kurt = kurtosis(CA,axis=None)
        # sk = skew(CA,axis=None)

        # Features = np.array([mean,median,std,kurt,sk,prev_label])
        # train_matrix.append(Features)

        pos = pos +1
        
        #CA = np.append(CA,CD2)
        #CA = np.append(CA,CD3)
        #CA = np.append(CA,CD4)

        #CA = np.append(CD1,CD2)
        #CA = np.append(CA,CD3)
        #CA = np.append(CA,CD3)



    
    print('Len features: ', len(CA))

    train_matrix = np.array(train_matrix)

    #ESTANDARIZACION ENTRENAMIEO
    train_norm, medias, stds = normalize_mat(train_matrix)

    #PCA
    if features != -1:
        pca = PCA(n_components=features)
        train_norm = pca.fit_transform(train_norm)

    #RANDOM FOREST
    clf1 = RandomForestClassifier(n_estimators=trees ,random_state=random.seed(1234))
    clf1.fit(train_norm,labels)

    valid_wavelet = []
    y_valid = valid.events[:,-1]
    pos=0
    for v in valid:
        if level == 1:
            CA_EEG, _ = pywt.wavedec(v[0,:],wavelet,level=1)
            CA_EOG, _ = pywt.wavedec(v[1,:],wavelet,level=1)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(v[1,:])
            CA = np.array(CA)
        if level == 2:
            CA_EEG, _ , _= pywt.wavedec(v[0,:],wavelet,level=2)
            CA_EOG, _, _ = pywt.wavedec(v[1,:],wavelet,level=2)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(v[1,:])
            CA = np.array(CA)
        if level == 3:
            CA_EEG, _, _, _ = pywt.wavedec(v[0,:],wavelet,level=3)
            CA_EOG, _, _, _ = pywt.wavedec(v[1,:],wavelet,level=3)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(v[1,:])
            CA = np.array(CA)
        if level == 4:
            CA_EEG, _, _, _, _ = pywt.wavedec(v[0,:],wavelet,level=4)
            CA_EOG, _, _, _, _ = pywt.wavedec(v[1,:],wavelet,level=4)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(v[1,:])
            CA = np.array(CA)
        if level == 5:
            CA_EEG, _, _, _, _, _ = pywt.wavedec(v[0,:],wavelet,level=5)
            CA_EOG, _, _, _, _, _ = pywt.wavedec(v[1,:],wavelet,level=5)
            #CA = list(CA_EEG) + list(CA_EOG)
            CA = list(CA_EEG) + list(v[1,:])
            CA = np.array(CA)


        #CA= np.ravel(CA)
        #ESTADARIZACION VALIDACION
        for i in range(len(CA)):
            CA[i]= (CA[i]-medias[i])/stds[i]
        valid_wavelet.insert(len(valid_wavelet), CA)
        
        #RFS 
        # if pos != 0:
        #     prev_label = y_valid[pos-1]
        # else:
        #     prev_label = y_valid[pos] 

        # mean = np.mean(CA)
        # median = np.median(CA)
        # std = np.std(CA)
        # kurt = kurtosis(CA,axis=None)
        # sk = skew(CA,axis=None)


        # Features = np.array([mean,median,std,kurt,sk,prev_label])
        # pos = pos +1

        # #ESTADARIZACION VALIDACION
        # for i in range(len(Features)):
        #     Features[i]= (Features[i]-medias[i])/stds[i]
        # valid_wavelet.insert(len(valid_wavelet), Features)

        

    valid_matrix = np.array(valid_wavelet)
    
    if features != -1:
        valid_matrix=pca.transform(valid_matrix)
    
    y_predict = clf1.predict(valid_matrix)

    acc = (accuracy_score(y_valid,y_predict))
    f1 = (f1_score(y_valid,y_predict,average='macro'))
    
    S1, S2, SWS, R = multilabel_confusion_matrix(y_valid, y_predict)
    #S1, S2, SWS = multilabel_confusion_matrix(y_valid, y_predict)

    acc_s1, f1_s1 = metrics.get_metrics(S1,1)
    acc_s2, f1_s2 = metrics.get_metrics(S2,1)
    acc_sws, f1_sws = metrics.get_metrics(SWS,1)
    acc_r, f1_r = metrics.get_metrics(R,1)


    #print('acc: ', acc, ' f1: ', f1, ' pres: ', pres, ' rec: ', rec)
    return acc,  acc_s1, acc_s2, acc_sws, acc_r, f1,  f1_s1, f1_s2, f1_sws, f1_r

#def save_results(f1, f1_s1, f1_s2, f1_sws, w, r, l, f,t):
def save_results(f1, f1_s1, f1_s2, f1_sws, f1_r, w, r, l, f,t):

    df = pd.read_csv("results.csv")
    f1= np.mean(f1)

    f1_s1= np.mean(f1_s1)
    f1_s2= np.mean(f1_s2)
    f1_sws= np.mean(f1_sws)
    f1_r = np.mean(f1_r)
    #print(' RESULTADOS OBTENIDOS A GAURDAR ',f1, f1_s1, f1_s2, f1_sws )


    df[w + '_' + str(r) + '_' + str(l) + '_' + str(f)+'_' + str(t)] = [f1,  f1_s1, f1_s2, f1_sws, f1_r]

    df.to_csv("results.csv", index=False)

    #line = 'modelo' + w + str(r) + str(l) + str(f): [f1, f1_s1,f1_s2,f1_s3]

def train(wavelet, rus, level, features,trees, epochs, test_epochs):
    acc = np.empty(len(test_epochs))
    f1 = np.empty(len(test_epochs))    

    acc_w = np.empty(len(test_epochs))
    f1_w = np.empty(len(test_epochs))

    acc_s1 = np.empty(len(test_epochs))
    f1_s1 = np.empty(len(test_epochs))

    acc_s2 = np.empty(len(test_epochs))
    f1_s2 = np.empty(len(test_epochs))

    acc_sws = np.empty(len(test_epochs))
    f1_sws = np.empty(len(test_epochs))

    acc_r = np.empty(len(test_epochs))
    f1_r = np.empty(len(test_epochs))

    epochs = mne.concatenate_epochs(epochs)

    print('EPOCAS ENTRENAMIENTO: ', epochs)

    epochs = balance_data.balance_epochs(epochs,30)
    labels = epochs.events[:, -1]    

    train_matrix= []
    pos = 0
    for e in epochs:
        #print('TAMAÑO: ', e.shape)
        if level == 1:
            CA, CD = pywt.wavedec(e[0,:],wavelet,level=1)
        if level == 2:
            CA, CD1, CD2 = pywt.wavedec(e[0,:],wavelet,level=2)
        if level == 3:
            #print('ENTRO AL LEVEL')
            CA, CD1, CD2, CD3 = pywt.wavedec(e[0:],wavelet,level=3)
        if level == 4:
            CA, CD1, CD2, CD3, CD4 = pywt.wavedec(e[0,:],wavelet,level=4)
        
        #ORIGINAL
        #CA= np.ravel(CA)
        #train_matrix.append(CA)

        if pos != 0:
            prev_label = labels[pos-1]
        else:
            prev_label = labels[pos] 

        mean = np.mean(CA)
        median = np.median(CA)
        std = np.std(CA)
        kurt = kurtosis(CA,axis=None)
        sk = skew(CA,axis=None)


        Features = np.array([mean,median,std,kurt,sk,prev_label])
        train_matrix.append(Features)
        pos = pos +1
        #print(train_matrix)

    
    
    
    #print('TRAIN LIST SHAPE ', len(train_matrix))
    #print('CA SHAPE ', CA.shape)

    train_matrix = np.array(train_matrix)

    #print('TRAIN MATRIX SHAPE ', train_matrix.shape)
    #ESTANDARIZACION ENTRENAMIEO
    train_norm, medias, stds = normalize_mat(train_matrix)

    #plot_wavelet(train_norm,labels)

    #PCA
    pca = PCA(n_components=features)
    if features != -1:
        print('ENTRA A PCA')
        
        train_norm = pca.fit_transform(train_norm)

    tSNE.plot_tSNE(train_norm,labels)

    #RANDOM FOREST
    clf1 = RandomForestClassifier(n_estimators=trees ,random_state=random.seed(1234))
    clf1.fit(train_norm,labels)
    index = 0
    for t in test_epochs:
        (acc[index], acc_w[index], acc_s1[index], acc_s2[index], acc_sws[index],acc_r[index],
        f1[index], f1_w[index], f1_s1[index], f1_s2[index],f1_sws[index], 
        f1_r[index]) = predecir_sj(wavelet,rus, level, features, clf1,t, medias,stds,pca)

        # 3 clases
        # (acc[index], acc_s1[index], acc_s2[index], acc_sws[index],
        # f1[index], f1_s1[index], f1_s2[index],f1_sws[index]
        # ) = predecir_sj(wavelet, rus, level, features, clf1,t , medias,stds,pca)
        
        index = index +1
    f = open("Fscore.txt", "w")
    f.write('Data: '+ 'f1: '+ str(f1_s1) +'f2 '+ str(f1_s2) +'f3'+ str(f1_sws))
    f.close()
    save_results(f1, f1_w, f1_s1, f1_s2, f1_sws, f1_r, wavelet, rus, level, features, trees)
    #save_results(f1, f1_s1, f1_s2, f1_sws, wavelet, rus, level, features, trees)
    
    
    # f = open("MODELO SIMPLE.txt", "w")
    # #f.write(str(family) + '\n')
    # f.write('Accuracy- mean: ' + str(np.mean(acc)) +'\n')
    # f.write('F1 Score - mean: ' + str(np.mean(f1)) +'\n')
    # f.write('Precision- mean: '+ str(np.mean(prec))+'\n')
    # f.write('Recall- mean: '+ str(np.mean(recall))+'\n')
    # f.write('Stage 1: Accuracy- '+ str(np.mean(acc_s1))+ ' Fscore: '+ str(np.mean(f1_s1))+ ' Precision- '+ str(np.mean(prec_s1))+ ' Recall-'+ str(np.mean(recall_s1))+'\n')
    # f.write('Stage 2: Accuracy- '+ str(np.mean(acc_s2))+ ' Fscore: '+ str(np.mean(f1_s2))+ ' Precision- '+ str(np.mean(prec_s2))+ ' Recall-'+ str(np.mean(recall_s2))+'\n')
    # f.write('Stage 3: Accuracy- '+ str(np.mean(acc_s3))+ ' Fscore: '+ str(np.mean(f1_s3))+ ' Precision- '+ str(np.mean(prec_s3))+ ' Recall-'+ str(np.mean(recall_s3))+'\n')
    # f.write('Data: '+ 'f1: '+ str(f1_s1) +'f2 '+ str(f1_s2) +'f3'+ str(f1_s3))
    # f.close()

def predecir_sj(wavelet, rus, level, features, rf, test, medias, stds,pca=10):

    print('test subject: ', test)
    test_wavelet = []
    y_test = test.events[:,-1]
    pos = 0
    for t in test:
        if level == 1:
            CA, CD = pywt.wavedec(t[0,:],wavelet,level=1)
        if level == 2:
            CA, CD1, CD2 = pywt.wavedec(t[0,:],wavelet,level=2)
        if level == 3:
            CA, CD1, CD2, CD3 = pywt.wavedec(t[0,:],wavelet,level=3)
        if level == 4:
            CA, CD1, CD2, CD3, CD4 = pywt.wavedec(t[0,:],wavelet,level=4)
        #CA, CD1, CD2, CD3 = pywt.wavedec(t[0,:],wavelet,level=3)

        #print('CA TEST SHAPE: ', CA.shape)
        #ESTADARIZACION VALIDACION

        #for i in range(len(CA)):
        #    CA[i]= (CA[i]-medias[i])/stds[i]
        #test_wavelet.insert(len(test_wavelet), CA)

        #mean, var, skew, kurt = skewnorm.stats(CA, moments='mvsk')
        #Features = np.array([mean,var,skew,kurt])
        if pos != 0:
            prev_label = y_test[pos-1]
        else:
            prev_label = y_test[pos] 

        mean = np.mean(CA)
        median = np.median(CA)
        std = np.std(CA)
        kurt = kurtosis(CA,axis=None)
        sk = skew(CA,axis=None)
        #H = entropy(CA)

        Features = np.array([mean,median,std,kurt,sk,prev_label])

        pos = pos +1

        for i in range(len(Features)):
            Features[i]= (Features[i]-medias[i])/stds[i]
        test_wavelet.insert(len(test_wavelet), Features)

        

    test_matrix = np.array(test_wavelet)
    
    if features != -1:
        test_matrix=pca.transform(test_matrix)
    
    y_predict = rf.predict(test_matrix)

    acc = (accuracy_score(y_test,y_predict))
    f1 = (f1_score(y_test,y_predict,average='macro'))
    
    #5 CLASES
    W, S1, S2, SWS, R = multilabel_confusion_matrix(y_test, y_predict)

    #3 CLASES
    #S1, S2, SWS = multilabel_confusion_matrix(y_test, y_predict)


    acc_w, f1_w = metrics.get_metrics(W,1)
    acc_s1, f1_s1 = metrics.get_metrics(S1,1)
    acc_s2, f1_s2 = metrics.get_metrics(S2,1)
    acc_sws, f1_sws = metrics.get_metrics(SWS,1)
    acc_r, f1_r = metrics.get_metrics(R,1)

    #plot_confusion_matrix(rf, test_matrix, y_test)  # doctest: +SKIP
    plt.show()  # doctest: +SKIP

    print('acc: ', acc, ' f1: ', f1)
    return acc, acc_w, acc_s1, acc_s2, acc_sws, acc_r, f1, f1_w, f1_s1, f1_s2, f1_sws, f1_r
    #return acc, acc_s1, acc_s2, acc_sws, f1, f1_s1, f1_s2, f1_sws

def get_test_by_age(epoch, sj, total_sj,group,ages):
    if group == 1:
        min_age = 26
        max_age = 35
    if group == 2:
        min_age = 50
        max_age = 60
    if group == 3:
        min_age = 66
        max_age = 75
    if group == 4:
        min_age = 85
        max_age = 101
    new_epochs = []
    new_sj = []
    for i in range(len(total_sj)):
        if (total_sj[i] in sj) and (ages[i] >= min_age) and (ages[i]<= max_age):
            index = sj.index(total_sj[i])
            new_epochs.insert(len(new_epochs),epoch[index])
            new_sj.insert(len(new_sj),(sj[index])) 
    return new_epochs,new_sj

def main(wavelet,RUS,level,n_features,n_trees):
    
    epochs, subjects = read_EDF_data_per_sj_4.get_eeg_data(0,40)

    # seed = random.random()
    # random.seed(seed)
    # random.shuffle(epochs)
    # random.seed(seed)
    # random.shuffle(subjects)
    # random.seed(seed)
    # random.shuffle(ages)

    cant_train = 70 * len(epochs) / 100
    cant_train= int(cant_train)

    #70 entrenamiento 30 test
    train_epochs = epochs[:cant_train]
    test_epochs = epochs[cant_train:]



    # train_epochs,train_sj,_ = read_EDF_data_per_sj.get_eeg_data(0,40)
    # test_epochs,_,_ = read_EDF_data_per_sj.get_eeg_data(40,61)
    # train_epochs = train_epochs[:14]
    # test_epochs = test_epochs[14:]

    
    # #datos de test
    #test_epochs = epochs[cant_train:]
    #test_sj = subjects[cant_train:]
    #test_epochs = mne.concatenate_epochs(test_epochs)

    #test_epochs, test_sj = get_test_by_age(test_epochs,test_sj,subjects,1,ages)
    

    #PARA 3 CLASES
    data= {'fscore': ['General', 'Stage 1', 'Stage 2' ,' Stage SWS', 'REM']}
    
    df = pd.DataFrame(data)
    df.to_csv('results.csv')

    acc = np.empty(len(train_epochs))
    f1 = np.empty(len(train_epochs))

    acc_w = np.empty(len(train_epochs))
    f1_w = np.empty(len(train_epochs))

    acc_s1 = np.empty(len(train_epochs))
    f1_s1 = np.empty(len(train_epochs))

    acc_s2 = np.empty(len(train_epochs))
    f1_s2 = np.empty(len(train_epochs))

    acc_sws = np.empty(len(train_epochs))
    f1_sws = np.empty(len(train_epochs))

    acc_r = np.empty(len(train_epochs))
    f1_r = np.empty(len(train_epochs))

    for w in wavelet:
        for r in RUS:
            for l in level:
                for f in n_features:
                    for t in n_trees:
                    #preparamos archivo csv para almacenar datos
                        #train(w, r, l, f,t, train_epochs.copy(),test_epochs)

                        for i in range(len(train_epochs)):
                            (acc[i], acc_s1[i], acc_s2[i], acc_sws[i], acc_r[i],
                            f1[i], f1_s1[i], f1_s2[i], f1_sws[i], f1_r[i]) = train_data(w, r, l, f, t, train_epochs.copy(),i)             
                        save_results(f1,f1_s1,f1_s2,f1_sws,f1_r, w,r,l,f,t)
                        print(np.mean(acc))
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