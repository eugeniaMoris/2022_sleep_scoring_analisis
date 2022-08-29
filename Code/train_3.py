#Clasificacion de estapas de sueño utilizando wivelets y el dataset de sleep-edf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix,confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import tSNE
import metrics
import read_EDF_data_per_sj
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
import xlwt
from numpy import savetxt


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
    #print('ENTRO A NORM', mat.shape)
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
            CA, CD = pywt.wavedec(e[0,:],wavelet,level=1)
        if level == 2:
            CA, CD1, CD2 = pywt.wavedec(e[0,:],wavelet,level=2)
        if level == 3:
            CA, CD1, CD2, CD3 = pywt.wavedec(e[0:],wavelet,level=3)
        if level == 4:
            CA, CD1, CD2, CD3, CD4 = pywt.wavedec(e[0,:],wavelet,level=4)
        if level == 5:
            CA, CD1, CD2, CD3, CD4, CD5 = pywt.wavedec(e[0,:],wavelet,level=5)
        
        CA= np.ravel(CA)
        #CA = np.append(CA,CD1)
        #train_matrix.append(CA)

        #print('SHAPE CA:', CA.shape)
        
        #APOCA ANTERIOR
        # if pos != 0:
        #     prev_label = labels[pos-1]
        # else:
        #     prev_label = labels[pos] 

        #print('CA shape: ', CA.shape)

        mean = np.mean(CA)
        median = np.median(CA)
        std = np.std(CA)
        kurt = kurtosis(CA,axis=None)
        sk = skew(CA,axis=None)

        feature_list = CA.tolist()
        

        feature_list.insert(len(feature_list),mean)
        feature_list.insert(len(feature_list),median)
        feature_list.insert(len(feature_list),std)
        feature_list.insert(len(feature_list),kurt)
        feature_list.insert(len(feature_list),sk)

        #print('len de CA: ', len(feature_list))

        Features= np.array(feature_list)
        Features= np.ravel(Features)

        

        #print('SHAPE OF FEATURES: ', Features.shape)


        #Features = np.array([mean,median,std,kurt,sk,prev_label])
        #Features = np.array([CA,mean,median,std,kurt,sk])


        train_matrix.append(Features)
        pos = pos +1
        


    
    print('Len features: ', len(Features))

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
            CA, CD = pywt.wavedec(v[0,:],wavelet,level=1)
        if level == 2:
            CA, CD1, CD2 = pywt.wavedec(v[0,:],wavelet,level=2)
        if level == 3:
            CA, CD1, CD2, CD3 = pywt.wavedec(v[0:],wavelet,level=3)
        if level == 4:
            CA, CD1, CD2, CD3, CD4 = pywt.wavedec(v[0,:],wavelet,level=4)
        if level == 5:
            CA, CD1, CD2, CD3, CD4, CD5 = pywt.wavedec(v[0,:],wavelet,level=5)
        
        #ESTADARIZACION VALIDACION
        #for i in range(len(CA)):
        #    CA[i]= (CA[i]-medias[i])/stds[i]
        #valid_wavelet.insert(len(valid_wavelet), CA)
        
        #EPOCA ANTERIOR
        # if pos != 0:
        #     prev_label = y_valid[pos-1]
        # else:
        #     prev_label = y_valid[pos] 

        CA= np.ravel(CA)

        feature_list = CA.tolist()
        mean = np.mean(CA)
        median = np.median(CA)
        std = np.std(CA)
        kurt = kurtosis(CA,axis=None)
        sk = skew(CA,axis=None)

        feature_list.insert(len(feature_list),mean)
        feature_list.insert(len(feature_list),median)
        feature_list.insert(len(feature_list),std)
        feature_list.insert(len(feature_list),kurt)
        feature_list.insert(len(feature_list),sk)

        Features= np.array(feature_list)


        #Features = np.array([mean,median,std,kurt,sk])
        #Features = np.array([mean,median,std,kurt,sk,prev_label])

        pos = pos +1

        #ESTADARIZACION VALIDACION
        for i in range(len(Features)):
            Features[i]= (Features[i]-medias[i])/stds[i]
        valid_wavelet.insert(len(valid_wavelet), Features)

        

    valid_matrix = np.array(valid_wavelet)
    
    if features != -1:
        valid_matrix=pca.transform(valid_matrix)
    
    y_predict = clf1.predict(valid_matrix)

    acc = (accuracy_score(y_valid,y_predict))
    f1 = (f1_score(y_valid,y_predict,average='macro'))
    
    
    S1, S2, SWS = multilabel_confusion_matrix(y_valid, y_predict)

    
    acc_s1, f1_s1 = metrics.get_metrics(S1,1)
    acc_s2, f1_s2 = metrics.get_metrics(S2,1)
    acc_sws, f1_sws = metrics.get_metrics(SWS,1)
    


    #print('acc: ', acc, ' f1: ', f1, ' pres: ', pres, ' rec: ', rec)
    return acc, acc_s1, acc_s2, acc_sws, f1, f1_s1, f1_s2, f1_sws

def save_results(f1, f1_s1, f1_s2, f1_sws, w, r, l, f,t):

    df = pd.read_csv("results.csv")
    f1= np.mean(f1)

    f1_s1= np.mean(f1_s1)
    f1_s2= np.mean(f1_s2)
    f1_sws= np.mean(f1_sws)
    #print(' RESULTADOS OBTENIDOS A GAURDAR ',f1, f1_s1, f1_s2, f1_sws )


    df[w + '_' + str(r) + '_' + str(l) + '_' + str(f)+'_' + str(t)] = [f1, f1_s1, f1_s2, f1_sws]

    df.to_csv("results.csv", index=False)

    #line = 'modelo' + w + str(r) + str(l) + str(f): [f1, f1_s1,f1_s2,f1_s3]

def save(data, path, labels):
    savetxt('features.csv', data, delimiter=',')
    savetxt('labels.csv', labels)


def train(wavelet, rus, level, features,trees, epochs, test_epochs):
    acc = np.empty(len(test_epochs))
    f1 = np.empty(len(test_epochs))    

    acc_s1 = np.empty(len(test_epochs))
    f1_s1 = np.empty(len(test_epochs))

    acc_s2 = np.empty(len(test_epochs))
    f1_s2 = np.empty(len(test_epochs))

    acc_sws = np.empty(len(test_epochs))
    f1_sws = np.empty(len(test_epochs))

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
        if level == 5:
            CA, CD1, CD2, CD3, CD4, CD5 = pywt.wavedec(e[0,:],wavelet, level=5)
        
        #ORIGINAL
        CA= np.ravel(CA)
        train_matrix.append(CA)

        #EPOCA ANTERIOR
        # if pos != 0:
        #     prev_label = labels[pos-1]
        # else:
        #     prev_label = labels[pos] 

        #PARA RFS
        # mean = np.mean(CA)
        # median = np.median(CA)
        # std = np.std(CA)
        # kurt = kurtosis(CA,axis=None)
        # sk = skew(CA,axis=None)


        # #PARA EFS
        # feature_list = CA.tolist()
        # feature_list.insert(len(feature_list),mean)
        # feature_list.insert(len(feature_list),median)
        # feature_list.insert(len(feature_list),std)
        # feature_list.insert(len(feature_list),kurt)
        # feature_list.insert(len(feature_list),sk)
        # Features= np.array(feature_list)
        # Features= np.ravel(Features)


        #Features = np.array([mean,median,std,kurt,sk,prev_label])
        #Features = np.array([mean,median,std,kurt,sk])

        #train_matrix.append(Features)
        pos = pos +1

    train_matrix = np.array(train_matrix)

    path= '/media/piddef/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final'
    save(train_matrix,path,labels)

    #ESTANDARIZACION ENTRENAMIEO --------------
    train_norm, medias, stds = normalize_mat(train_matrix)

    #plot_wavelet(train_norm,labels)

    #PCA
    pca = PCA(n_components=features)
    if features != -1:
        print('ENTRA A PCA')
        
        train_norm = pca.fit_transform(train_norm)

    #tSNE.plot_tSNE(train_norm,labels)

    #RANDOM FOREST
    clf1 = RandomForestClassifier(n_estimators=trees ,random_state=random.seed(1234))
    clf1.fit(train_norm,labels)
    index = 0
    for t in test_epochs:
        (acc[index], acc_s1[index], acc_s2[index], acc_sws[index],
        f1[index], f1_s1[index], f1_s2[index],f1_sws[index]
        ) = predecir_sj(wavelet, rus, level, features, clf1,t , medias,stds,pca)
        
        index = index +1
    f = open("Fscore.txt", "w")
    f.write('Data: '+ 'f1: '+ str(f1_s1) +'f2 '+ str(f1_s2) +'f3'+ str(f1_sws))
    f.close()

    save_results(f1, f1_s1, f1_s2, f1_sws, wavelet, rus, level, features, trees)
    #save_results(acc, acc_s1, acc_s2, acc_sws, wavelet, rus, level, features, trees)
    

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
        if level == 5:
            CA, CD1, CD2, CD3, CD4, CD5 = pywt.wavedec(t[0,:],wavelet,level=5)
        #CA, CD1, CD2, CD3 = pywt.wavedec(t[0,:],wavelet,level=3)

        #print('CA TEST SHAPE: ', CA.shape)
        #ESTADARIZACION VALIDACION
        

        for i in range(len(CA)):
           CA[i]= (CA[i]-medias[i])/stds[i]
        test_wavelet.insert(len(test_wavelet), CA)

        #PREVIOUS EPOCH
        # if pos != 0:
        #     prev_label = y_test[pos-1]
        # else:
        #     prev_label = y_test[pos] 

        #PARA RFS
        # CA= np.ravel(CA)

        # mean = np.mean(CA)
        # median = np.median(CA)
        # std = np.std(CA)
        # kurt = kurtosis(CA,axis=None)
        # sk = skew(CA,axis=None)

        # #PARA EFS   
        # feature_list = CA.tolist()
        # feature_list.insert(len(feature_list),mean)
        # feature_list.insert(len(feature_list),median)
        # feature_list.insert(len(feature_list),std)
        # feature_list.insert(len(feature_list),kurt)
        # feature_list.insert(len(feature_list),sk)
        # Features= np.array(feature_list)
        # Features= np.ravel(Features)

        # #Features = np.array([mean,median,std,kurt,sk])
        # #Features = np.array([mean,median,std,kurt,sk,prev_label])


        # pos = pos +1

        # for i in range(len(Features)):
        #     Features[i]= (Features[i]-medias[i])/stds[i]
        # test_wavelet.insert(len(test_wavelet), Features)

        

    test_matrix = np.array(test_wavelet)
    
    if features != -1:
        test_matrix=pca.transform(test_matrix)
    
    y_predict = rf.predict(test_matrix)

    acc = (accuracy_score(y_test,y_predict))
    f1 = (f1_score(y_test,y_predict,average='macro'))

    #3 CLASES
    S1, S2, SWS = multilabel_confusion_matrix(y_test, y_predict)

    acc_s1, f1_s1 = metrics.get_metrics(S1,1)
    acc_s2, f1_s2 = metrics.get_metrics(S2,1)
    acc_sws, f1_sws = metrics.get_metrics(SWS,1)

    print(confusion_matrix(y_true=y_test,y_pred=y_predict))


    #plot_confusion_matrix(rf, test_matrix, y_test)  # doctest: +SKIP
    #plt.show()  # doctest: +SKIP

    return acc, acc_s1, acc_s2, acc_sws, f1, f1_s1, f1_s2, f1_sws

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
    
    # epochs_G1,_= read_EDF_data_per_sj.get_eeg_data(0,40)
    # #epochs_G2,_= read_EDF_data_per_sj.get_eeg_data(40,60)
    # #epochs_G3,_= read_EDF_data_per_sj.get_eeg_data(61,80)
    # #epochs_G4,_= read_EDF_data_per_sj.get_eeg_data(81,100)


    # cant_train_G1 = 70 * len(epochs_G1) / 100
    # cant_train_G1= int(cant_train_G1)
    # train_epochs_G1 = epochs_G1[:cant_train_G1]
    # test_epochs_G1 = epochs_G1[cant_train_G1:]


    # cant_train_G2 = 70 * len(epochs_G2) / 100
    # cant_train_G2= int(cant_train_G2)
    # train_epochs_G2 = epochs_G2[:cant_train_G2]
    # test_epochs_G2 = epochs_G2[cant_train_G2:]

    # cant_train_G3 = 70 * len(epochs_G3) / 100
    # cant_train_G3= int(cant_train_G3)
    # train_epochs_G3 = epochs_G3[:cant_train_G3]
    # test_epochs_G3 = epochs_G3[cant_train_G3:]

    # cant_train_G4 = 70 * len(epochs_G4) / 100
    # cant_train_G4= int(cant_train_G4)
    # train_epochs_G4 = epochs_G4[:cant_train_G4]
    # test_epochs_G4 = epochs_G4[cant_train_G4:]

    #train_epochs = train_epochs_G1[:4] + train_epochs_G2[:4] + train_epochs_G3[:4] + train_epochs_G4[:4]
    #print(len(train_epochs))
    #test_epochs = test_epochs_G4

    # seed = random.random()
    # random.seed(seed)
    # random.shuffle(epochs)
    # random.seed(seed)
    # random.shuffle(subjects)
    # random.seed(seed)
    # random.shuffle(ages)

    epochs,train_sj = read_EDF_data_per_sj.get_eeg_data(0,40)

    cant_train = 70 * len(epochs) / 100
    cant_train= int(cant_train)

    # #70 entrenamiento 30 test
    train_epochs = epochs[:cant_train]
    #train_sj = subjects[:cant_train]

    # # #datos de test
    test_epochs = epochs[cant_train:]
    #test_sj = subjects[cant_train:]

    # train_epochs,train_sj,_ = read_EDF_data_per_sj.get_eeg_data(0,40)
    # test_epochs,_,_ = read_EDF_data_per_sj.get_eeg_data(40,61)
    # train_epochs = train_epochs[:14]
    # test_epochs = test_epochs[14:]

    
    #test_epochs = mne.concatenate_epochs(test_epochs)

    #test_epochs, test_sj = get_test_by_age(test_epochs,test_sj,subjects,1,ages)
    

    #PARA 3 CLASES
    data= {'fscore': ['Wake', 'Stage 1', 'Stage 2' ,' Stage SWS']}
    
    df = pd.DataFrame(data)
    df.to_csv('results.csv')

    acc = np.empty(len(train_epochs))
    f1 = np.empty(len(train_epochs))

    acc_s1 = np.empty(len(train_epochs))
    f1_s1 = np.empty(len(train_epochs))

    acc_s2 = np.empty(len(train_epochs))
    f1_s2 = np.empty(len(train_epochs))

    acc_sws = np.empty(len(train_epochs))
    f1_sws = np.empty(len(train_epochs))


    for w in wavelet:
        for r in RUS:
            for l in level:
                for f in n_features:
                    for t in n_trees:
                    #preparamos archivo csv para almacenar datos
                        train(w, r, l, f,t, train_epochs.copy(),test_epochs)

                        #VALIDACION
                        # for i in range(len(train_epochs)):
                        #     (acc[i], acc_s1[i], acc_s2[i], acc_sws[i],
                        #     f1[i], f1_s1[i], f1_s2[i], f1_sws[i]) = train_data(w, r, l, f, t, train_epochs.copy(),i)             
                        # save_results(f1,f1_s1,f1_s2,f1_sws, w,r,l,f,t)
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