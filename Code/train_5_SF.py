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

def save(data, path, labels):
    np.savetxt('features_SF.csv', data, delimiter=',')
    np.savetxt('labels_SF.csv', labels)

def wavelet_level(señal, level,wavelet, valid, medias, stds):
    train_matrix= []
    level = int(level)
    #print('BEFORE GOT THE FEATURES')

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

        #RFS
        Features = np.array([mean_eeg,median_eeg,std_eeg,kurt_eeg,sk_eeg,mean_eog,median_eog,std_eog,kurt_eog,sk_eog,mean_emg,median_emg,std_emg,kurt_emg,sk_emg])
        #Features = np.array([mean_eeg,median_eeg,std_eeg,kurt_eeg,sk_eeg,mean_eog,median_eog,std_eog,kurt_eog,sk_eog])
        #Features = np.array([mean_eeg,median_eeg,std_eeg,kurt_eeg,sk_eeg])
        #print('FEATURES: ', Features)
        
        #MCM
        # Features = np.ravel(CA_EEG)
        # Features = np.append(Features, CA_EOG)
        # Features = np.append(Features, CA_EMG)
        #CA= CA_EEG

        #PARA EFS
        #feature_list = CA.tolist()
        #EEG
        # feature_list.insert(len(feature_list),mean_eeg)
        # feature_list.insert(len(feature_list),median_eeg)
        # feature_list.insert(len(feature_list),std_eeg)
        # feature_list.insert(len(feature_list),kurt_eeg)
        # feature_list.insert(len(feature_list),sk_eeg)
        #EOG
        # feature_list.insert(len(feature_list),mean_eog)
        # feature_list.insert(len(feature_list),median_eog)
        # feature_list.insert(len(feature_list),std_eog)
        # feature_list.insert(len(feature_list),kurt_eog)
        # feature_list.insert(len(feature_list),sk_eog)
        #EMG
        # feature_list.insert(len(feature_list),mean_emg)
        # feature_list.insert(len(feature_list),median_emg)
        # feature_list.insert(len(feature_list),std_emg)
        # feature_list.insert(len(feature_list),kurt_emg)
        # feature_list.insert(len(feature_list),sk_emg)


        # Features= np.array(feature_list)
        Features= np.ravel(Features)

        #print('GOT THE FEATURES')
        #Features = np.ravel(CA)
        if valid==0:
            train_matrix.append(Features)
        elif valid==1:
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
    print('SUJETO: ')
    epochs.pop(index)

    epochs = mne.concatenate_epochs(epochs)
    #BALANCEO DE CLASES - SE REDUCE CLASE 2
    epochs = balance_data.balance_epochs(epochs,rus)
    labels = epochs.events[:, -1]


    #WAVELET ENTRENAMIENTO
    train_matrix= wavelet_level(epochs,level,wavelet,0,0,0)
    train_matrix = np.array(train_matrix)

    

    #ESTANDARIZACION ENTRENAMIEO
    train_norm, medias, stds = normalize_mat(train_matrix)

    path= '/home/emoris/wavelets'
    save(train_norm,path,labels)



    #PCA
    if features != -1:
        pca = PCA(n_components=features)
        train_norm = pca.fit_transform(train_norm)

    #RANDOM FOREST
    clf1 = RandomForestClassifier(n_estimators=trees ,random_state=random.seed(1234))
    clf1.fit(train_norm,labels)


    #WAVELET VALIDACION
    valid_wavelet = wavelet_level(valid,level,wavelet,1,medias.copy(),stds.copy())
    y_valid = valid.events[:,-1]
      

    valid_matrix = np.array(valid_wavelet)
    
    if features != -1:
        valid_matrix=pca.transform(valid_matrix)
    
    y_predict = clf1.predict(valid_matrix)

    acc = (accuracy_score(y_valid,y_predict))
    f1 = (f1_score(y_valid,y_predict,average='macro'))
    
    W, S1, S2, SWS, R = multilabel_confusion_matrix(y_valid, y_predict)
    #S1, S2, SWS = multilabel_confusion_matrix(y_valid, y_predict)

    acc_w, f1_w, sen_w, spe_w= metrics.get_metrics(W,1)
    acc_s1, f1_s1, sen_s1, spe_s1 = metrics.get_metrics(S1,1)
    acc_s2, f1_s2, sen_s2, spe_s2 = metrics.get_metrics(S2,1)
    acc_sws, f1_sws, sen_sws, spe_sws = metrics.get_metrics(SWS,1)
    acc_r, f1_r, sen_r, spe_r = metrics.get_metrics(R,1)
    print(sen_w,spe_w,sen_s1,spe_s1,sen_s2,spe_s2,sen_sws,spe_sws,sen_r,spe_r)

    f = open("Acc_MCM_SF.txt", "w")
    f.write('Data: '+ 'acc: '+ str(np.mean(acc)) +'acc_w '+ str(np.mean(acc_w))+'acc_s1 '+ str(np.mean(acc_s1))+'acc_s2 '+ str(np.mean(acc_s2))+'acc_sws '+ str(np.mean(acc_sws))+'acc_r '+ str(np.mean(acc_r)))
    f.close()

    file = open("Fscore_MCM_SF.txt", "w")
    file.write('Data: '+ 'fs_w '+ str(f1_w)+'/n'
    + 'fs_s1 '+ str(f1_s1)+'/n'+
    'fs_s2 '+ str(f1_s2)+'/n'+
    'f1_sws '+ str(f1_sws) +'/n'
    +'f1_r '+ str(f1_r))
    file.close()

    file = open("sen_spe_SF.txt", "w")
    file.write('Data: '+ '\n'+
    'WAKE sensitivity '+ str(sen_w)+' - specificity ' + str(spe_w)+ '\n' + 
    'STAGE 1 sensitivity '+ str(sen_s1)+ '- specificity ' + str(spe_s1) +'\n'+
    'STAGE 2 sensitivity '+ str(sen_s2)+ '- specificity ' + str(spe_s2) +  '\n'+
    'SWS semsitivity '+ str(sen_sws) + '- specificity ' + str(spe_sws) +'\n'+
    'REM sensitivity '+ str(sen_r) + '- specificity ' + str(spe_r))
    file.close()


    #print('acc: ', acc, ' f1: ', f1, ' pres: ', pres, ' rec: ', rec)
    return acc, acc_w, acc_s1, acc_s2, acc_sws, acc_r, f1, f1_w, f1_s1, f1_s2, f1_sws, f1_r

#def save_results(f1, f1_s1, f1_s2, f1_sws, w, r, l, f,t):
def save_results(f1, f1_w, f1_s1, f1_s2, f1_sws, f1_r, w, r, l, f,t):

    df = pd.read_csv("results_MCM_SF.csv")
    f1= np.mean(f1)

    f1_w = np.mean(f1_w)
    f1_s1= np.mean(f1_s1)
    f1_s2= np.mean(f1_s2)
    f1_sws= np.mean(f1_sws)
    f1_r = np.mean(f1_r)
    #print(' RESULTADOS OBTENIDOS A GAURDAR ',f1, f1_s1, f1_s2, f1_sws )


    df[w + '_' + str(r) + '_' + str(l) + '_' + str(f)+'_' + str(t)] = [f1, f1_w, f1_s1, f1_s2, f1_sws, f1_r]

    df.to_csv("results_MCM_SF.csv", index=False)

    print('fscores: ',f1)
    #line = 'modelo' + w + str(r) + str(l) + str(f): [f1, f1_s1,f1_s2,f1_s3]

def train(wavelet, rus, level, features,trees, epochs, test_epochs):
    acc = np.empty(len(test_epochs))
    f1 = np.empty(len(test_epochs))
    #presition = np.empty(len(test_epochs))
    #recall = np.empty(len(test_epochs))     

    acc_w = np.empty(len(test_epochs))
    f1_w = np.empty(len(test_epochs))
    #p_w = np.empty(len(test_epochs))
    #r_w = np.empty(len(test_epochs))

    acc_s1 = np.empty(len(test_epochs))
    f1_s1 = np.empty(len(test_epochs))
    #p_s1 = np.empty(len(test_epochs))
    #r_s1 = np.empty(len(test_epochs))

    acc_s2 = np.empty(len(test_epochs))
    f1_s2 = np.empty(len(test_epochs))
    #p_s2 = np.empty(len(test_epochs))
    #r_s2 = np.empty(len(test_epochs))

    acc_sws = np.empty(len(test_epochs))
    f1_sws = np.empty(len(test_epochs))
    #p_sws = np.empty(len(test_epochs))
    #r_sws = np.empty(len(test_epochs))

    acc_r = np.empty(len(test_epochs))
    f1_r = np.empty(len(test_epochs))
    #p_r = np.empty(len(test_epochs))
    #r_r = np.empty(len(test_epochs))

    epochs = mne.concatenate_epochs(epochs)
    #print('BEFORE BALANCE DATA', ' RUS ' ,rus)
    if rus != 100:
    
        epochs = balance_data.balance_epochs(epochs,rus)
    labels = epochs.events[:, -1]   

    #print('AFTER BALANCE DATA')
    train_matrix= wavelet_level(epochs,level,wavelet,0,0,0)
    train_matrix = np.array(train_matrix)


    #ESTANDARIZACION ENTRENAMIEO
    train_norm, medias, stds = normalize_mat(train_matrix)
    #print('AFTER NORMALICE DATA')

    #plot_wavelet(train_norm,labels)
    pca = PCA(n_components=features)

    #PCA
    if features != -1:
        #print('ENTRA A PCA')
        train_norm = pca.fit_transform(train_norm)
    #print('AFTER PCA DATA')
    

    #tSNE.plot_tSNE(train_norm,labels)

    #RANDOM FOREST
    clf1 = RandomForestClassifier(n_estimators=trees ,random_state=random.seed(1234))
    clf1.fit(train_norm,labels)
    
    #print('AFTER RANDOM FORREST')

    index = 0
    #print('TEST EPOCHS: ', test_epochs)
    for t in test_epochs:
        #print('index test: ', index)
        (acc[index], acc_w[index], acc_s1[index], acc_s2[index], acc_sws[index],acc_r[index],
        f1[index], f1_w[index], f1_s1[index], f1_s2[index],f1_sws[index], 
        f1_r[index]) = predecir_sj(wavelet,rus, level, features, clf1, t, medias ,stds ,pca,index)

        index = index +1

    # f = open("Acc_MCM_SF.txt", "w")
    # f.write('Data: '+ 'acc: '+ str(np.mean(acc)) +'acc_w '+ str(np.mean(acc_w))+'acc_s1 '+ str(np.mean(acc_s1))+'acc_s2 '+ str(np.mean(acc_s2))+'acc_sws '+ str(np.mean(acc_sws))+'acc_r '+ str(np.mean(acc_r)))
    # f.close()

    # file = open("Fscore_MCM_SF.txt", "w")
    # file.write('Data: '+ 'fs_w '+ str(f1_w)+' /n'
    # + 'fs_s1 '+ str(f1_s1)+'/n'+
    # 'fs_s2 '+ str(f1_s2)+'/n'+
    # 'f1_sws '+ str(f1_sws) +'/n'
    # +'f1_r '+ str(f1_r))
    # file.close()



    save_results(f1, f1_w, f1_s1, f1_s2, f1_sws, f1_r, wavelet, rus, level, features, trees)
    
    
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
    return clf1

def predecir_sj(wavelet, rus, level, features, rf, test, medias, stds,pca,index):

    print('test subject: ', test)
    print('BEFIRE TEST WAVELET')

    test_wavelet = wavelet_level(test,level,wavelet,1,medias.copy(),stds.copy())
    y_test = test.events[:,-1]
    print('AFTER TEST WAVELET')
      

    test_matrix = np.array(test_wavelet)
    
    if features != -1:
        test_matrix=pca.transform(test_matrix)
    
    y_predict = rf.predict(test_matrix)

    acc = (accuracy_score(y_test,y_predict))
    f1 = (f1_score(y_test,y_predict,average='macro'))
    #sensitivity = metrics.get_recall(y_test,y_predict)
    #specificity = metrics.get_specificity(y_test,y_predict)
    
    data = pd.DataFrame((
    {
    'True':y_test, 
    'Predict': y_predict
    }))
    data.to_csv('test_subject' + str(index) + '.csv', index=False)

    #5 CLASES
    W, S1, S2, SWS, R = multilabel_confusion_matrix(y_test, y_predict)


    acc_w, f1_w, sen_w, spe_w = metrics.get_metrics(W,1)
    acc_s1, f1_s1, sen_s1, spe_s1 = metrics.get_metrics(S1,1)
    acc_s2, f1_s2, sen_s2, spe_s2 = metrics.get_metrics(S2,1)
    acc_sws, f1_sws, sen_sws, spe_sws = metrics.get_metrics(SWS,1)
    acc_r, f1_r, sen_r, spe_r = metrics.get_metrics(R,1)

    file = open("sen_spe_SF.txt", "w")
    file.write('Data: sensitivity'+  '\n'+
    'WAKE sensitivity '+ str(sen_w)+' - specificity ' + str(spe_w)+ '\n' + 
    'STAGE 1 sensitivity '+ str(sen_s1)+ '- specificity ' + str(spe_s1) +'\n'+
    'STAGE 2 sensitivity '+ str(sen_s2)+ '- specificity ' + str(spe_s2) +  '\n'+
    'SWS semsitivity '+ str(sen_sws) + '- specificity ' + str(spe_sws) +'\n'+
    'REM sensitivity '+ str(sen_r) + '- specificity ' + str(spe_r))
    file.close()

    plot_confusion_matrix(rf, test_matrix, y_test,normalize='true')  # doctest: +SKIP
    plt.savefig('confusion_matrix_SF_'+ str(index)+'.png')  # doctest: +SKIP

    print('acc: ', acc, ' f1: ', f1)
    return acc, acc_w, acc_s1, acc_s2, acc_sws, acc_r, f1, f1_w, f1_s1, f1_s2, f1_sws, f1_r

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
    
    #epochs, subjects, ages= read_EDF_data_per_sj.get_eeg_data(0,40)

    # seed = random.random()
    # random.seed(seed)
    # random.shuffle(epochs)
    # random.seed(seed)
    # random.shuffle(subjects)
    # random.seed(seed)
    # random.shuffle(ages)

    

    epochs,train_sj = read_EDF_data_per_sj_5.get_eeg_data(0,40,0,'late')
    #print('Train epochs len: ', len(epochs))
    #test_epochs,_,_ = read_EDF_data_per_sj_5.get_eeg_data(40,61)
    #train_epochs = train_epochs[:1]
    #test_epochs = train_epochs[1:]

    #train_epochs= epochs

    cant_train = 70 * len(epochs) / 100
    cant_train= int(cant_train)

    #70 entrenamiento 30 test
    train_epochs = epochs[:cant_train]
    test_epochs = epochs[cant_train:]
    print('len test subject: ', len(test_epochs))


    #test_epochs = mne.concatenate_epochs(test_epochs)

    #test_epochs, test_sj = get_test_by_age(test_epochs,test_sj,subjects,1,ages)
    

    #PARA 3 CLASES
    data= {'fscore': ['General', 'Wake', 'Stage 1', 'Stage 2' ,' Stage SWS', 'REM']}
    
    #df = pd.DataFrame(data)
    #df.to_csv('results_MCM_SF.csv')

    # acc = np.empty(len(train_epochs))
    # f1 = np.empty(len(train_epochs))

    # acc_w = np.empty(len(train_epochs))
    # f1_w = np.empty(len(train_epochs))

    # acc_s1 = np.empty(len(train_epochs))
    # f1_s1 = np.empty(len(train_epochs))

    # acc_s2 = np.empty(len(train_epochs))
    # f1_s2 = np.empty(len(train_epochs))

    # acc_sws = np.empty(len(train_epochs))
    # f1_sws = np.empty(len(train_epochs))

    # acc_r = np.empty(len(train_epochs))
    # f1_r = np.empty(len(train_epochs))

    for w in wavelet:
        for r in RUS:
            for l in level:
                for f in n_features:
                    for t in n_trees:
                    #preparamos archivo csv para almacenar datos
                        #print('Tamaño_epochs: ', len(train_epochs), 'len test epochs: ', len(test_epochs))
                        rf = train(w, r, l, f,t, train_epochs.copy(),test_epochs)
                        

                        # for i in range(len(train_epochs)):
                        #     (acc[i], acc_w [i], acc_s1[i], acc_s2[i], acc_sws[i], acc_r[i],
                        #     f1[i], f1_w[i], f1_s1[i], f1_s2[i], f1_sws[i], f1_r[i]) = train_data(w, r, l, f, t, train_epochs.copy(),i)             
                        # save_results(f1,f1_w,f1_s1,f1_s2,f1_sws,f1_r, w,r,l,f,t)

    #print('cantidad de sujetos: ', len(train_sj))
if __name__ == '__main__':
    #se agregan todos los parametros que pueden pasarse al software cuando se llama
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--wavelet', required=True, nargs='+', type= str, default = 'coif1',help='Lista de las familias de wavelet a utilizar')
    parser.add_argument('-r', '--RUS', required = True, nargs='+',  type= int, default= 100, help='Porcentaje de reduccion Stage 2')
    parser.add_argument('-l', '--level', required= True, nargs='+',  type= int, default= 1, help='Nivel de profundidad de las wavelet')
    parser.add_argument('-f', '--n_features', required= True, nargs='+', type= int, default= -1, help='Cantidad de features reducidas por PCA')
    parser.add_argument('-t', '--n_trees', required= True, nargs='+', type= int, default= 100, help='Cantidad de features reducidas por PCA')
    
    args = parser.parse_args()

#    os.makedirs('./output/checkpoints/', exist_ok=True)
#
    main(**vars(args))