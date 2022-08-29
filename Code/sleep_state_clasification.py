#Clasificacion de estapas de sueño utilizando wivelets y el dataset de sleep-edf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
import metrics_2classes
import read_EDF_data 
import balance_data
import matplotlib.pyplot as plt
import numpy as np
import pywt
import mne

epochs = read_EDF_data.merge_data()
epochs = balance_data.balance_epochs(epochs,20)
print(epochs)
labels = epochs.events[:, -1]
#print('labels: ', labels.shape)
#print( pywt.wavelist(family=None, kind='discrete'))
#coef_final = np.empty((labels.shape))
freq_final= []
coef_final=[]
datos = []
scales = np.arange(1, 1500)
for epoch in epochs:
    #datos.append(epoch[0,:])#tomo el primer canal
    #coeff, freqs = pywt.dwt(epoch[0,:],'bior1.1')#discrete wavelets
    #coeff, freqs = pywt.dwt(epoch[0,:],'coif1')#discrete wavelets
    #coeff, freqs = pywt.dwt(epoch[0,:],'db12')#discrete wavelets
    #coeff, freqs = pywt.dwt(epoch[0,:],'haar')#discrete wavelets
    #coeff, freqs = pywt.dwt(epoch[0,:],'rbio1.1')#discrete wavelets
    #coeff, freqs = pywt.dwt(epoch[0,:],'sym2')#discrete wavelets

    #CA, CD = pywt.dwt(epoch[0,:],'dmey')#discrete wavelets
    #CA, CD1, CD2 = pywt.wavedec(epoch[0,:],'dmey',level=2)
    #CA, CD1, CD2, CD3 = pywt.wavedec(epoch[0,:],'dmey',level=3)
    CA, CD1, CD2, CD3, CD4 = pywt.wavedec(epoch[0,:],'dmey',level=4)
    #CA, CD1, CD2, CD3, CD4, CD5 = pywt.wavedec(epoch[0,:],'dmey',level=5)

    #coeff=np.append(coeff,coeff_a)
    #print('shapes:' , CA.shape, CD1.shape,CD2.shape, CD3.shape, CD4.shape)
    CA = np.append(CA,CD1)
    CA = np.append(CA,CD2)
    CA = np.append(CA,CD3)
    CA = np.append(CA,CD4)
    #print('despues: ',CA.shape)
    #coeff=coeff.flatten()
    coef_final.append(CA)

coef_final = np.array(coef_final)

print('cant coheficientes: ', len(coef_final))

#separo datos en entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(coef_final,labels,test_size=0.1, stratify=labels,shuffle=False)
#x_train, x_test, y_train, y_test = train_test_split(datos,labels,test_size=0.1, stratify=labels)
print('shape: ',x_train.shape)

#skf = StratifiedShuffleSplit()
skf = StratifiedKFold(n_splits=10)

Accuracy = np.empty(skf.n_splits)
F1_score = np.empty(skf.n_splits)    
Precision = np.empty(skf.n_splits)    
Recall = np.empty(skf.n_splits)

accuracy_s1 = np.empty(skf.n_splits)
precision_s1 = np.empty(skf.n_splits)
recall_s1 = np.empty(skf.n_splits)
fscore_s1 = np.empty(skf.n_splits)

accuracy_s2 = np.empty(skf.n_splits)
precision_s2 = np.empty(skf.n_splits)
recall_s2 = np.empty(skf.n_splits)
fscore_s2 = np.empty(skf.n_splits)

accuracy_s3 = np.empty(skf.n_splits)
precision_s3 = np.empty(skf.n_splits)
recall_s3 = np.empty(skf.n_splits)
fscore_s3 = np.empty(skf.n_splits)


pos=0

#print('x_train shape: ', x_train.type)
for train_index, validation_index in skf.split(x_train,y_train):

    #print('train_index: ', train_index, ' validation_index: ', validation_index)
    x, x_valid = x_train[train_index], x_train[validation_index]
    y, y_valid = y_train[train_index], y_train[validation_index]

    #x,y = balance_one_to_one(x, y)

    print('Train shape: ', y.shape, ' Test shape: ', y_valid.shape)
    clf = RandomForestClassifier()
    clf.fit(x,y)

    y_predict = clf.predict(x_valid)

    acc = (accuracy_score(y_valid,y_predict))
    f1 = (f1_score(y_valid,y_predict,average='macro'))
    pres = (precision_score(y_valid,y_predict,average='macro'))
    rec = (recall_score(y_valid,y_predict,average='macro'))

    Accuracy[pos]=acc
    F1_score[pos]=f1
    Precision[pos]=pres
    Recall[pos]= rec

    S1, S2, S3 = multilabel_confusion_matrix(y_valid, y_predict)

    accuracy_s1[pos], precision_s1[pos], recall_s1[pos], fscore_s1[pos] = metrics_2classes.get_metrics(S1,1)
    accuracy_s2[pos], precision_s2[pos], recall_s2[pos], fscore_s2[pos] = metrics_2classes.get_metrics(S2,1)
    accuracy_s3[pos], precision_s3[pos], recall_s3[pos], fscore_s3[pos] = metrics_2classes.get_metrics(S3,1)
    pos = pos + 1
    #print('prediccion shape: ', y_predict.shape)
    print("Accuracy: %.2f%%" % acc)
    print("F1_Score: %.2f%%" % f1)
    #print("Precision: %.2f%%" % pres)
    #print("Recall: %.2f%%" % rec)
    print("Confusion Matrix: ")
    print(confusion_matrix(y_valid,y_predict))

print ('Accuracy- mean: ', np.mean(Accuracy), ' std: ', np.std(Accuracy))
print ('F1 Score - mean: ', np.mean(F1_score), ' std: ', np.std(F1_score))
print ('Precision- mean: ', np.mean(Precision), ' std: ', np.std(Precision))
print ('Recall- mean: ', np.mean(Recall), ' std: ', np.std(Recall))

print ('Stage 1: Accuracy- ', np.mean(accuracy_s1), ' Fscore: ', np.mean(fscore_s1), ' Precision- ', np.mean(precision_s1), ' Recall-', np.mean(recall_s1))
print ('Stage 2: Accuracy- ', np.mean(accuracy_s2), ' Fscore: ', np.mean(fscore_s2), ' Precision- ', np.mean(precision_s2), ' Recall-', np.mean(recall_s2))
print ('Stage 3: Accuracy- ', np.mean(accuracy_s3), ' Fscore: ', np.mean(fscore_s3), ' Precision- ', np.mean(precision_s3), ' Recall-', np.mean(recall_s3))

#print(Accuracy)
#print(F1_score)
#print(Precision)
#print(Recall)

#y_test_predict = clf.predict(x_test)
#print("Accuracy: %.2f%%" % accuracy_score(y_test,y_test_predict))
#print("F1_Score: %.2f%%" %f1_score(y_test,y_test_predict,average='macro'))
#print("Precision: %.2f%%" %precision_score(y_test,y_test_predict,average='macro'))
#print("Recall: %.2f%%" % recall_score(y_test,y_test_predict,average='macro'))
#print("Confusion Matrix: ")
#print(confusion_matrix(y_test,y_test_predict))



    





