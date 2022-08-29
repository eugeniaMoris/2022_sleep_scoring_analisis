import numpy as np
import joblib
import read_EDF_data_per_sj_5
from scipy.stats import norm, kurtosis, skew, entropy
import pywt
import balance_data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import cross_val_score,cross_val_predict
import metrics
import collections
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import confusion_matrix_plot
import random

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


#EARLY
fs_model_E = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Early/cascade_1.joblib')
ss1_model_E = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Early/cascade_2.joblib')
ss2_model_E = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Early/ccascade_3.joblib')

mean_fs_E = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Early/medias_1.npy')
std_fs_E = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Early/std_1.npy')

mean_ss1_E = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Early/medias_2.npy')
std_ss1_E = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Early/std_2.npy')

mean_ss2_E = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Early/medias_3.npy')
std_ss2_E = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Early/std_3.npy')

#LATE
fs_model_L = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/cascade_1.joblib')
ss1_model_L = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/cascade_2.joblib')
ss2_model_L = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/ccascade_3.joblib')

mean_fs_L = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/medias_1.npy')
std_fs_L = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/std_1.npy')

mean_ss1_L = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/medias_2.npy')
std_ss1_L = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/std_2.npy')

mean_ss2_L = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/medias_3.npy')
std_ss2_L = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/std_3.npy')

epochs_E,train_sj_E = read_EDF_data_per_sj_5.get_eeg_data(0,40,1,'early')
epochs_L,train_sj_L = read_EDF_data_per_sj_5.get_eeg_data(0,40,1,'late')

cant_train = 70 * len(epochs_E) / 100
cant_train= int(cant_train)

train_epochs = epochs_E[:cant_train]
test_epochs_E = epochs_E[cant_train:]
test_epochs_L = epochs_L[cant_train:]


f1_general_EE= []
f1_w_EE = []
f1_s1_EE = []
f1_s2_EE = []
f1_sws_EE = []
f1_r_EE = []

f1_general_LE= []
f1_w_LE = []
f1_s1_LE = []
f1_s2_LE = []
f1_sws_LE = []
f1_r_LE = []

f1_general_EL= []
f1_w_EL = []
f1_s1_EL = []
f1_s2_EL = []
f1_sws_EL = []
f1_r_EL = []

f1_general_LL= []
f1_w_LL = []
f1_s1_LL = []
f1_s2_LL = []
f1_sws_LL = []
f1_r_LL = []




#TESTEO SOBRE EARLY
for t in test_epochs_E:
    y_test = t.events[:,-1]

    #MODELO EARLY
    #OBTENGO WAVELETS DE EARLY
    t_wavelet_fs = wavelet_level(t,4,'sym2',1,mean_fs_E,std_fs_E)
    t_wavelet_ss1 = wavelet_level(t,2,'coif1',1,mean_ss1_E,std_ss1_E)
    t_wavelet_ss2 = wavelet_level(t,4,'coif1',1,mean_ss2_E,std_ss2_E)

    #PREDICCION SOBRE EARLY
    predict_fs_E = fs_model_E.predict(t_wavelet_fs)
    predict_ss1_E = ss1_model_E.predict(t_wavelet_ss1)
    predict_ss2_E = ss2_model_E.predict(t_wavelet_ss2)

    final_predict_E = np.zeros(len(predict_fs_E))

    for i in range(len(predict_fs_E)):
        if predict_fs_E[i] == 1:
            final_predict_E[i]= predict_ss1_E[i]
        else:
            final_predict_E[i] = predict_ss2_E[i]

    #METRICAS EARLY
    acc = (accuracy_score(y_test,final_predict_E))
    f1 = (f1_score(y_test,final_predict_E,average='macro'))
    W, S1, S2, SWS, R = multilabel_confusion_matrix(y_test, final_predict_E)

    f1_general_EE.insert(len(f1_general_EE), f1)
    f1_w_EE.insert(len(f1_w_EE), metrics.get_fscore(W,1))
    f1_s1_EE.insert(len(f1_s1_EE), metrics.get_fscore(S1,1))
    f1_s2_EE.insert(len(f1_s2_EE), metrics.get_fscore(S2,1))
    f1_sws_EE.insert(len(f1_sws_EE), metrics.get_fscore(SWS,1))
    f1_r_EE.insert(len(f1_r_EE), metrics.get_fscore(R,1))

    #OBTENGO WAVELETS DE EARLY segun modelo late
    t_wavelet_fs = wavelet_level(t,4,'coif1',1,mean_fs_L,std_fs_L)
    t_wavelet_ss1 = wavelet_level(t,1,'coif1',1,mean_ss1_L,std_ss1_L)
    t_wavelet_ss2 = wavelet_level(t,4,'rbio1.1',1,mean_ss2_L,std_ss2_L)

    #MODELO LATE PREDICCION SOBRE EARLY
    predict_fs_L = fs_model_L.predict(t_wavelet_fs)
    predict_ss1_L = ss1_model_L.predict(t_wavelet_ss1)
    predict_ss2_L = ss2_model_L.predict(t_wavelet_ss2)

    final_predict_L = np.zeros(len(predict_fs_L))

    for i in range(len(predict_fs_L)):
        if predict_fs_L[i] == 1:
            final_predict_L[i]= predict_ss1_L[i]
        else:
            final_predict_L[i] = predict_ss2_L[i]

    #METRICAS EARLY
    acc = (accuracy_score(y_test,final_predict_L))
    f1 = (f1_score(y_test,final_predict_L,average='macro'))
    W, S1, S2, SWS, R = multilabel_confusion_matrix(y_test, final_predict_L)

    f1_general_LE.insert(len(f1_general_LE), f1)
    f1_w_LE.insert(len(f1_w_LE), metrics.get_fscore(W,1))
    f1_s1_LE.insert(len(f1_s1_LE), metrics.get_fscore(S1,1))
    f1_s2_LE.insert(len(f1_s2_LE), metrics.get_fscore(S2,1))
    f1_sws_LE.insert(len(f1_sws_LE), metrics.get_fscore(SWS,1))
    f1_r_LE.insert(len(f1_r_LE), metrics.get_fscore(R,1))

#TESTEO SOBRE LATE
for t in test_epochs_L:
    y_test = t.events[:,-1]

    #MODELO EARLY
    #OBTENGO WAVELETS DE EARLY
    t_wavelet_fs = wavelet_level(t,4,'sym2',1,mean_fs_E,std_fs_E)
    t_wavelet_ss1 = wavelet_level(t,2,'coif1',1,mean_ss1_E,std_ss1_E)
    t_wavelet_ss2 = wavelet_level(t,4,'coif1',1,mean_ss2_E,std_ss2_E)

    #PREDICCION SOBRE EARLY
    predict_fs_E = fs_model_E.predict(t_wavelet_fs)
    predict_ss1_E = ss1_model_E.predict(t_wavelet_ss1)
    predict_ss2_E = ss2_model_E.predict(t_wavelet_ss2)

    final_predict_E = np.zeros(len(predict_fs_E))

    for i in range(len(predict_fs_E)):
        if predict_fs_E[i] == 1:
            final_predict_E[i]= predict_ss1_E[i]
        else:
            final_predict_E[i] = predict_ss2_E[i]

    #METRICAS EARLY
    acc = (accuracy_score(y_test,final_predict_E))
    f1 = (f1_score(y_test,final_predict_E,average='macro'))
    W, S1, S2, SWS, R = multilabel_confusion_matrix(y_test, final_predict_E)

    f1_general_EL.insert(len(f1_general_EL), f1)
    f1_w_EL.insert(len(f1_w_EL), metrics.get_fscore(W,1))
    f1_s1_EL.insert(len(f1_s1_EL), metrics.get_fscore(S1,1))
    f1_s2_EL.insert(len(f1_s2_EL), metrics.get_fscore(S2,1))
    f1_sws_EL.insert(len(f1_sws_EL), metrics.get_fscore(SWS,1))
    f1_r_EL.insert(len(f1_r_EL), metrics.get_fscore(R,1))

    #OBTENGO WAVELETS DE EARLY segun modelo late
    t_wavelet_fs = wavelet_level(t,4,'coif1',1,mean_fs_L,std_fs_L)
    t_wavelet_ss1 = wavelet_level(t,1,'coif1',1,mean_ss1_L,std_ss1_L)
    t_wavelet_ss2 = wavelet_level(t,4,'rbio1.1',1,mean_ss2_L,std_ss2_L)

    #MODELO LATE PREDICCION SOBRE EARLY
    predict_fs_L = fs_model_L.predict(t_wavelet_fs)
    predict_ss1_L = ss1_model_L.predict(t_wavelet_ss1)
    predict_ss2_L = ss2_model_L.predict(t_wavelet_ss2)

    final_predict_L = np.zeros(len(predict_fs_L))

    for i in range(len(predict_fs_L)):
        if predict_fs_L[i] == 1:
            final_predict_L[i]= predict_ss1_L[i]
        else:
            final_predict_L[i] = predict_ss2_L[i]

    #METRICAS EARLY
    acc = (accuracy_score(y_test,final_predict_L))
    f1 = (f1_score(y_test,final_predict_L,average='macro'))
    W, S1, S2, SWS, R = multilabel_confusion_matrix(y_test, final_predict_L)

    f1_general_LL.insert(len(f1_general_LL), f1)
    f1_w_LL.insert(len(f1_w_LL), metrics.get_fscore(W,1))
    f1_s1_LL.insert(len(f1_s1_LL), metrics.get_fscore(S1,1))
    f1_s2_LL.insert(len(f1_s2_LL), metrics.get_fscore(S2,1))
    f1_sws_LL.insert(len(f1_sws_LL), metrics.get_fscore(SWS,1))
    f1_r_LL.insert(len(f1_r_LL), metrics.get_fscore(R,1))

#fscore_EE = [f1_general_E,f1_w_E,f1_s1_E,f1_s2_E,f1_sws_E,f1_r_E]
#fscore_LE = [f1_general_L,f1_w_L,f1_s1_L,f1_s2_L,f1_sws_L,f1_r_L]

general = [f1_general_EE,f1_general_EL, f1_general_LE,f1_general_LL]
wake = [f1_w_EE,f1_w_EL, f1_w_LE,f1_w_LL]
s1 = [f1_s1_EE,f1_s1_EL, f1_s1_LE,f1_s1_LL]
s2 = [f1_s2_EE,f1_s2_EL, f1_s2_LE,f1_s2_LL]
sws = [f1_sws_EE,f1_sws_EL, f1_sws_LE,f1_sws_LL]
rem = [f1_r_EE,f1_r_EL, f1_r_LE,f1_r_LL]

#fscore=[fscore_EE,fscore_LE]
general = pd.DataFrame((
    {
    'EE': f1_general_EE, 
    'EL' : f1_general_EL,
    'LE': f1_general_LE,
    'LL': f1_general_LL
    }))

wake = pd.DataFrame((
    {
    'EE': f1_w_EE, 
    'EL' : f1_w_EL,
    'LE': f1_w_LE,
    'LL': f1_w_LL
    }))

s1 = pd.DataFrame((
{
'EE': f1_s1_EE, 
'EL' : f1_s1_EL,
'LE': f1_s1_LE,
'LL': f1_s1_LL
}))

s2 = pd.DataFrame((
    {
    'EE': f1_s2_EE, 
    'EL' : f1_s2_EL,
    'LE': f1_s2_LE,
    'LL': f1_s2_LL
    }))

sws = pd.DataFrame((
    {
    'EE': f1_sws_EE, 
    'EL' : f1_sws_EL,
    'LE': f1_sws_LE,
    'LL': f1_sws_LL
    }))

rem = pd.DataFrame((
    {
    'EE': f1_r_EE, 
    'EL' : f1_r_EL,
    'LE': f1_r_LE,
    'LL': f1_r_LL
    }))

general.to_csv('general.csv', index=False)

tuples = list(
    zip(
        *[
            ["Early", "Late"],
            ["Early", "Late"],
        ]
    )
)


index = pd.MultiIndex.from_tuples(tuples, names=["Train", "Test"])

#df = pd.DataFrame(general, index=index, columns=["14", "15","16","17","18","19"])
#print(df)

plt.rcParams.update({'font.size': 12})

fig, axes = plt.subplots(1, 6, sharex=True, figsize=(16,6))

#fig.suptitle('3 rows x 4 columns axes with no data')

sns.set_theme(style="whitegrid")


axes[0].grid(True)

sns.barplot(ax=axes[0],data=general, palette='tab20')
axes[0].set_title('General')
axes[0].set_ylim([0, 1])
axes[0].set_ylabel("F-score")

axes[1].grid(True)
sns.barplot(ax=axes[1],data=wake, palette='tab20')
axes[1].set_title('Wake')
axes[1].set_ylim([0, 1])

axes[2].grid(True)
sns.barplot(ax=axes[2],data=s1, palette='tab20')
axes[2].set_title('Stage 1')
axes[2].set_ylim([0, 1])

axes[3].grid(True)
sns.barplot(ax=axes[3],data=s2, palette='tab20')
axes[3].set_title('Stage 2')
axes[3].set_ylim([0, 1])

axes[4].grid(True)
sns.barplot(ax=axes[4],data=sws, palette='tab20')
axes[4].set_title('SWS')
axes[4].set_ylim([0, 1])


axes[5].grid(True)
sns.barplot(ax=axes[5],data=rem, palette='tab20')
axes[5].set_title('REM')
axes[5].set_ylim([0, 1])


fig.tight_layout()






#sns.violinplot(data=data,palette=['pink','palevioletred','skyblue','deepskyblue','coral','orangered','limegreen','forestgreen','goldenrod','darkgoldenrod'])
#sns.swarmplot(data=data, color="white")
#sns.violinplot(data)
plt.show()