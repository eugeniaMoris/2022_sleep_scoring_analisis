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

#G1
fs_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/3_cascada/CASCADA1/cascade_1.joblib')
ss1_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/3_cascada/CASCADA2/cascade_2.joblib')
ss2_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/3_cascada/CASCADA3/ccascade_3.joblib')

mean_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/3_cascada/CASCADA1/medias_1.npy')
std_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/3_cascada/CASCADA1/std_1.npy')

mean_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/3_cascada/CASCADA2/medias_2.npy')
std_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/3_cascada/CASCADA2/std_2.npy')

mean_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/3_cascada/CASCADA3/medias_3.npy')
std_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/3_cascada/CASCADA3/std_3.npy')

#G2
# fs_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g2/cascade_1.joblib')
# ss1_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g2/cascade_2.joblib')
# ss2_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g2/ccascade_3.joblib')

# mean_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g2/medias_1.npy')
# std_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g2/std_1.npy')

# mean_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g2/medias_2.npy')
# std_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g2/std_2.npy')

# mean_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g2/medias_3.npy')
# std_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g2/std_3.npy')

# #G3
# fs_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g3/cascade_1.joblib')
# ss1_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g3/cascade_2.joblib')
# ss2_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g3/ccascade_3.joblib')

# mean_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g3/medias_1.npy')
# std_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g3/std_1.npy')

# mean_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g3/medias_2.npy')
# std_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g3/std_2.npy')

# mean_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g3/medias_3.npy')
# std_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g3/std_3.npy')

#G4
# fs_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g4/cascade_1.joblib')
# ss1_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g4/cascade_2.joblib')
# ss2_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g4/ccascade_3.joblib')

# mean_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g4/medias_1.npy')
# std_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g4/std_1.npy')

# mean_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g4/medias_2.npy')
# std_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g4/std_2.npy')

# mean_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g4/medias_3.npy')
# std_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/4_AGES/cascada_g4/std_3.npy')

#EARLY
# fs_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/cascade_1.joblib')
# ss1_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/cascade_2.joblib')
# ss2_model = joblib.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/ccascade_3.joblib')

# mean_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/medias_1.npy')
# std_fs = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/std_1.npy')

# mean_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/medias_2.npy')
# std_ss1 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/std_2.npy')

# mean_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/medias_3.npy')
# std_ss2 = np.load('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/5_EarlyLate/Late/std_3.npy')

epochs,train_sj = read_EDF_data_per_sj_5.get_eeg_data(0,40,0,'late')
#epochs,train_sj = read_EDF_data_per_sj_5.get_eeg_data(41,60,0,'late')
#epochs,train_sj = read_EDF_data_per_sj_5.get_eeg_data(61,80,0,'late')
#epochs,train_sj = read_EDF_data_per_sj_5.get_eeg_data(81,140,0,'late')

path= '/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Age_csv/review_recall_G1G1_cascade_data.csv'

cant_train = 70 * len(epochs) / 100
cant_train= int(cant_train)

train_epochs = epochs[:cant_train]
test_epochs = epochs[cant_train:]

f1_general= []
f1_w = []
f1_s1 = []
f1_s2 = []
f1_sws = []
f1_r = []




for t in test_epochs:
    y_test = t.events[:,-1]

    t_wavelet_fs = wavelet_level(t,4,'dmey',1,mean_fs,std_fs)
    
    #tss1 = balance_data.balance_epochs(t.copy(),50)
    t_wavelet_ss1 = wavelet_level(t,1,'haar',1,mean_ss1,std_ss1)

    #tss2 = balance_data.balance_epochs(t.copy(),20)
    t_wavelet_ss2 = wavelet_level(t,4,'sym2',1,mean_ss2,std_ss2)


    predict_fs = fs_model.predict(t_wavelet_fs)
    print('first step: ' , collections.Counter(predict_fs))
    predict_ss1 = ss1_model.predict(t_wavelet_ss1)
    print('second step 1: ' , collections.Counter(predict_ss1))
    predict_ss2 = ss2_model.predict(t_wavelet_ss2)
    print('second step 2: ' , collections.Counter(predict_ss2))

    # result = fs_model.score(t_wavelet_fs, y_test)




    # predict_fs = cross_val_predict(fs_model, t_wavelet_fs, y_test)
    # predict_ss1 = cross_val_predict(ss1_model, t_wavelet_ss1, y_test)
    # predict_ss2 = cross_val_predict(ss2_model, t_wavelet_ss2, y_test)

    final_predict = np.zeros(len(predict_fs))
    #print('final predict: ' , final_predict.shape)

    for i in range(len(predict_fs)):
        if predict_fs[i] == 1:
            final_predict[i]= predict_ss1[i]
        else:
            final_predict[i] = predict_ss2[i]
    #print('final predict: ' , final_predict)
    acc = (accuracy_score(y_test,final_predict))
    print('ACCURACY :' + str(acc))
    f1 = (f1_score(y_test,final_predict,average='macro'))
    W, S1, S2, SWS, R = multilabel_confusion_matrix(y_test, final_predict)

    # class_names = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3/4', 'Sleep stage R']
    # confusion_matrix_plot.plot_confusion_matrix(y_test, final_predict, classes=[0,1,2,3,4], normalize=True,
    #                   title='Normalized confusion matrix')
    # plt.show()

    # f1_general.insert(len(f1_general), f1)
    # f1_w.insert(len(f1_w), metrics.get_fscore(W,1))
    # f1_s1.insert(len(f1_s1), metrics.get_fscore(S1,1))
    # f1_s2.insert(len(f1_s2), metrics.get_fscore(S2,1))
    # f1_sws.insert(len(f1_sws), metrics.get_fscore(SWS,1))
    # f1_r.insert(len(f1_r), metrics.get_fscore(R,1))

    f1_general.insert(len(f1_general), f1)
    f1_w.insert(len(f1_w), metrics.get_recall(W))
    f1_s1.insert(len(f1_s1), metrics.get_recall(S1))
    f1_s2.insert(len(f1_s2), metrics.get_recall(S2))
    f1_sws.insert(len(f1_sws), metrics.get_recall(SWS))
    f1_r.insert(len(f1_r), metrics.get_recall(R))

# wake_SM= [0.75102041, 0.83341671, 0.43369735, 0.63026521, 0.68119891, 0.4       ]
# s1_SM= [0.24347826, 0.30357143, 0.13605442, 0.0,         0.11594203, 0.05970149]
# s2_SM= [0.84985836, 0.77602906, 0.73761468, 0.69245073, 0.75119048, 0.7165404 ]
# sws_SM= [0.92253521, 0.86969253, 0.80952381, 0.81795511, 0.82272727, 0.67724868]
# rem_SM= [0.84938272, 0.5694051,  0.45779221, 0.43223443, 0.62553191, 0.49904031]


fscore = [f1_general,f1_w,f1_s1,f1_s2,f1_sws,f1_r]
data = pd.DataFrame((
    {
    'G':f1_general, 
    'W': f1_w, 
    'S1' : f1_s1,
    'S2': f1_s2,
    'SWS': f1_sws,
    'REM': f1_r
    }))

print( ' general ', np.mean(f1_general))
print(  'wake ', np.mean(f1_w))
print( 'S1 ',  np.mean(f1_s1))
print(  'S2 ', np.mean(f1_s2))
print(  'SWS ',np.mean(f1_sws))
print( 'REM ',np.mean(f1_r))

print(data)
data.to_csv(path, index=False)


sns.set_theme(style="whitegrid")
sns.violinplot(data=data,palette=['pink','palevioletred','skyblue','deepskyblue','coral','orangered','limegreen','forestgreen','goldenrod','darkgoldenrod'])
sns.swarmplot(data=data, color="white")
#sns.violinplot(data)
plt.show()