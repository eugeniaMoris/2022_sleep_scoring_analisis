import numpy as np
def get_metrics(cm,b=1):
    beta = b
    #accuracy = get_accuracy(cm)
    #precision = get_precision(cm)
    sensitivity = get_recall(cm)
    specificity = get_specificity(cm)

    #fscore = get_fscore(cm, beta)
    #print('accuracy: ', accyracy, 'precision: ', precision, ' recall: ', recall , ' fscore: ', fscore)
    return 1, 1, sensitivity, specificity

def get_accuracy(cm):
    return (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])

def get_precision(cm):
    return cm[1,1]/(cm[1,1]+cm[0,1]) #false positive

def get_recall(cm):
    return cm[1,1]/(cm[1,1]+cm[1,0]) #false negative

def get_specificity(cm):
    return cm[0,0] / (cm[0,0]+cm[0,1])

def get_fscore(cm,beta):
    return ((beta**2) + 1)*cm[1,1]/(((beta**2) + 1)*cm[1,1]+(beta**2)*cm[1,0]+cm[0,1])


#cm = np.array([[710, 44],[72, 180]])
#get_metrics(cm,1)