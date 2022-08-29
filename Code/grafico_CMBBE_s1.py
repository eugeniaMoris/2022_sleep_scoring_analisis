import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#WAKE
s1_WM=[0.06060606, 0.01834862, 0.01923077, 0.0,         0.0591133,  0.02040816]

s1_SM=[0.24347826, 0.30357143, 0.13605442, 0.0 ,        0.11594203, 0.05970149]

s1_EM= [0.12658228, 0.10738255, 0.10526316, 0.0,         0.08205128, 0.02955665]

data= pd.DataFrame(
    {
    'WM': s1_WM, #WAVELET MODEL
    'SM' : s1_SM, #ESTADISTIC FEATURE SET
    'EM': s1_EM #EXTEND MODEL
    })

sns.set(rc={'figure.figsize':(6,12)})

sns.set_theme(style="whitegrid")
plt.ylim(-0.2, 1.2)


ax= sns.violinplot(data=data,palette=['lightblue','deepskyblue','dodgerblue'])
ax = sns.swarmplot(data=data, color="white")

plt.title('S1')
plt.show()

