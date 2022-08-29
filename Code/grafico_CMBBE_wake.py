import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#WAKE
wake_WM=[0.78148148, 0.78729548, 0.46941679, 0.58832449, 0.4244186,  0.44242424]

wake_SM=[0.75102041, 0.83341671, 0.43369735, 0.63026521, 0.68119891, 0.4       ]

wake_EM=[0.75435203, 0.81692913, 0.4088748,  0.61682243, 0.45695364, 0.44854071]

data= pd.DataFrame(
    {
    'WM': wake_WM, #WAVELET MODEL
    'SM' : wake_SM, #ESTADISTIC FEATURE SET
    'EM': wake_EM #EXTEND MODEL
    })

sns.set(rc={'figure.figsize':(6,12)})

sns.set_theme(style="whitegrid")
plt.ylim(-0.2, 1.2)


ax= sns.violinplot(data=data,palette=['lightpink','hotpink','mediumvioletred'])
ax = sns.swarmplot(data=data, color="white")

plt.title('Wake')
plt.show()