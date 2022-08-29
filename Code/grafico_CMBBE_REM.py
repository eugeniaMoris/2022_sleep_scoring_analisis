import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#WAKE

rem_WM=[0.70655271, 0.54857143, 0.43434343, 0.44927536, 0.09252669, 0.50309278]

rem_SM= [0.84938272, 0.5694051,  0.45779221, 0.43223443, 0.62553191, 0.49904031]

rem_EM=[0.81038961, 0.5625,     0.42546064, 0.43223443, 0.42896936, 0.5243129 ]

data= pd.DataFrame(
    {
    'WM': rem_WM, #WAVELET MODEL
    'SM' : rem_SM, #ESTADISTIC FEATURE SET
    'EM': rem_EM #EXTEND MODEL
    })
sns.set(rc={'figure.figsize':(6,12)})
sns.set_theme(style="whitegrid")
plt.ylim(-0.2, 1.2)

ax= sns.violinplot(data=data,palette=['tan','peru','sienna'])
ax = sns.swarmplot(data=data, color="white")

plt.title('REM')
plt.show()
