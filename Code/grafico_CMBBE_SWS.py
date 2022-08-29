import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#WAKE
sws_WM=[0.72921109, 0.28169014, 0.65948276, 0.46265938, 0.62256809, 0.34146341]

sws_SM=[0.92253521, 0.86969253, 0.80952381, 0.81795511, 0.82272727, 0.67724868]

sws_EM=[0.9048474,  0.84341085, 0.79849341, 0.82097187, 0.80764904, 0.66288952]

data= pd.DataFrame(
    {
    'WM': sws_WM, #WAVELET MODEL
    'SM' : sws_SM, #ESTADISTIC FEATURE SET
    'EM': sws_EM #EXTEND MODEL
    })
sns.set(rc={'figure.figsize':(6,12)})
sns.set_theme(style="whitegrid")
plt.ylim(-0.2, 1.2)


ax= sns.violinplot(data=data,palette=['limegreen','forestgreen','darkgreen'])
ax = sns.swarmplot(data=data, color="white")

plt.title('SWS')
plt.show()

