import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

s2_WM=[0.77219685, 0.68170169, 0.72317881, 0.66505692, 0.66633416, 0.71614429]

s2_SM=[0.84985836, 0.77602906, 0.73761468, 0.69245073, 0.75119048, 0.7165404 ]

s2_EM= [0.82410959, 0.75893887, 0.72928666, 0.6900628,  0.71866595, 0.73735232]

data= pd.DataFrame(
    {
    'WM': s2_WM, #WAVELET MODEL
    'SM' : s2_SM, #ESTADISTIC FEATURE SET
    'EM': s2_EM #EXTEND MODEL
    })
sns.set(rc={'figure.figsize':(6,12)})
sns.set_theme(style="whitegrid")
plt.ylim(-0.2, 1.2)


ax= sns.violinplot(data=data,palette=['sandybrown','orange','darkorange'])
ax = sns.swarmplot(data=data, color="white")

plt.title('S2')
plt.show()