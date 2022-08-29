import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import pandas as pd

g1g1 = pd.read_csv('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Age_csv/G4G1_cascade_data.csv')
g1g2 = pd.read_csv('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Age_csv/G4G2_cascade_data.csv')
g1g3 = pd.read_csv('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Age_csv/G4G3_cascade_data.csv')
g1g4 = pd.read_csv('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Age_csv/G4G4_cascade_data.csv')


g_g1= g1g1['G']
g_g2= g1g2['G']
g_g3= g1g3['G']
g_g4= g1g4['G']

W_g1= g1g1['W']
W_g2= g1g2['W']
W_g3= g1g3['W']
W_g4= g1g4['W']

s1_g1= g1g1['S1']
s1_g2= g1g2['S1']
s1_g3= g1g3['S1']
s1_g4= g1g4['S1']

s2_g1= g1g1['S2']
s2_g2= g1g2['S2']
s2_g3= g1g3['S2']
s2_g4= g1g4['S2']

sws_g1= g1g1['SWS']
sws_g2= g1g2['SWS']
sws_g3= g1g3['SWS']
sws_g4= g1g4['SWS']

r_g1= g1g1['REM']
r_g2= g1g2['REM']
r_g3= g1g3['REM']
r_g4= g1g4['REM']
          

data_g= [g_g1,g_g2,g_g3,g_g4]
data_w=[W_g1,W_g2,W_g3,W_g4]
data_s1=[s1_g1,s1_g2,s1_g3,s1_g4]
data_s2=[s2_g1,s2_g2,s2_g3,s2_g4]
data_sws=[sws_g1,sws_g2,sws_g3,sws_g4]
data_r=[r_g1,r_g2,r_g3,r_g4]

x_ticks_labels = ['G1','G2','G3','G4']

colors = ["#76A5AF", "#6fA8DC","#8E7CC3","#C27BA0"]
palette = sns.set_palette(sns.color_palette(colors))

sns.set_theme(style="whitegrid",font_scale=1.5)

fig, axes = plt.subplots(1, 6, sharex=True, figsize=(23,5),sharey=True)
plt.subplots_adjust(wspace=.0)


sns.barplot(ax= axes[0],data=data_g, palette=["#76A5AF", "#6fA8DC","#8E7CC3","#C27BA0"])
sns.barplot(ax= axes[1],data=data_w, palette=["#76A5AF", "#6fA8DC","#8E7CC3","#C27BA0"])
sns.barplot(ax= axes[2],data=data_s1, palette=["#76A5AF", "#6fA8DC","#8E7CC3","#C27BA0"])
sns.barplot(ax= axes[3],data=data_s2, palette=["#76A5AF", "#6fA8DC","#8E7CC3","#C27BA0"])
sns.barplot(ax= axes[4],data=data_sws, palette= ["#76A5AF", "#6fA8DC","#8E7CC3","#C27BA0"])
sns.barplot(ax= axes[5],data=data_r, palette= ["#76A5AF", "#6fA8DC","#8E7CC3","#C27BA0"])

axes[0].set_ylabel("G4-model F-score")
axes[0].set_title("General")
axes[1].set_title("W")
axes[2].set_title("S1")
axes[3].set_title("S2")
axes[4].set_title("SWS")
axes[5].set_title("REM")

axes[0].set_ylim([0, 1])
axes[1].set_ylim([0, 1])
axes[2].set_ylim([0, 1])
axes[3].set_ylim([0, 1])
axes[4].set_ylim([0, 1])
axes[5].set_ylim([0, 1])

axes[0].set_xticklabels(x_ticks_labels)




# axes[0].spines['right'].set_visible(False)
# axes[1].spines['left'].set_visible(False)
# axes[1].spines['right'].set_visible(False)
# axes[2].spines['left'].set_visible(False)
# axes[2].spines['right'].set_visible(False)
# axes[3].spines['left'].set_visible(False)
# axes[3].spines['right'].set_visible(False)
# axes[4].spines['left'].set_visible(False)
# axes[4].spines['right'].set_visible(False)
# axes[5].spines['left'].set_visible(False)

sns.despine(fig,left=True,right=True)
#sns.despine(trim=True)




fig.tight_layout()
plt.show()