import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt 

wm = pd.read_csv('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/2_model_selection/para_CM_plot/WM/test_subject_1.csv')
sm = pd.read_csv('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/2_model_selection/para_CM_plot/SM/test_subject1.csv')
em = pd.read_csv('/mnt/Almacenamiento/Doctorado/Materias/Computer_Vision/Computer_vision_final/experiments/RESULTS/2_model_selection/para_CM_plot/EM/test_subject_1.csv')

print(wm)

cm_wm = pd.crosstab(wm['True'], wm['Predict'], rownames=['True'], colnames=['Predicted'],normalize=1).round(2)
cm_sm = pd.crosstab(sm['True'], sm['Predict'], rownames=['True'], colnames=['Predicted'],normalize=1).round(2)
cm_em = pd.crosstab(em['True'], em['Predict'], rownames=['True'], colnames=['Predicted'],normalize=1).round(2)

sns.set_theme(style="whitegrid",font_scale=1.5)


fig, axes = plt.subplots(1, 3, figsize=(20,5))

sns.heatmap(ax= axes[0], data=cm_wm, annot=True,cmap='viridis')
sns.heatmap(ax= axes[1], data=cm_sm, annot=True,cmap='viridis')
sns.heatmap(ax= axes[2], data=cm_em, annot=True,cmap='viridis')

fig.tight_layout()
plt.show()