import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

stage1_MCM=[0.31914894, 0.45882353, 0.21818182, 0.13513514, 0.66511628, 0.50803859]
stage2_MCM=[0.91312867, 0.89822718, 0.75271565, 0.68316832, 0.81240064, 0.89836588]
stageSWS_MCM=[0.892261,   0.8957529,  0.6712963,  0.62695925, 0.91170825, 0.74654378]

stage1_CM1=[0.45283019, 0.51401869, 0.46511628, 0.16666667, 0.60816327, 0.56725146]
stage2_CM1=[0.92952381, 0.8923476, 0.81677977, 0.7991121,  0.78667724, 0.91513492]
stageSWS_CM1=[0.90878939, 0.90756303, 0.76363636, 0.70512821, 0.91930541, 0.76619718]

stage1_CM2=[0.32467532, 0.52040816, 0.31219512, 0.325, 0.66519824, 0.63157895]
stage2_CM2=[0.91050584, 0.90421456, 0.82672975, 0.77382319, 0.84306293, 0.89806026]
stageSWS_CM2=[0.91156463, 0.88700565, 0.76217765, 0.69924812, 0.89655172, 0.78338279]

stage1_CM3=[0.13333333, 0.24778761, 0.05769231, 0.0,         0.59405941, 0.12807882]
stage2_CM3=[0.92986699, 0.89321534, 0.87941022, 0.83247156, 0.82926829, 0.9015919 ]
stageSWS_CM3=[0.8877193,  0.85196375, 0.76480541, 0.71561531, 0.85054945, 0.64808362]

stage1_SF= [0.35658915, 0.36440678, 0.30508475, 0.10126582, 0.59471366, 0.48351648]
stage2_SF= [0.9178618,  0.85422343, 0.83382267, 0.83136412, 0.79552239, 0.8957498 ]
stageSWS_SF= [0.92753623, 0.89817232, 0.81526718, 0.75512195, 0.89957265, 0.81313131]

stage1_MCM2=[0.4,        0.53503185, 0.36649215, 0.11267606, 0.66334165, 0.56737589]
stage2_MCM2=[0.92436975, 0.88947717, 0.77480916, 0.71951952, 0.82461538, 0.91282051]
stageSWS_MCM2=[0.88821752, 0.86533666, 0.69554753, 0.65365854, 0.90962099, 0.7597254 ]

#MCM = Multiclass model
#CM = cascade model
#SF = Simpler features


# data= pd.DataFrame(
#     {
#     'S1: RFS': stage1_SF,
#     'S1: MCM' : stage1_MCM,
#     'S1: MCM2' : stage1_MCM2,
#     'S1: CM1' : stage1_CM1,
#     'S1: CM2': stage1_CM2,
#     'S1: CM3': stage1_CM3,

#     'S2: RFS': stage2_SF,
#     'S2: MCM' : stage2_MCM,
#     'S2: MCM2' : stage2_MCM2,
#     'S2: CM1' : stage2_CM1,
#     'S2: CM2': stage2_CM2,
#     'S2: CM3': stage2_CM3,

#     'SWS: RFS': stageSWS_SF,
#     'SWS: MCM' : stageSWS_MCM,
#     'SWS: MCM2' : stageSWS_MCM2,
#     'SWS: CM1' : stageSWS_CM1,
#     'SWS: CM2': stageSWS_CM2,
#     'SWS: CM3': stageSWS_CM3

#     })

data= pd.DataFrame(
    {
    'RFS': stage1_SF,
    'MCM' : stage1_MCM,
    'MCM2' : stage1_MCM2,
    'CM1' : stage1_CM1,
    'CM2': stage1_CM2,
    'CM3': stage1_CM3

    })

data2= pd.DataFrame(
    {
    'RFS': stage2_SF,
    'MCM' : stage2_MCM,
    'MCM2' : stage2_MCM2,
    'CM1' : stage2_CM1,
    'CM2': stage2_CM2,
    'CM3': stage2_CM3

    })

data3= pd.DataFrame(
    {
    'RFS': stageSWS_SF,
    'MCM' : stageSWS_MCM,
    'MCM2' : stageSWS_MCM2,
    'CM1' : stageSWS_CM1,
    'CM2': stageSWS_CM2,
    'CM3': stageSWS_CM3
    })

# data= pd.DataFrame(
#     {'Stage 1' : stage1_less,
#     'Stage 2 ' : stage2_less,
#     'Stage 3 ' : stage3_less
#     })
sns.set_theme(style="whitegrid")
plt.ylim(-0.5, 1.1)


ax= sns.violinplot(data=data,palette='YlOrBr')
ax = sns.swarmplot(data=data, color="white")
plt.title('Stage 1')
plt.show()

plt.ylim(-0.5, 1.1)
ax1= sns.violinplot(data=data2,palette='crest')
ax1 = sns.swarmplot(data=data2, color="white")
plt.title('Stage 2')
plt.show()

plt.ylim(-0.5, 1.1)
ax2= sns.violinplot(data=data3,palette='flare')
ax2 = sns.swarmplot(data=data3, color="white")
plt.title('Stage SWS')
plt.show()
