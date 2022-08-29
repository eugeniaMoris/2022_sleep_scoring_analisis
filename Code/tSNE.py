from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_tSNE (x_train, y_train):

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    sns.set(rc={'figure.figsize':(11.7,8.27)})
    palette = sns.color_palette("deep", 3)
    tsne = TSNE()
    X_embedded = tsne.fit_transform(x_train)

    sns.scatterplot(x= X_embedded[:,0], y= X_embedded[:,1], hue=y_train, legend='full', palette=palette)
    #xs = X_embedded[:,0]
    #ys = X_embedded[:,1]
    #zs = X_embedded[:,2]
    #ax.scatter(xs, ys, zs, c=y_train)
    
    plt.show()