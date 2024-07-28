import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

from utils.tools import flip_matrix
import scipy.stats as stats
import pandas as pd

dataset_names=['KG-MTP','Hydra+MultiRocket','MultiRocket','Hydra','TS-CHIEF','InceptionTime','DrCIF','TDE','STC','HIVE-COTE 2.0']
pd=pd.read_csv('../results/KG-MTPResults/verus_campare.csv')
wins=np.zeros([10,10],dtype=np.int)
draws=np.zeros([10,10],dtype=np.int)
losses=np.zeros([10,10],dtype=np.int)
p_values=np.zeros([10,10])

for i in range(len(dataset_names)):
    # p_values = []
    for j in range(i+1,len(dataset_names)):
        diff=pd[dataset_names[i]]-pd[dataset_names[j]]
        p_value=stats.wilcoxon(pd[dataset_names[i]],pd[dataset_names[j]],zero_method='pratt').pvalue
        # p_values.append(p_value)
        p_values[i,j]=round(p_value,3)
        win=sum(i > 0 for i in diff)
        draw=sum(i == 0 for i in diff)
        loss=sum(i < 0 for i in diff)
        wins[i,j]=win
        draws[i, j] = draw
        losses[i, j] = loss

p_values=flip_matrix(p_values.T)
wins=flip_matrix(wins.T)
draws=flip_matrix(draws.T)
losses=flip_matrix(losses.T)
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)

cmap = LinearSegmentedColormap.from_list('mycmap', ['white', 'orange'])

im = ax.imshow(p_values,cmap=cmap)
# Draw horizontal grid lines
for i in range(p_values.shape[0]):
    ax.hlines(i + 0.5, -0.5, i + 0.5, colors='black', linewidths=1)

# Draw vertical grid lines
for i in range(p_values.shape[1]):
    ax.vlines(i + 0.5, i + 0.5, p_values.shape[0] - 0.5, colors='black', linewidths=1)
# Add text inside each box
for i in range(p_values.shape[0]):
    for j in range(i):
        if p_values[i, j] >= 0.004:
            text = ax.text(j, i, str("{:.3f}".format(p_values[i, j]))+"\n"+str(wins[i,j])+'/'+str(draws[i,j])+'/'+str(losses[i,j]), ha="center", va="center", color="black",fontweight='bold',fontsize=18)
        else:
            text = ax.text(j, i,str("{:.3f}".format(p_values[i, j]))+"\n"+str(wins[i,j])+'/'+str(draws[i,j])+'/'+str(losses[i,j]), ha="center", va="center", color="black",fontsize=18)

plt.xticks(np.arange(len(dataset_names)), np.flip(dataset_names),rotation=90,fontweight='bold',fontsize=15)
plt.yticks(np.arange(len(dataset_names)), np.flip(dataset_names),fontweight='bold',fontsize=15)

cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20)

cbar.ax.text(0.5, 1.0, 'p-value', ha='center', va='bottom',fontweight='bold',fontsize=20)

plt.show()

