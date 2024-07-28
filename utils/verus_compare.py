import numpy as np
from itertools import combinations
import pandas as pd
import os
dataset_names=['HIVE-COTE 2.0','TS-CHIEF','Hydra+MultiRocket','InceptionTime','DrCIF','TDE','STC','MultiRocket','Hydra']

pd=pd.read_csv('../results/KG-MTPResults/verus_campare.csv')
import matplotlib.pyplot as plt
a=pd['KG-MTP'].values
for i in dataset_names:
        b=pd[i].values
        diff = a-b
        win = str(sum(i > 0 for i in diff))
        draw = str(sum(i == 0 for i in diff))
        loss =str(sum(i < 0 for i in diff))

        fig,ax=plt.subplots(figsize=(8,8),dpi=300)
        Axis_line=np.linspace(*ax.get_xlim(),2)
        ax.plot(Axis_line,Axis_line,linestyle='--',linewidth=2,color='blue')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.scatter(b,a,color='blue',s=10)
        x = [1/20,1]
        y = [0,19/20]
        ax.plot(x, y, linestyle='--',color='gray')
        x = [0,  19/20]
        y = [1/20,1]
        ax.plot(x, y, linestyle='--',color='gray')
        ax.text(0.05, 0.98, 'KG-MTP is better', color='gray', ha='left', va='top')
        ax.text(0.95, 0.02, str(i)+' is better', color='gray', ha='right', va='bottom')
        ax.text(0.05, 0.5, 'win/draw/loss\n'+win+'/'+draw+'/'+loss, color='black', ha='center', va='center',
                bbox=dict(facecolor='lightyellow', edgecolor='black'))

        font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14,
        }
        font1 = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 16,
        }
        plt.title("KG-MTP vs "+str(i),font1)
        plt.xlabel(str(i),font)
        plt.ylabel("KG-MTP",font)
        plt.savefig('../results/KG-MTPResults/KG-MTP_vs_'+str(i)+'.jpg', dpi=400,format="jpg")
        # ax.legend()
        plt.show()
        plt.clf()

