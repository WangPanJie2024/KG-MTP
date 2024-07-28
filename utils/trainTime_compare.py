import numpy as np
from itertools import combinations
import pandas as pd
import math
math.log2(0.132364213)
import os
# dataset_names=['HIVE-COTE 2.0','TS-CHIEF','Hydra+MultiRocket','InceptionTime','DrCIF','TDE','STC','MultiRocket',]
dataset_names=['MultiRocket']


pd=pd.read_csv('../results/traintime.csv')
import matplotlib.pyplot as plt
a=pd['KG-MTP'].values
for i in dataset_names:
        b=pd[i].values
        # diff = a-b
        # win = str(sum(i > 0 for i in diff))
        # draw = str(sum(i == 0 for i in diff))
        # loss =str(sum(i < 0 for i in diff))

        fig,ax=plt.subplots(figsize=(8,8),dpi=300)
        ax.scatter(b, a, color='blue', s=10)
        ax.set_xscale('log',base=3)
        ax.set_yscale('log',base=3)
        ax.set_xlim(left=np.float32(1/9),right=np.float32(512))
        ax.set_ylim(bottom=np.float32(1/9),top=np.float32(512))
        # ax.set_xlim([-0.125, 512])
        # ax.set_ylim([-0.125, 512])
        # Axis_line = np.linspace(*ax.get_xlim(), 2)
        # Axes
        # ax.plot([-0.125, 512], [-0.125, 512], linestyle='--', linewidth=2, color='blue')
        ax.plot([0,512], [0,512], linestyle='--', linewidth=2, color='blue')
        ax.plot([0,512], [0,1024], linestyle='--', linewidth=2, color='gray')
        ax.plot([0,512], [0,1536], linestyle='--', linewidth=2, color='gray')
        ax.plot([0,512], [0,256], linestyle='--', linewidth=2, color='gray')




        # Axis_line = np.linspace(*ax.get_ylim(), 2)
        # ax.plot(Axis_line, Axis_line, linestyle='--', linewidth=2, color='blue')


        # ax.plot([2,0], [0,256], linestyle='--',linewidth=2,color='gray')
        # x = [0,  19/20]
        # y = [1/20,1]
        # ax.plot([Axis_line[0]*2,Axis_line[1]*2],Axis_line, linestyle='--',color='gray')
        # ax.plot(Axis_line*3,Axis_line, linestyle='--',color='gray')

        # ax.text(0.05, 0.98, 'KG-MTP is better', color='gray', ha='left', va='top')
        # ax.text(0.95, 0.02, str(i)+' is better', color='gray', ha='right', va='bottom')
        # ax.text(0.05, 0.5, 'win/draw/loss\n'+win+'/'+draw+'/'+loss, color='black', ha='center', va='center',
        #         bbox=dict(facecolor='lightyellow', edgecolor='black'))

        font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14,
        }
        # font1 = {'family' : 'Times New Roman',
        # 'weight' : 'bold',
        # 'size'   : 16,
        # }
        plt.title("Total training time, Seconds")
        plt.xlabel("Total training time of "+str(i)+" : "+str(np.round(np.sum(b)/60, decimals=2))+" minutes",font)
        plt.ylabel("Total training time of KG-MTP: "+str(np.round(np.sum(a)/60, decimals=2))+" minutes",font)
        plt.savefig('../results/KG-MTP_vs_'+str(i)+'_trainTime.jpg', dpi=400,format="jpg")
        # ax.legend()
        plt.show()
        plt.clf()

