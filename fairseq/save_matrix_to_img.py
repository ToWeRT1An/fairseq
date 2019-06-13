import torch

from torchvision.utils import save_image

from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sn
figure(num=None, figsize=(20,20), dpi=80, facecolor='w', edgecolor='k')
import matplotlib as mpl

def save_attn(m,file_path ,xlabel='auto',ylabel='auto'):
    m = m.numpy()
    if len(m.shape) == 3:
        figure(num=None, figsize=(m.shape[1]*2,m.shape[2]*2), dpi=80, facecolor='w', edgecolor='k')
        for i in range(m.shape[0]):
            
            hm = sn.heatmap(m[i],annot=True,fmt='10.1f',xticklabels=xlabel[i],yticklabels=ylabel[i])
            plt.savefig(file_path+'/'+str(i)+'_'+str(m.shape[0])+'_'+\
                       str(m.shape[1])+'_'+str(m.shape[-1])+'.jpg')
 
    elif len(m.shape) == 2:
        figure(num=None, figsize=(m.shape[0]*2,m.shape[1]*2), dpi=80, facecolor='w', edgecolor='k',)
        hm = sn.heatmap(m,annot=True,fmt='10.1f',xticklabels=xlabel,yticklabels=ylabel)
        plt.savefig(file_path+'/'+'_'+str(m.shape[0])+'_'+\
                   str(m.shape[1])+'.jpg') 
