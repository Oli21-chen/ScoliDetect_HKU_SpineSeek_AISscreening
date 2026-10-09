# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:01:37 2023
[DK_tData, DK_vData, DK_Labels,DK_ages,DK_sexs]
[SZ_tData, SZ_vData, SZ_Labels,SZ_ages,SZ_sexs]
@author: Olive
"""
import numpy as np
from utili_dataload import load_hkusz,load_dkch,data_processing
import matplotlib.pyplot as plt
import pdb

sz_data = load_hkusz(
        COBB_TH=10,
        TABLE_SIZE=32,
        IMG_SIZE=128,
        sampling_rate = '5',
        frame_size= '128',
        filter_age = True,
        age_th = 18,
        )
dk_data = load_dkch(
        COBB_TH=10,
        TABLE_SIZE=32,
        IMG_SIZE=128,
        sampling_rate = '5',
        frame_size= '128',
        filter_age = True,
        age_th = 18,
        )

DK_tData, DK_vData, DK_Screen,DK_Severity, DK_cobbs,DK_ages,DK_sexs = dk_data
SZ_tData, SZ_vData, SZ_Screen, SZ_Severity, SZ_cobbs,SZ_ages,SZ_sexs = sz_data
'''  age '''
ages = np.concatenate([DK_ages,SZ_ages],axis=0)
ages_mean = np.mean(ages)
ages_std = np.std(ages) 

print ('age mean:{}, std:{}'.format(ages_mean,ages_std))

# ''' sex '''
# sexs = np.concatenate([DK_sexs,SZ_sexs],0)
# sexs_f = len(sexs[sexs=='F'])
# sexs_m = len(sexs[sexs=='M'])
# print('sex Female:{}, Male:{}'.format(sexs_f,sexs_m))

''' cobb '''
cobbs = np.concatenate([DK_cobbs,SZ_cobbs],0)
cobbs_max = np.max(cobbs,1)
cobbs_mean = np.mean(cobbs_max)
cobbs_std = np.std(cobbs_max)
print ('cobbs mean:{}, std:{}'.format(cobbs_mean,cobbs_std))

''' plot '''
fig,ax = plt.subplots(2,1,figsize=(9, 9),sharey=False)
fig.suptitle('property')
ax[0].hist(ages,30)
ax[0].set_title('Ages distribution')


ax[1].hist(cobbs_max,30)
ax[1].set_title('Cobb angle distribution')

plt.show()
plt.tight_layout()




