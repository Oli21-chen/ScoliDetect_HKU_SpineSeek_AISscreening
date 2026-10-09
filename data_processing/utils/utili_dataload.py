# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:07:19 2023
- video_libs_5(sampling rate)_128(img size)
- Age filter
- Label data analysis(mean,std)
- Severity label 
    
@author: Olive
"""
import numpy as np
from numpy import load
import pdb
# from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
# import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
# import tensorflow as tf


def load_hkusz(  
        COBB_TH=15,
        TABLE_SIZE=32,
        IMG_SIZE=224,
        sampling_rate = None,
        frame_size= None,
        filter_age = True,
        age_th = 18,
        screen = True,
        severity=True,
        events = 32,
        SZ_vData=[],
        SZ_tData=[],
        SZ_cali=[],
        SZ_cobbs=[],
        SZ_ages=[],
        SZ_sexs=[],
        
        ):
    ''' 'HKUSZ'  '''
    sz_vid_raw =load(
        r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\video_libs_'+sampling_rate+'_'+frame_size+'_fixedv2.npy',
        allow_pickle=True)#\log1
    sz_tab_raw = load(
        r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\table_libs_'+sampling_rate+'_'+frame_size+'_fixedv2.npy',
        allow_pickle=True)
    try:
        sz_tab_cali = load(
            r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\table_cali_'+sampling_rate+'_'+frame_size+'_fixedv2.npy',
            allow_pickle=True)
    except FileNotFoundError:    
        sz_tab_cali = np.zeros([500,100])
    sz_label =load(
        r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\label_libs_'+sampling_rate+'_'+frame_size+'.npy',
        allow_pickle=True)
    
    sz_sex = sz_label[:,1]
    sz_age = sz_label[:,0]
    sz_cobb = sz_label[:,2:4]
    for idx,(vid,tab,cali,cobb,age,sex) in enumerate(zip(sz_vid_raw,sz_tab_raw,sz_tab_cali,sz_cobb,sz_age,sz_sex)):
        
        vid = np.reshape(vid,[-1,IMG_SIZE,IMG_SIZE,3])
    
        # tab = tf.expand_dims(tab, -1)
        # tab = tf.image.resize( tab,[TABLE_SIZE,TABLE_SIZE])
        # tab = tf.squeeze(tab).numpy()
        
        if filter_age:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3) and age <= age_th:# and np.shape(tab)==(TABLE_SIZE,TABLE_SIZE)
                
                SZ_vData += [vid]
                SZ_cobbs += [cobb]
                SZ_tData += [tab]
                SZ_cali += [cali]
                SZ_ages += [age]
                SZ_sexs += [sex]
            # else:
            #     print(idx,np.shape(vid))
        else:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3):# and np.shape(tab)==(TABLE_SIZE,TABLE_SIZE):
                SZ_vData += [vid]
                SZ_cobbs += [cobb]
                SZ_tData += [tab]
                SZ_cali += [cali]
                SZ_ages += [age]
                SZ_sexs += [sex]
            # else:   
            #     print(idx,np.shape(vid))
    SZ_vData = np.array(SZ_vData)
    SZ_tData = np.array(SZ_tData)
    SZ_cali = np.array(SZ_cali)
    SZ_cobbs = np.array(SZ_cobbs)
    SZ_ages = np.array(SZ_ages)
    SZ_sexs = np.array(SZ_sexs)
    
    # pdb.set_trace()
    if severity:
        SZ_Severity = np.zeros([len(SZ_cobbs),1])
        for i,j in enumerate(SZ_cobbs):#Label_cobb_sele
            SZ_Severity[i]=max(j)
            if  max(j)<=20:
                SZ_Severity[i]=0
            elif 20<max(j)<=40:
                SZ_Severity[i]=1
            else:
                SZ_Severity[i]=2
                
    if screen:            
        SZ_Screen=np.zeros([len(SZ_cobbs),1])
        for i,j in enumerate(SZ_cobbs):#Label_cobb_sele
            SZ_Screen[i]=max(j)
            if  max(j)>=COBB_TH:
                SZ_Screen[i]=1
            else:
                SZ_Screen[i]=0
    # print('SZ num of control:',len(SZ_Labels[SZ_Labels==0]))
    # print('SZ num of Patient:',len(SZ_Labels[SZ_Labels==1]))
    return [SZ_tData, SZ_vData, SZ_cali, SZ_Screen,SZ_Severity,SZ_cobbs,SZ_ages,SZ_sexs]

def load_dkch(
        COBB_TH=15,
        TABLE_SIZE=32,
        IMG_SIZE=224,
        sampling_rate = None,
        frame_size= None,
        filter_age = True,
        age_th = 18,
        screen = True,
        severity=True,
        events=32,
        DK_vData=[],
        DK_tData=[],
        DK_cali=[],
        DK_cobbs=[],
        DK_ages=[],
        DK_sexs=[],
        ):
    ''' 'DKCH'  '''
    dk_vid_raw =load(
        r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\video_libs_'+sampling_rate+'_'+frame_size+'_fixedv2.npy',
        allow_pickle=True)#\log1
    dk_tab_raw = load(
        r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\table_libs_'+sampling_rate+'_'+frame_size+'_fixedv2.npy',
        allow_pickle=True)
    try:
        dk_tab_cali = load(
            r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\table_cali_'+sampling_rate+'_'+frame_size+'_fixedv2.npy',
            allow_pickle=True)
    except FileNotFoundError:    
        dk_tab_cali = np.zeros([500,100])
        
    dk_label =load(
        r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\label_libs_'+sampling_rate+'_'+frame_size+'.npy',
        allow_pickle=True)
    dk_sex = dk_label[:,1]
    dk_age = dk_label[:,0]
    dk_cobb = dk_label[:,2:4]
    # DK_vData ,DK_tData, DK_cobb_labels = [], [], []
    for idx,(vid,tab,cali,cobb,age,sex) in enumerate(zip(dk_vid_raw,dk_tab_raw,dk_tab_cali,dk_cobb,dk_age,dk_sex)):
    
        vid=np.reshape(vid,[-1,IMG_SIZE,IMG_SIZE,3])  
        
        # tab = tf.expand_dims(tab, -1)
        # tab = tf.image.resize( tab,[TABLE_SIZE,TABLE_SIZE])
        # tab = tf.squeeze(tab).numpy()
        
        if filter_age:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3) and age<=age_th:#and np.shape(tab)==(TABLE_SIZE,TABLE_SIZE)
                DK_vData += [vid]
                DK_tData += [tab]
                DK_cali += [cali]
                DK_cobbs += [cobb]
                DK_ages += [age]
                DK_sexs += [sex]
            # else:
            #     print(idx,np.shape(vid))
        else:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3):# and np.shape(tab)==(TABLE_SIZE,TABLE_SIZE):
               
                DK_vData += [vid]
                DK_tData += [tab]
                DK_cali += [cali]
                DK_cobbs += [cobb]
                DK_ages += [age]
                DK_sexs += [sex]
            # else:
            #     print(idx)
            
    DK_vData = np.array(DK_vData)
    DK_tData = np.array(DK_tData)
    DK_cali = np.array(DK_cali)
    DK_cobbs = np.array(DK_cobbs)
    DK_ages = np.array(DK_ages)
    DK_sexs = np.array(DK_sexs)
    
    if severity:
        DK_Severity = np.zeros([len(DK_cobbs),1])
        for i,j in enumerate(DK_cobbs):#Label_cobb_sele
            DK_Severity[i]=max(j)
            if  max(j)<=20:
                DK_Severity[i]=0
            elif 20<max(j)<=40:
                DK_Severity[i]=1
            else:
                DK_Severity[i]=2
                
    
    if screen:
        DK_Screen = np.zeros([len(DK_cobbs),1])
        for i,j in enumerate(DK_cobbs):#Label_cobb_sele
            DK_Screen[i]=max(j)
            if  max(j)>=COBB_TH:
                DK_Screen[i]=1
            else:
                DK_Screen[i]=0
    # print('DK num of control:',len(DK_Labels[DK_Labels==0]))
    # print('DK num of Patient:',len(DK_Labels[DK_Labels==1]))
    return [DK_tData, DK_vData,DK_cali, DK_Screen,DK_Severity,DK_cobbs,DK_ages,DK_sexs]



def data_processing(
        COBB_TH = 14,
        TABLE_SIZE = 32,
        IMG_SIZE = 128,
        ind_k = 0,# select k_folder index
        n_splits = 5 ,# choose the folder numbers
        sampling_rate = '9',
        frame_size= '128',
        filter_age = True,
        age_th = 18,
        events = 32
        ): 
    
    sz_data = load_hkusz(
            COBB_TH=COBB_TH,
            TABLE_SIZE=TABLE_SIZE,
            IMG_SIZE=IMG_SIZE,
            sampling_rate = sampling_rate,
            frame_size= frame_size,
            filter_age = filter_age,
            age_th = age_th,
            events = events
            )
    dk_data = load_dkch(
            COBB_TH=COBB_TH,
            TABLE_SIZE=TABLE_SIZE,
            IMG_SIZE=IMG_SIZE,
            sampling_rate = sampling_rate,
            frame_size= frame_size,
            filter_age = filter_age,
            age_th = age_th,
            events = events
            )
    DK_tData, DK_vData,DK_cali, DK_Screen,DK_Severity,DK_cobbs,DK_ages,DK_sexs = dk_data
    SZ_tData, SZ_vData,SZ_cali, SZ_Screen,SZ_Severity,SZ_cobbs,SZ_ages,SZ_sexs = sz_data
    # pdb.set_trace()
    print('Whole data size {}, Control:{},AIS:{}'.format(
          len(DK_Screen)+len(SZ_Screen),
          len(DK_Screen[DK_Screen==0])+len(SZ_Screen[SZ_Screen==0]),
          len(DK_Screen[DK_Screen==1])+len(SZ_Screen[SZ_Screen==1])
          ))
    
    table_lib = np.concatenate((DK_tData,SZ_tData))
    video_lib = np.concatenate((DK_vData,SZ_vData))
    # cali_lib = np.concatenate((DK_cali,SZ_cali))
    screen_lib = np.concatenate((DK_Screen,SZ_Screen))
 
    # Assuming SZ_Screen is your target variable
    test_size = 50  
    tr_idx, te_idx  = train_test_split(
        np.arange(len(screen_lib)),
        test_size = test_size,
        stratify = np.squeeze(screen_lib),
        random_state = 14
    )
    table_traintem = table_lib[tr_idx]
    screen_traintem = screen_lib[tr_idx]
    video_traintem = video_lib[tr_idx]
    
    table_test = table_lib[te_idx]
    screen_test = screen_lib[te_idx]
    video_test = video_lib[te_idx]

    val_size = 50
    tr_idx, te_idx  = train_test_split(
        np.arange(len(screen_traintem)),
        test_size = val_size,
        stratify = np.squeeze(screen_traintem),
        random_state = 14
    )
    table_train = table_traintem[tr_idx]
    screen_train = screen_traintem[tr_idx]
    video_train = video_traintem[tr_idx]

    
    table_val = table_traintem[te_idx]
    screen_val = screen_traintem[te_idx]
    video_val = video_traintem[te_idx]
            
    # print('Train size:{},test size:{}'.format(len(table_train),len(table_test)))
    # Grouping
    train_lib = [table_train,video_train ,screen_train]
    test_lib = [table_test,video_test ,screen_test]
    val_lib = [table_val,video_val ,screen_val]

    print('train size:',len(table_train))
    print('test size:',len(table_test))
    print('val size:',len(table_val))
    return train_lib, test_lib, val_lib 
    # [table_train,
    #         table_test],[
    #             video_train,
    #             video_test],[
    #                 cali_train,
    #                 cali_test],[
    #                     screen_train,
    #                     screen_test],[
    #                         severity_train,
    #                         severity_test],[
    #                             age_train,
    #                             age_test],[
    #                                 sex_train,
    #                                 sex_test]
                                                                                
# [table_train,
#         table_test],[
#             screen_train,
#             screen_test] = data_processing(
#                                             COBB_TH = 10,
#                                             TABLE_SIZE = 32,
#                                             ind_k = 0,
#                                             n_splits = 10
#                                             )