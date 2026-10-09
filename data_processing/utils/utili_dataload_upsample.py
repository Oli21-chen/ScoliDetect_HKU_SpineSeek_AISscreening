# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:20:52 2024
30 frames (B, 30, N)
- video_libs_5(sampling rate)_128(img size)
- Age filter
- Label data analysis(mean,std)
- Severity label 
    
@author: Olive
"""
import numpy as np
from numpy import load
import pdb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# import tensorflow.keras.backend as K
import tensorflow as tf
# from typing import List
SEED = 14
np.random.seed(SEED)

def load_hkusz1(  
        COBB_TH=10,
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
        SZ_cobbs=[],
        SZ_ages=[],
        SZ_sexs=[],
        
        ):
    ''' 'HKUSZ'  '''
    sz_vid_raw =load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\sz_video_libs_9_128_fixedv1.npy",
        allow_pickle=True)#\log1
    sz_tab_raw = load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\sz_table_libs_9_128_fixedv1.npy",
        allow_pickle=True)
    sz_label =load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\sz_label_libs_9_128_fixedv1.npy",
        allow_pickle=True)
        
    sz_sex = sz_label[:,1]
    sz_age = sz_label[:,0]
    sz_cobb = sz_label[:,2:4]
    # pdb.set_trace()
    for idx,(vid,tab,cobb,age,sex) in enumerate(zip(sz_vid_raw,sz_tab_raw,sz_cobb,sz_age,sz_sex)):
        
        vid = np.reshape(vid,[-1,IMG_SIZE,IMG_SIZE,3])

        if filter_age:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3) and age <= age_th:
                
                SZ_vData += [vid]
                SZ_cobbs += [cobb]
                SZ_tData += [tab]
                SZ_ages += [age]
                SZ_sexs += [sex]
            # else:
            #     print(idx,np.shape(vid))
        else:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3):
                SZ_vData += [vid]
                SZ_cobbs += [cobb]
                SZ_tData += [tab]
                SZ_ages += [age]
                SZ_sexs += [sex]
            # else:   
            #     print(idx,np.shape(vid))
    SZ_vData = np.array(SZ_vData)
    SZ_tData = np.array(SZ_tData)
    SZ_cobbs = np.array(SZ_cobbs)
    SZ_ages = np.array(SZ_ages)
    SZ_sexs = np.array(SZ_sexs)
    
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
    # print('SZ num of control:',len(SZ_Screen[SZ_Screen==0]))
    # print('SZ num of Patient:',len(SZ_Screen[SZ_Screen==1]))
    return SZ_vData, SZ_tData, SZ_Screen,SZ_Severity,SZ_cobbs,SZ_ages,SZ_sexs

def load_hkusz2(
        COBB_TH=10,
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
    SZ_cobbs=[],
    SZ_ages=[],
    SZ_sexs=[],
    ):
    
    sz_vid_raw =load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\sz_video_libs_9_128_fixedv2.npy",
        allow_pickle=True)#\log1
    sz_tab_raw = load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\sz_table_libs_9_128_fixedv2.npy",
        allow_pickle=True)
    sz_label =load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\sz_label_libs_9_128_fixedv2.npy",
        allow_pickle=True)
        
    sz_sex = sz_label[:,1]
    sz_age = sz_label[:,0]
    sz_cobb = sz_label[:,2:4]
    # pdb.set_trace()
    for idx,(vid,tab,cobb,age,sex) in enumerate(zip(sz_vid_raw,sz_tab_raw,sz_cobb,sz_age,sz_sex)):
        
        vid = np.reshape(vid,[-1,IMG_SIZE,IMG_SIZE,3])
        if filter_age:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3) and age <= age_th:
                
                SZ_vData += [vid]
                SZ_cobbs += [cobb]
                SZ_tData += [tab]
                SZ_ages += [age]
                SZ_sexs += [sex]
            # else:
            #     print(idx,np.shape(vid))
        else:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3):
                SZ_vData += [vid]
                SZ_cobbs += [cobb]
                SZ_tData += [tab]
                SZ_ages += [age]
                SZ_sexs += [sex]
            # else:   
            #     print(idx,np.shape(vid))
    SZ_vData = np.array(SZ_vData)
    SZ_tData = np.array(SZ_tData)
    SZ_cobbs = np.array(SZ_cobbs)
    SZ_ages = np.array(SZ_ages)
    SZ_sexs = np.array(SZ_sexs)
    
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
    return SZ_vData, SZ_tData, SZ_Screen,SZ_Severity,SZ_cobbs,SZ_ages,SZ_sexs

def load_dkch1(
        COBB_TH=10,
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
        DK_cobbs=[],
        DK_ages=[],
        DK_sexs=[],
        ):
    ''' 'DKCH'  '''
    dk_vid_raw =load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\dk_video_libs_9_128_fixedv1.npy",
        allow_pickle=True)#\log1
    dk_tab_raw = load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\dk_table_libs_9_128_fixedv1.npy",
        allow_pickle=True)
    dk_label =load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\dk_label_libs_9_128_fixedv1.npy",
        allow_pickle=True)
    dk_sex = dk_label[:,1]
    dk_age = dk_label[:,0]
    dk_cobb = dk_label[:,2:4]
    # pdb.set_trace()
    # DK_vData ,DK_tData, DK_cobb_labels = [], [], []
    for idx,(vid,tab,cobb,age,sex) in enumerate(zip(dk_vid_raw,dk_tab_raw,dk_cobb,dk_age,dk_sex)):
        vid=np.reshape(vid,[-1,IMG_SIZE,IMG_SIZE,3])  
        if filter_age:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3) and age<=age_th:
                DK_vData += [vid]
                DK_tData += [tab]
                DK_cobbs += [cobb]
                DK_ages += [age]
                DK_sexs += [sex]
        else:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3):
                DK_vData += [vid]
                DK_tData += [tab]
                DK_cobbs += [cobb]
                DK_ages += [age]
                DK_sexs += [sex]
            # else:
            #     print(idx)
            
    DK_vData = np.array(DK_vData)
    DK_tData = np.array(DK_tData)
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
    return DK_vData, DK_tData, DK_Screen,DK_Severity,DK_cobbs,DK_ages,DK_sexs

def load_dkch2(
        COBB_TH=10,
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
    DK_cobbs=[],
    DK_ages=[],
    DK_sexs=[],
    ):
    ''' 'DKCH'  '''
    dk_vid_raw =load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\dk_video_libs_9_128_fixedv2.npy",
        allow_pickle=True)#\log1
    dk_tab_raw = load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\dk_table_libs_9_128_fixedv2.npy",
        allow_pickle=True)
    dk_label =load(
        r"C:\Users\Olive\Desktop\Table_scolidetect\data\dk_label_libs_9_128_fixedv2.npy",
        allow_pickle=True)
    dk_sex = dk_label[:,1]
    dk_age = dk_label[:,0]
    dk_cobb = dk_label[:,2:4]
    # pdb.set_trace()
    # DK_vData ,DK_tData, DK_cobb_labels = [], [], []
    for idx,(vid,tab,cobb,age,sex) in enumerate(zip(dk_vid_raw,dk_tab_raw,dk_cobb,dk_age,dk_sex)):
        vid=np.reshape(vid,[-1,IMG_SIZE,IMG_SIZE,3])  
        if filter_age:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3) and age<=age_th:
                DK_vData += [vid]
                DK_tData += [tab]
                DK_cobbs += [cobb]
                DK_ages += [age]
                DK_sexs += [sex]
        else:
            if np.shape(vid)==(events,IMG_SIZE,IMG_SIZE,3):
                DK_vData += [vid]
                DK_tData += [tab]
                DK_cobbs += [cobb]
                DK_ages += [age]
                DK_sexs += [sex]
            # else:
            #     print(idx)
            
    DK_vData = np.array(DK_vData)
    DK_tData = np.array(DK_tData)
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
    return DK_vData, DK_tData, DK_Screen,DK_Severity,DK_cobbs,DK_ages,DK_sexs
        



    
def data_processing(
        COBB_TH = 15,
        IMG_SIZE = 128,
        ): 
    
    sz_data = load_hkusz1(
            COBB_TH = COBB_TH,
            IMG_SIZE = IMG_SIZE,
            sampling_rate = '9',
            frame_size = '128',
            filter_age = True,
            age_th = 18,
            events = 30
            )
    dk_data = load_dkch1(
            COBB_TH = COBB_TH,
            IMG_SIZE = IMG_SIZE,
            sampling_rate = '9',
            frame_size = '128',
            filter_age = True,
            age_th = 18,
            events = 30
            )
    dk_updata = load_dkch2(
            COBB_TH = COBB_TH,
            IMG_SIZE = IMG_SIZE,
            sampling_rate = '9',
            frame_size = '128',
            filter_age = True,
            age_th = 18,
            events = 30
            )
    sz_updata = load_hkusz2(
            COBB_TH = COBB_TH,
            IMG_SIZE = IMG_SIZE,
            sampling_rate = '9',
            frame_size = '128',
            filter_age = True,
            age_th = 18,
            events = 30
            )    
    DK_vData, DK_tData, DK_Screen,DK_Severity,DK_cobbs,DK_ages,DK_sexs = dk_data
    DK_upv, DK_upt, DK_upScreen,DK_upSeverity,DK_upcobbs,DK_upages,DK_upsexs = dk_updata
    SZ_vData, SZ_tData, SZ_Screen,SZ_Severity,SZ_cobbs,SZ_ages,SZ_sexs = sz_data
    SZ_upv,SZ_upt,SZ_upScreen,SZ_upSeverity,SZ_upcobbs,SZ_upages,SZ_upsexs = sz_updata
    
    table_lib = np.concatenate((DK_tData,SZ_tData,DK_upt,SZ_upt))
    video_lib = np.concatenate((DK_vData,SZ_vData,DK_upv,SZ_upv))
    # cali_lib = np.concatenate((DK_cali,SZ_cali))
    screen_lib = np.concatenate((DK_Screen,SZ_Screen,DK_upScreen,SZ_upScreen))
    # demo_ages = np.concatenate((DK_ages,DK_upages,SZ_ages,SZ_upages))
    # demo_cobbs = np.concatenate((DK_cobbs,DK_upcobbs,SZ_cobbs,SZ_upcobbs))
    # demo_sexs = np.concatenate((DK_sexs,DK_upsexs,SZ_sexs,SZ_upsexs))
    
    # pdb.set_trace()
    # Assuming SZ_Screen is your target variable
    test_size = 150  
    tr_idx, te_idx  = train_test_split(
        np.arange(len(screen_lib)),
        test_size = test_size,
        stratify = np.squeeze(screen_lib),
        random_state = SEED
    )
    table_traintem = table_lib[tr_idx]
    screen_traintem = screen_lib[tr_idx]
    video_traintem = video_lib[tr_idx]
    # age_traintem = demo_ages[tr_idx]
    # cobb_traintem = demo_cobbs[tr_idx]
    # sex_traintem = demo_sexs[tr_idx]
    
    table_test = table_lib[te_idx]
    screen_test = screen_lib[te_idx]
    video_test = video_lib[te_idx]
    # age_test = demo_ages[te_idx]
    # cobb_test = demo_cobbs[te_idx]
    # sex_test = demo_sexs[te_idx]
    
    val_size = 150
    tr_idx, te_idx  = train_test_split(
        np.arange(len(screen_traintem)),
        test_size = val_size,
        stratify = np.squeeze(screen_traintem),
        random_state = SEED
    )
    table_train = table_traintem[tr_idx]
    screen_train = screen_traintem[tr_idx]
    video_train = video_traintem[tr_idx]
    # age_train = age_traintem[tr_idx]
    # cobb_train = cobb_traintem[tr_idx]
    # sex_train = sex_traintem[tr_idx]
    
    table_val = table_traintem[te_idx]
    screen_val = screen_traintem[te_idx]
    video_val = video_traintem[te_idx]
    # age_val = age_traintem[te_idx]
    # cobb_val = cobb_traintem[te_idx]
    # sex_val = sex_traintem[te_idx]

    # Grouping
    train_lib = [table_train,video_train ,screen_train]
    test_lib = [table_test,video_test ,screen_test]
    val_lib = [table_val,video_val ,screen_val]
    
    # age_lib = [age_traintem, age_test]
    # cobb_lib = [cobb_traintem,cobb_test]
    # sex_lib = [sex_traintem, sex_test]
    # pdb.set_trace()
    
    print('train size:',len(table_train))
    print('test size:',len(table_test))
    print('val size:',len(table_val))
    return train_lib, test_lib, val_lib   
    # pdb.set_trace()
    # cobb_libs = np.max(demo_cobbs,1)
    # return demo_ages, cobb_libs, demo_sexs
  
    
# train_lib, test_lib, val_lib   = data_processing(COBB_TH =11)

# pos_idx = np.where(demo_cobbs>=11,1,0)
# neg_idx =  np.where(demo_cobbs<11,1,0)

# pos_age = demo_ages[pos_idx.astype(bool)]
# neg_age = demo_ages[neg_idx.astype(bool)]
# print('pos_age',np.mean(pos_age),np.std(pos_age))
# print('neg_age',np.mean(neg_age),np.std(neg_age))

# pos_gender = demo_sexs[pos_idx.astype(bool)]
# neg_gender = demo_sexs[neg_idx.astype(bool)]

# unique, counts = np.unique(pos_gender, return_counts=True)
# result = dict(zip(unique, counts))
# print('pos',result)
# unique, counts = np.unique(neg_gender, return_counts=True)
# result = dict(zip(unique, counts))
# print('neg',result)
