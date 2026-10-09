# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:55:29 2023
Step_2_Sampling_StraightTurningV1: 
    naively sample data
    
- default paras:
    videos frame rate: 30 Hz
    vid_height: 1920
    vid_width: 1080
    
interpolation=cv2.INTER_AREA
interpolation=cv2.INTER_NEAREST
interpolation=cv2.INTER_CUBIC

- raw_value is used for joint position recalibration

@author: Olive
"""
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import numpy as np
import cv2
import os
from numpy import save
from utili_Getpose_v4 import GetAllFeatures


def _get_table_libs(
        path,
        max_seq = 32,
        FRAME_APART=1,
        INITIAL_FRAME=0,
        from_start = True):
    df = pd.read_csv(''.join(path))
    if df.shape[0]==0:
        raise TypeError("To user: the file might be empty, the file path:\n"+''.join(path))
    # Prepare the data
    df = df.dropna()
    #values without the frame index
    values = df.values[INITIAL_FRAME:,:-1]
    if len(values)>300:# for frame rate = 60
        values = values[::2]
    saved_values = np.squeeze(GetAllFeatures(values))        
    if len(saved_values) >= max_seq*FRAME_APART:
        if from_start:
            saved_values = saved_values[:max_seq*FRAME_APART,...]
            raw_values = values[:max_seq*FRAME_APART,...]
        else:    
            saved_values = saved_values[-max_seq*FRAME_APART:-1,...]
            raw_values = values[-max_seq*FRAME_APART:-1,...]
    else:
        saved_values = np.concatenate([saved_values,
                                       saved_values[0:int(
                                           max_seq*FRAME_APART- len(saved_values)),...]],
                                      axis=0)
        raw_values =  np.concatenate([values,
                                      values[0:int(
                                           max_seq*FRAME_APART- len(saved_values)),...]],
                                      axis=0)
    saved_values = saved_values[::FRAME_APART,...]
    raw_values = raw_values[::FRAME_APART,...]
    print('table shape:{},{}'.format(np.shape(saved_values),np.shape(raw_values)))
    assert len(saved_values)== max_seq
    return np.array(saved_values), np.array(raw_values)


def _get_video_libs(
        path,
        max_seq = 32, 
        resize = (128, 128),
        FRAME_APART = 1,
        INITIAL_FRAME = 0,
        from_start = True):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, INITIAL_FRAME+1) # set initial frame
    print('frame rate:',cap.get(cv2.CAP_PROP_FPS))
    assert int(cap.get(cv2.CAP_PROP_FPS)) == 30
    saved_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(
            cv2.resize(
                frame, 
                resize,
                interpolation=cv2.INTER_NEAREST), 
            cv2.COLOR_BGR2RGB)
        saved_frames += [frame]
    cap.release()    
    # saved all frames
    if len(saved_frames) < max_seq*FRAME_APART:
        # pdb.set_trace()
        saved_frames += saved_frames[0:int( max_seq*FRAME_APART - len(saved_frames))]
    else:
        if from_start:
            saved_frames = saved_frames[:max_seq*FRAME_APART]
        else:    
            saved_frames = saved_frames[-max_seq*FRAME_APART:-1]
    saved_frames = saved_frames[::FRAME_APART]
    print('video shape:{} \n'.format(np.shape(np.array(saved_frames))))

    assert len(saved_frames)== max_seq
    return np.array(saved_frames)

def run_sz(
        table_path,
        video_path,
        video_libs = [],
        table_libs = [],
        table_cali = [],
        label_libs = [],
        IMG_SIZE = 128,
        MAX_SEQ_LENGTH = 32,
        FRAME_APART = 9,
        MIN_FILE_INDEX = 1,
        INITIAL_FRAME = 0 #300 in total
        ):
    para_name = 'sz'
  
    ''' Search videos based on the order in table '''
    if para_name == 'sz':
        dictionary=pd.read_excel(r'C:\Users\Olive\.spyder-py3\Database\Label_SZgait_4video.xlsx'
                          ,usecols=['File No.','age','sex','M_cobb_l','M_cobb_r'])  
        dictionary = dictionary.dropna()
        dictionary_val = dictionary.values
        starting_index = np.where(dictionary_val[:,0] == MIN_FILE_INDEX)[0][0]
        print('starting_index',starting_index)
        # pdb.set_trace()
        sub_label = dictionary_val[starting_index:,1:5]
        sub_index = dictionary_val[starting_index:,0]

        # tablefile-index <- label-index
        for i,(index,label) in enumerate(zip(sub_index,sub_label)):
            print('para_name, index:',para_name,index)
            video_name = video_path+os.sep+para_name+'_'+str(int(index))+'_step1.mp4'
            if not os.path.exists(video_name):
                continue
            table_name = table_path+os.sep+para_name+'_'+str(int(index))+'_step_1.csv'
            if not os.path.exists(table_name):
                continue
            # pdb.set_trace()
            ''' libs of  table '''    
            saved_table, raw_table = _get_table_libs(
                table_name,
                MAX_SEQ_LENGTH,
                FRAME_APART,
                INITIAL_FRAME
                )
            table_libs += [saved_table]
            table_cali += [raw_table]
            ''' libs of video  '''
            saved_vid = _get_video_libs(
                video_name,
                MAX_SEQ_LENGTH,
                (IMG_SIZE,IMG_SIZE),
                FRAME_APART,
                INITIAL_FRAME)
            video_libs += [saved_vid]
            
            ''' libs of label  '''
            # 
            label_libs += [label]
        
        table_libs = np.array(table_libs)
        # table_cali = np.array(table_cali)
        # video_libs = np.array(video_libs)                    
        # label_libs = np.array(label_libs)                    
        # pdb.set_trace()
        print('table_libs:', np.shape(table_libs))
        # print('table_cali:', np.shape(table_cali))
        # print('video_libs:', np.shape(video_libs))
        # print('label_libs:', np.shape(label_libs))
        
    save(r'C:\Users\Olive\Desktop\Table_scolidetect\data\table_libs_9_128_fixedv2.npy', table_libs) 
    # save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\table_cali_9_128_fixedv1.npy', table_cali) 
    save(r'C:\Users\Olive\Desktop\Table_scolidetect\data\video_libs_9_128_fixedv2.npy', video_libs) 
    save(r'C:\Users\Olive\Desktop\Table_scolidetect\data\label_libs_9_128_fixedv2.npy', label_libs) 
    print('saved')
    return

def run_dk(
        table_path,
        video_path,
        video_libs = [],
        table_libs = [],
        table_cali = [],
        label_libs = [],
        IMG_SIZE = 128,
        MAX_SEQ_LENGTH = 32,
        FRAME_APART = 9,
        MIN_FILE_INDEX = 1,
        INITIAL_FRAME = 0 #300 in total
        ):
    para_name = 'dk'
   
    if para_name == 'dk':
        dictionary=pd.read_excel(r'C:\Users\Olive\.spyder-py3\Database\Label_DKgait_4video.xlsx'
                          ,usecols=['File No.','age','sex','M_cobb_l','M_cobb_r'])  
        dictionary = dictionary.dropna()
        dictionary_val = dictionary.values
        sub_label = dictionary_val[:,1:5]
        sub_index = dictionary_val[:,0]
        for i,(index,label) in enumerate(zip(sub_index,sub_label)):

            print(' para_name, index:',para_name,index)
            video_name = video_path+os.sep+para_name+'_'+str(int(index))+'_step1.mp4'
            if not os.path.exists(video_name):
                continue
            table_name = table_path+os.sep+para_name+'_'+str(int(index))+'_step_1.csv'
            if not os.path.exists(table_name):
                continue
            
            # pdb.set_trace()
            ''' libs of  table '''    
            saved_table,raw_table = _get_table_libs(
                table_name,
                MAX_SEQ_LENGTH,
                FRAME_APART,
                INITIAL_FRAME
                )
            table_libs +=[saved_table]
            table_cali += [raw_table]
            ''' libs of video  '''
            saved_vid = _get_video_libs(
                video_name,
                MAX_SEQ_LENGTH,
                (IMG_SIZE,IMG_SIZE),
                FRAME_APART,
                INITIAL_FRAME)
            video_libs += [saved_vid]
            
            ''' libs of label  '''
            
            label_libs += [label]
        
        table_libs = np.array(table_libs)
        # table_cali = np.array(table_cali)
        # video_libs = np.array(video_libs)                    
        # label_libs = np.array(label_libs)   
        # pdb.set_trace()                 
        print('table_libs:', np.shape(table_libs))
        # print('table_cali:', np.shape(table_cali))
        # print('video_libs:', np.shape(video_libs))
        # print('label_libs:', np.shape(label_libs))
    save(r'C:\Users\Olive\Desktop\Table_scolidetect\data\dk_table_libs_9_128_fixedv1.npy', table_libs) 
    # save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\table_cali_9_128_fixedv1.npy', table_cali) 
    save(r'C:\Users\Olive\Desktop\Table_scolidetect\data\dk_video_libs_9_128_fixedv1.npy', video_libs) 
    save(r'C:\Users\Olive\Desktop\Table_scolidetect\data\dk_label_libs_9_128_fixedv1.npy', label_libs)  
    print('saved')
    return

def run_others(
        table_path,
        video_path,
        video_libs = [],
        table_libs = [],
        table_cali = [],
        IMG_SIZE = 128,
        MAX_SEQ_LENGTH = 32,
        FRAME_APART = 9,
        MIN_FILE_INDEX = 1,
        INITIAL_FRAME = 0 #300 in total
        ):
    para_name = 'pk'
    # index_df  = pd.read_excel(r"C:\Users\Olive\Desktop\HKU_PhD\Pui Kiu\筛查日志\疑似侧弯名单302.xlsx"
    #              ,sheet_name='Index',usecols=['index','File No.'])
    # sub_index = index_df['index'].tolist()
    sub_index = np.arange(1,742)
    for i,index in enumerate(sub_index):
        print(' para_name, index:',para_name,index)
        video_name = video_path+os.sep+para_name+'_'+str(int(index))+'_step1.mp4'
        if not os.path.exists(video_name):
            continue
        table_name = table_path+os.sep+para_name+'_'+str(int(index))+'_step_1.csv'
        if not os.path.exists(table_name):
            continue
         
        ''' libs of  table '''    
        saved_table,raw_table = _get_table_libs(
            table_name,
            MAX_SEQ_LENGTH,
            FRAME_APART,
            INITIAL_FRAME
            )
        table_libs +=[saved_table]
        table_cali += [raw_table]
        ''' libs of video  '''
        saved_vid = _get_video_libs(
            video_name,
            MAX_SEQ_LENGTH,
            (IMG_SIZE,IMG_SIZE),
            FRAME_APART,
            INITIAL_FRAME)
        video_libs += [saved_vid]
       
    
    table_libs = np.array(table_libs)
    # table_cali = np.array(table_cali)
    video_libs = np.array(video_libs)                                     
    print('table_libs:', np.shape(table_libs))
    # print('table_cali:', np.shape(table_cali))
    print('video_libs:', np.shape(video_libs))
    save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_pk\table_libs_9_128_fixedv1.npy', table_libs) 
    # save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\table_cali_9_128_fixedv2.npy', table_cali) 
    save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_pk\video_libs_9_128_fixedv1.npy.npy', video_libs) 
    print('saved')
    return 
if __name__ == '__main__':

    # table_path = r"C:\Users\Olive\Desktop\HKU_PhD\Pui Kiu\筛查日志\2024pk_withGT\2024pk_tables"
    # video_path = r"C:\Users\Olive\Desktop\HKU_PhD\Pui Kiu\筛查日志\2024pk_withGT\2024pk_videos"
    table_path = r"C:\Users\Olive\Desktop\video_retrival\dk_table_refinedblaze"
    video_path = r"C:\Users\Olive\Desktop\video_retrival\dk_video_refinedblaze"
    # run_sz(
    #     table_path,
    #     video_path,
    #     IMG_SIZE = 128,
    #     MAX_SEQ_LENGTH = 32,
    #     FRAME_APART = 9,
    #     MIN_FILE_INDEX = 1,
    #     INITIAL_FRAME = 60 #300 in total
    #     )
    run_dk(
            table_path,
            video_path,
        IMG_SIZE = 128,
        MAX_SEQ_LENGTH = 30,
        FRAME_APART = 5,
        MIN_FILE_INDEX = 1,
        INITIAL_FRAME = 0 #300 in total
        )


    
    
   