# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:54:51 2024
Step_2_Sampling_StraightTurningV3: 
    Shift signal by placing peak at the middle (150), the results is really best.
    it works for using blazepose coordinates, the data extracted in ./Database

After sampling these data, check that performance againts data-raw(not fixed),
be declaring, check raw data vs refined data (remove unexpected motion) by shifting peaks
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

from numpy import save
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import numpy as np
import cv2
import os
from utili_Getpose_v4 import GetAllFeatures
from scipy.signal import find_peaks
from scipy import stats
from scipy.interpolate import interp1d

def interpolate1d(seq,max_seq):
    x_new = np.linspace(1, len(seq), num=max_seq)
    interpolated_data = np.zeros((max_seq, np.shape(seq)[1]))
    # pdb.set_trace()
    for i in range(seq.shape[1]):
        x = np.arange(1,seq.shape[0]+1)
        y = seq[:,i]
        f = interp1d(x, y, kind='cubic')
        interpolated_data[:,i] = f(x_new)
    # pdb.set_trace()
    # plt.figure()
    # plt.plot(np.sum(interpolated_data,1))
    # plt.title('interpolated')
    # plt.show()
    return interpolated_data

def get_symmetric_indices(peak, max_seq, min_val=0, max_val=300):
    original_indices = [int(peak)]
    left = int(peak) - 1
    right = int(peak) + 1
    count = 1
    # pdb.set_trace()
    while count <= max_seq:
        if right <= max_val:
            original_indices.append(right)
            count += 1
            if count >= max_seq:
                break
        if left >= min_val:
            original_indices.append(left)
            count += 1
            if count >= max_seq:
                break
        if left < min_val and right > max_val:
            break
        left -= 1
        right += 1
    return original_indices

def peak_shift_mid(seq,whole_leng = 300):
    ''' Find peak, and get left_right balanced length signal '''
    assert len(seq)==whole_leng
    seq_z = stats.zscore( np.sum(seq,1))
    peaks, _ = find_peaks(seq_z,distance = 300)
    assert len(peaks) == 1, "peaks amount should be 1"
    # pdb.set_trace()
    shift_len = int(whole_leng//2-peaks)
    
    if shift_len >0:
        seq_filtered = np.concatenate((seq[-shift_len:], seq[:-shift_len]),0)
    else:
        shift_len = abs(shift_len)
        seq_filtered = np.concatenate((seq[shift_len:], seq[:shift_len]),0)

    if len(seq_filtered) != 300:
        pdb.set_trace()
        print('')
    # plt.figure()
    # plt.plot(seq_z,label='raw')
    # plt.plot(peaks, seq_z[peaks],'ro')
    # plt.plot( stats.zscore(np.sum(seq_filtered,1)),label='shifted')
    # plt.plot(np.zeros_like(seq_z), "--", color="gray")
    # plt.legend()
    # plt.show()
    return seq_filtered, peaks #(time, variates)


def _get_table_libs(
        path,
        max_seq = 32,
        INITIAL_FRAME=0,
        from_start = True,
        check = False
        ):
    df = pd.read_csv(''.join(path))
    if df.shape[0]==0:
        raise TypeError("To user: the file might be empty, the file path:\n"+''.join(path))
    # Prepare the data
    df = df.dropna()
    #values without the frame index
    values = df.values[INITIAL_FRAME:,:-1]
    if check:
        pdb.set_trace()
    # original_indices = np.arange(len(values))
    if len(values)>300:# for frame rate = 60
        values = values[::2]
        # original_indices =  original_indices[::2]
    saved_values = np.squeeze(GetAllFeatures(values)) 
    filtered_values,peaks = peak_shift_mid(saved_values)
    FRAME_APART = int(len(filtered_values)//max_seq)
    FRAME_APART = 1 if FRAME_APART == 0 else FRAME_APART
    # if len(filtered_values) >= max_seq*FRAME_APART:
    if from_start:
        filtered_values = filtered_values[:max_seq*FRAME_APART,...]
        # original_indices = original_indices[:max_seq * FRAME_APART]
        raw_values = values[:max_seq*FRAME_APART,...]
    else:    
        filtered_values = filtered_values[-max_seq*FRAME_APART:-1,...]
        # original_indices = original_indices[-max_seq * FRAME_APART:]
        raw_values = values[-max_seq*FRAME_APART:-1,...]
    filtered_values = filtered_values[::FRAME_APART,...]
    # original_indices = original_indices[::FRAME_APART]
    raw_values = raw_values[::FRAME_APART,...]
    # assert len(filtered_values) == max_seq, 'length is not correct'
    if len(filtered_values) != max_seq:
        pdb.set_trace()
    return np.array(filtered_values),peaks, np.array(raw_values)

def _get_video_libs(
        path,
        max_seq: int, 
        resize: tuple,
        # original_indices: list
        peaks: int
        ):
    cap = cv2.VideoCapture(path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, INITIAL_FRAME-1) # set initial frame
    print('frame rate:',cap.get(cv2.CAP_PROP_FPS))
    assert int(cap.get(cv2.CAP_PROP_FPS)) == 30
    # to check length with peak index
    frame_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    original_indices = get_symmetric_indices(peaks,
                                             max_seq,
                                             min_val = 0, 
                                             max_val = frame_length)
    original_indices = [int(arr-1) for arr in original_indices]
    original_indices = sorted(original_indices)
    saved_frames = []
    for index in original_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(
            cv2.resize(
                frame, 
                resize,
                interpolation=cv2.INTER_NEAREST), 
            cv2.COLOR_BGR2RGB)
        saved_frames.append(frame)
    cap.release()    
 
    print('video shape:{} \n'.format(np.shape(np.array(saved_frames))))
    if len(saved_frames)!= max_seq:
        pdb.set_trace()
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
        INITIAL_FRAME = 0, #300 in total
        check = False
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
        sub_label = dictionary_val[starting_index:,1:5]
        sub_index = dictionary_val[starting_index:,0]
        index_in_table = []
        # tablefile-index <- label-index
        for i,(index,label) in enumerate(zip(sub_index,sub_label)):
            # pdb.set_trace()
            print('para_name, index:',para_name,index)
            video_name = video_path+os.sep+para_name+'_'+str(int(index))+'_step1.mp4'
            if not os.path.exists(video_name):
                continue
            table_name = table_path+os.sep+para_name+'_'+str(int(index))+'_step_1.csv'
            if not os.path.exists(table_name):
                continue
            ''' libs of  table '''    
            saved_table,peaks,raw_values = _get_table_libs(
                table_name,
                MAX_SEQ_LENGTH,
                INITIAL_FRAME,
                from_start = True,
                check = check
                )
            if len(saved_table) != MAX_SEQ_LENGTH:
                continue
            table_libs += [saved_table]
            table_cali += [raw_values]
            ''' libs of video  '''
            saved_vid = _get_video_libs(
                video_name,
                MAX_SEQ_LENGTH,
                (IMG_SIZE,IMG_SIZE),
                peaks
                )
            video_libs += [saved_vid]
            ''' libs of label  '''
            # label_libs += [label]
            index_in_table += [index]
        index_in_table = np.array(index_in_table)
        table_libs = np.array(table_libs)
        # table_cali = np.array(table_cali)
        video_libs = np.array(video_libs)                    
        # label_libs = np.array(label_libs)                    
        print('table_libs:', np.shape(table_libs))
        # print('table_cali:', np.shape(table_cali))
        print('video_libs:', np.shape(video_libs))
        # print('label_libs:', np.shape(label_libs))
        
    save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\table_libs_9_128_fixedv3.npy', table_libs) 
    # save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\table_cali_9_128_fixedv2.npy', table_cali) 
    save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\video_libs_9_128_fixedv3.npy', video_libs) 
    # save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_hkusz\label_libs_9_128.npy', label_libs) 
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
            ''' libs of  table '''    
            saved_table,indices,raw_values = _get_table_libs(
                table_name,
                MAX_SEQ_LENGTH,
                INITIAL_FRAME,
                from_start = True
                )
            table_libs += [saved_table]
            table_cali += [raw_values]
            ''' libs of video  '''
            saved_vid = _get_video_libs(
                video_name,
                MAX_SEQ_LENGTH,
                (IMG_SIZE,IMG_SIZE),
                indices)
            video_libs += [saved_vid]
            
            ''' libs of label  '''
            label_libs += [label]
        
        table_libs = np.array(table_libs)
        # table_cali = np.array(table_cali)
        video_libs = np.array(video_libs)                    
        # label_libs = np.array(label_libs)            
        print('table_libs:', np.shape(table_libs))
        # print('table_cali:', np.shape(table_cali))
        print('video_libs:', np.shape(video_libs))
        # print('label_libs:', np.shape(label_libs))
    save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\table_libs_9_128_fixedv3.npy', table_libs) 
    # save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\table_cali_9_128_fixedv3.npy', table_cali) 
    save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\video_libs_9_128_fixedv3.npy', video_libs) 
    # save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\label_libs_9_128_fixedv3.npy', label_libs)  
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
    para_name = 'other'
    index_df  = pd.read_excel(r"C:\Users\Olive\Desktop\HKU_PhD\Pui Kiu\筛查日志\疑似侧弯名单302.xlsx"
                 ,sheet_name='Index',usecols=['index','File No.'])
    sub_index = index_df['index'].tolist()
   
    for i,index in enumerate(sub_index):
        print(' para_name, index:',para_name,index)
        video_name = video_path+os.sep+para_name+'_'+str(int(index))+'_step1.mp4'
        if not os.path.exists(video_name):
            continue
        table_name = table_path+os.sep+para_name+'_'+str(int(index))+'_step_1.csv'
        if not os.path.exists(table_name):
            continue
        
        ''' libs of  table '''    
        saved_table,indices,raw_values = _get_table_libs(
            table_name,
            MAX_SEQ_LENGTH,
            INITIAL_FRAME,
            from_start = True
            )
       
        table_libs += [saved_table]
        table_cali += [raw_values]
        ''' libs of video  '''
        saved_vid = _get_video_libs(
            video_name,
            MAX_SEQ_LENGTH,
            (IMG_SIZE,IMG_SIZE),
            indices)
        video_libs += [saved_vid]
    
    table_libs = np.array(table_libs)
    # table_cali = np.array(table_cali)
    # video_libs = np.array(video_libs)                                     
    print('table_libs:', np.shape(table_libs))
    # print('table_cali:', np.shape(table_cali))
    # print('video_libs:', np.shape(video_libs))
    save(r'C:\Users\Olive\Desktop\HKU_PhD\Pui Kiu\筛查日志\2024pk_withGT\table_libs_9_128_fixedv3.npy', table_libs) 
    # save(r'C:\Users\Olive\.spyder-py3\npy_cam_high_dk\table_cali_9_128_fixedv2.npy', table_cali) 
    save(r'C:\Users\Olive\Desktop\HKU_PhD\Pui Kiu\筛查日志\2024pk_withGT\video_libs_9_128_fixedv3.npy', video_libs) 
    print('saved')
    return

if __name__ == '__main__':
    table_path = r"C:\Users\Olive\Desktop\HKU_PhD\Pui Kiu\筛查日志\2024pk_withGT\2024pk_tables"
    video_path = r"C:\Users\Olive\Desktop\HKU_PhD\Pui Kiu\筛查日志\2024pk_withGT\2024pk_videos"
    # table_path = r"C:\Users\Olive\Desktop\video_retrival\dk_table_refinedblaze"
    # video_path = r"C:\Users\Olive\Desktop\video_retrival\dk_video_refinedblaze"
    # run_sz(
    #     table_path = table_path,
    #     video_path = video_path,
    #     IMG_SIZE = 128,
    #     MAX_SEQ_LENGTH = 32,
    #     MIN_FILE_INDEX = 1,
    #     INITIAL_FRAME = 0 #300 in total
    #     )
    run_others(
        table_path = table_path,
        video_path = video_path,
        IMG_SIZE = 128,
        MAX_SEQ_LENGTH = 32,
        MIN_FILE_INDEX = 1,
        INITIAL_FRAME = 0 #300 in total
        )


    
    
   