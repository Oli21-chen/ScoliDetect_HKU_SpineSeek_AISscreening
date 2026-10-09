# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:08:02 2023
shift+alt+E : run selected codes
ctrl+/ : mute code
shift+F10 : run all code
shift+F9 : debug

In OpenCV, BGR sequence is used instead of RGB. This means the first channel is blue, the second channel is green,
and the third channel is red.
@author: Olive
"""

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import pdb
import cv2
import pandas as pd
import fnmatch
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
IMG_SIZE = 720
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
POSE_CONNECTIONS = frozenset([(11, 12), (11, 13),(12, 14), (11, 23), (12, 24),
                              (23, 24), (23, 25),(24, 26), (25, 27), (26, 28),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), #left hand
                              (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), #right hand
                              (30, 32),(28, 30), (28, 32),#right foot
                              (29, 31),  (27, 31),(27, 29)])#, left foot

def rescale_frame(frame, percent=75):
    dy = int(frame.shape[0] * percent/ 100)
    dx = int(frame.shape[1] * percent/ 100)
    dim = (dx, dy)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)#cv2.INTER_CUBIC,cv2.INTER_NEAREST

class posedetection(object):
    def __init__(self, video_path):
        self.frame_flag = False
        self.video_path = video_path
        self.current_frame = 0
        self.total_landmarks = 33
        self.cap = cv2.VideoCapture(self.video_path)
        '''Set the starting frame '''
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,80)

    def GetDirect(self,landmarks):
        dirc_ear = landmarks[7][0] - landmarks[8][0]
        dirc_eye = landmarks[2][0] - landmarks[5][0]
        dirc_inner_eye = landmarks[1][0] - landmarks[4][0]
        flag = np.array([dirc_ear,dirc_eye,dirc_inner_eye])
        if len(flag[flag>0])/3 > 0.5:
            dirc_flag = 1
        else:
            dirc_flag = -1
        # print('direction: ', dirc_flag)
        return dirc_flag

    def blazepose(self, sample_index, person_index,resize=(IMG_SIZE,IMG_SIZE)):
        i = 0  ##for record each frame
        direct_flag_past_t = 0
        # person_index = person_index
        raw_frame_no = []
        totalframerate = int(self.cap.get(cv2.CAP_PROP_FPS))
        totalframecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        videoWriter = cv2.VideoWriter(r'C:\Users\Olive\Desktop\sz_{}_step1.mp4'.format(person_index),
                                      cv2.VideoWriter_fourcc(*'mp4v'),  # *'MP4V',#('m', 'p', '4', 'v'),
                                      30, (1080,1920))#int(self.cap.get(3)), int(self.cap.get(4))
        with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                min_detection_confidence=0.9,
                min_tracking_confidence=0.9
        ) as pose:
            while self.cap.isOpened():
                
                success, image = self.cap.read()
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                # print("The current frame is ", self.current_frame)
                # print("The past frame is ", self.past_frame)
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)#cv2.ROTATE_90_COUNTERCLOCKWISE
                  # wid=720,hei=1280,wid=1080,hei=1920
                image_height, image_width, _ = image.shape
                # To improve performance, optionally mark the image as not writeable to pass by reference.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # make detection
                results = pose.process(image)

                if results.pose_landmarks is None:  # People disappear in Video
                    if self.current_frame <= totalframecount:  # Sometimes, people disappeared during test
                        print('Did not detect any information')
                else:
                    # Draw the pose annotation on the image.
                    annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    annotated_image.flags.writeable = True
                    # mp_drawing.draw_landmarks(
                    #     annotated_image,
                    #     results.pose_landmarks,
                    #     POSE_CONNECTIONS,
                    #     landmark_drawing_spec= mp_drawing_styles.get_default_pose_landmarks_style()
                    # )
                    # Show the frame index on saved videos
                    # annotated_image = cv2.putText(annotated_image, str('Frame number:' + str(self.current_frame)),
                    #                     tuple(np.multiply([0.1, 0.1], [image_height, image_width]).astype(int)),
                    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    annotated_image.flags.writeable = False
                    # The ratio of good points should be bigger than threshold
                    landmarks = np.array([np.array([results.pose_landmarks.landmark[i].x * image_width,
                                                    results.pose_landmarks.landmark[i].y * image_height,
                                                    results.pose_landmarks.landmark[i].z * image_width])
                                          for i in range(self.total_landmarks)])

                    direct_flag_curr_t = self.GetDirect(landmarks)
                    ''' 'Save frames into videos ' '''
                    if direct_flag_past_t != 0:
                        if direct_flag_curr_t == 1 and direct_flag_past_t * direct_flag_curr_t > 0: # coming =1, going =-1
                            videoWriter.write(annotated_image)  # save embedded videos
                            raw_frame_no += [self.current_frame]
                    direct_flag_past_t = direct_flag_curr_t
                    ''' show the resized images'''
                    annotated_image = cv2.resize(annotated_image, resize)
                    cv2.imshow('MediaPipe Pose', annotated_image)  # cv2.flip(image, 1)
                    # checking condi
                    if len(raw_frame_no) >= 30*3 and direct_flag_curr_t == -1:
                        break
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        self.cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()
        print("The total frame rate in this video is ", totalframerate)
        print("The total number of frames in this video is ", totalframecount)
        print('image_height: {}  image_width:{}'.format(image_height, image_width))
        return landmarks


if __name__ == '__main__':
    para_data = 'sz'
    
    video_path = r'C:\Users\Olive\Desktop\video'
    directory = os.listdir(video_path)
    num = 155
    # index=np.arange(10,40,1)
    # index=np.setdiff1d(index,exclusion_index)

    # record_fname=list()
    # # file_index=['S_'+str(i)+'_high_Trim' for i in index]#keep file name in order\
    # file_index = ['0' + str(i)+'_top_co_1' for i in index]  # keep file name in order
    
    # for searchstring in file_index:
    #     record_fname.extend([[video_path+os.sep+fname] for fname in directory if searchstring in fname])
    # sub_index=1
    # frame_set = []
    # for i,person_idx in zip(record_fname[:],index):#2 clockwise
    #     if sub_index == 2:#3
    #         sub_index=1
    #     vpath=str(','.join(i))
    #     print(vpath)
    #     features = posedetection(vpath)
    #     landmarks = features.blazepose(sub_index, person_idx)
    #     sub_index+=1
    if para_data == 'sz' :
        label = pd.read_excel(r'C:\Users\Olive\.spyder-py3\Data_structured\Label_SZgait_4video.xlsx'
                     ,usecols=['File No.','name'])
        label = label.values
        for index,file_info in label:
            # if index == 57:
            # pdb.set_trace()
            for fname in directory:
                if file_info in fname:
                    print(fname)
                    vpath = video_path+os.sep+fname
                    print(vpath)
                    features = posedetection(vpath)
                    landmarks = features.blazepose(1, index)
                     


    # for i in [40,41]:
    #     file_index = ['0'+str(i)+'_top_co_1']
    #     for searchstring in file_index:
    #         for fname in directory:
    #             if searchstring in fname:
    #                 vpath = video_path+os.sep+fname
    #                 print(vpath)
    #                 features = posedetection(vpath)
    #                 landmarks = features.blazepose(1, i)
                
                
    
    
    