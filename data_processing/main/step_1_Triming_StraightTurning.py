"""
Created on Thu Jun  8 11:42:29 2023
shift+alt+E : run selected codes
ctrl+/ : mute code
shift+F10 : run all code
shift+F9 : debug

In OpenCV, BGR sequence is used instead of RGB. This means the first channel is blue, the second channel is green,
and the third channel is red.
- YoLoV8 pose is Bottom- up approaches and it Needs Pytorch
https://docs.ultralytics.com/modes/predict/#arguments

输入视频后，resize到1080p

# args:
    1. automatically select going straight and body turning.
    2. conda install m2-base： run linux commend in conda
    
YOLOv8 pose models use the -pose suffix, i.e. yolov8n-pose.pt. These models 
are trained on the COCO keypoints dataset and are suitable for a variety of 
pose estimation tasks.

In the default YOLOv8 pose model, there are 17 keypoints, each representing
 a different part of the human body. Here is the mapping of each index to its
 respective body joint:

0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 
6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist
 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle
@author: Olive
"""
import torch #need pytorch to use gpu
import csv
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import pdb
import time
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from ultralytics import YOLO

landmark_indices_blazepose = [0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28] 

landmark_names_yolov8 = [
        'nose',
        'left_eye','right_eye',
        'left_ear','right_ear',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
    ]

def rescale_frame(frame, percent=75):
    dy = int(frame.shape[0] * percent/ 100)
    dx = int(frame.shape[1] * percent/ 100)
    dim = (dx, dy)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)#cv2.INTER_CUBIC,cv2.INTER_NEAREST


class posedetection(object):
    def __init__(self,para_dataname, video_path,saved_video_path,csv_path,person_id):
        self.para_dataname = para_dataname
        self.video_path = video_path
        self.saved_video_path = saved_video_path
        self.csv_path = csv_path
        self.person_id = person_id
        self.current_frame = 0
        self.blaze_total_landmarks = 33
        self.yolo_total_landmarks = 17
        self.cap = cv2.VideoCapture(self.video_path)
        self.Pose_YoloV8model = YOLO(r'.\yolov8x-pose-p6.pt')  # build a new model from YAML
        self.blazepose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                min_detection_confidence=0.9,
                min_tracking_confidence=0.9)
        
       # r'C:\Users\Olive\Desktop\sz_{}_step1.mp4'
        self.videoWriter = cv2.VideoWriter(saved_video_path+'\\'+self.para_dataname+'_{}_step1.mp4'.format(self.person_id),
                                      cv2.VideoWriter_fourcc(*'mp4v'),  # *'MP4V',#('m', 'p', '4', 'v'),
                                      30, (1080,1920))#int(self.cap.get(3)), int(self.cap.get(4))
        '''Set the starting frame '''
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES,80)

    def _direction_flags(self,
                         landmarks_blazepose,
                         landmarks_yolo,
                         boxes_yolov8,
                         img_size,
                         turning_th = 15):
        ''' condition go_forward, go_backward, turning
        - get forward/backward by the mesh points direction.
        - align top-down and bottom-up together to enhance the detection stability
        '''
        # pdb.set_trace()
        nums,scores = [],[]
        (w,h) = img_size
        
        if len(np.shape(landmarks_yolo))==2:
            landmarks_yolo =landmarks_yolo[np.newaxis,...]
        
        if len(landmarks_yolo[0])!=self.yolo_total_landmarks:
            # pdb.set_trace()
            print('-------- skip in flag -----------' )
            dirc_flag=None
            landmarks_yolo=None
            boxes_yolo=None
            return dirc_flag,landmarks_yolo,boxes_yolo
        else:
            # pdb.set_trace()
            if landmarks_blazepose is None:
                for num, multi_landmarks_yolo in enumerate(landmarks_yolo):
                    
                    lr_shoulder = 0.5*(multi_landmarks_yolo[5][0] + multi_landmarks_yolo[6][0])
                    lr_hip = 0.5*(multi_landmarks_yolo[11][0] + multi_landmarks_yolo[12][0])
                    discriminative_shoulder_hip = np.sqrt((lr_shoulder-w*0.5)**2)+np.sqrt((lr_hip-w*0.5)**2)
                    nums+=[num]
                    scores+=[discriminative_shoulder_hip]
                # pdb.set_trace()    
                select_id = np.argmin(scores)
                
            else:    
                # pdb.set_trace()
                for num, multi_landmarks_yolo in enumerate(landmarks_yolo):
                    
                    discriminative_shoulder = (np.linalg.norm(landmarks_blazepose[11][:2]-multi_landmarks_yolo[5])+
                                                np.linalg.norm(landmarks_blazepose[12][:2]-multi_landmarks_yolo[6]))*0.5
                    discriminative_hip = (np.linalg.norm(landmarks_blazepose[23][:2]-multi_landmarks_yolo[11])+
                                                np.linalg.norm(landmarks_blazepose[24][:2]-multi_landmarks_yolo[12]))*0.5
                    nums+=[num]
                    scores+=[discriminative_shoulder+discriminative_hip]
                # pdb.set_trace()    
                select_id = np.argmin(scores)
                
            # pdb.set_trace()      
            boxes_yolo = boxes_yolov8[select_id,...]
            landmarks_yolo = np.squeeze(landmarks_yolo[select_id,...])
            
            lr_shoulder = landmarks_yolo[5][0]-landmarks_yolo[6][0]
            lr_elbow = landmarks_yolo[7][0] - landmarks_yolo[8][0]
            lr_hip = landmarks_yolo[11][0]-landmarks_yolo[12][0]
            lr_knee = landmarks_yolo[13][0]-landmarks_yolo[14][0]
            # lr_ankle = landmarks_yolo[15][0] - landmarks_yolo[16][0]
            
            flag_all = np.array([lr_shoulder, lr_hip, lr_elbow, lr_knee])
            # print(flag_all)
            # pdb.set_trace()
            if len(flag_all[flag_all>0]) == len(flag_all) and np.all(abs(flag_all)>turning_th): #np.all(flag_all>0):
                dirc_flag = 1
                # print('\n Backing \t')
            elif len(flag_all[flag_all<0]) == len(flag_all) and np.all(abs(flag_all)>turning_th): #np.all(flag_all<0):
                dirc_flag = 2
                # print('\n Going \t')
            else:
                dirc_flag = 3
                # print('\n Turning \t')

            return dirc_flag, landmarks_yolo,boxes_yolo
   
    def _time_difference(self,
                         landmarks_yolo,
                         past_landmarks_yolo,
                         curr_boxes_yolo,
                         past_boxes_yolo,
                         dirc_flag,
                         score_th = 1.2,
                         td_th = 5
                         ):

        x_curr,y_curr,_,_ = np.squeeze(curr_boxes_yolo.xywh.detach().cpu().numpy())
        x_past,y_past,_,_ = np.squeeze(past_boxes_yolo.xywh.detach().cpu().numpy())
        score_x = np.sqrt((x_curr-x_past)**2)
        score_y = np.sqrt((y_curr-y_past)**2)
        
        # pdb.set_trace()
        # td_knee_l_x = np.sqrt(abs(landmarks_yolo[13][0]-past_landmarks_yolo[13][0])**2)
        td_knee_l_y = np.sqrt(abs(landmarks_yolo[13][1]-past_landmarks_yolo[13][1])**2)
        
        # td_knee_r_x = np.sqrt(abs(landmarks_yolo[14][0]-past_landmarks_yolo[14][0])**2)
        td_knee_r_y =np.sqrt(abs(landmarks_yolo[14][1]-past_landmarks_yolo[14][1])**2)
        
        # td_ankle_l_x = np.sqrt(abs(landmarks_yolo[15][0]-past_landmarks_yolo[15][0])**2)
        td_ankle_l_y = np.sqrt(abs(landmarks_yolo[15][1]-past_landmarks_yolo[15][1])**2)
        
        # td_ankle_r_x = np.sqrt(abs(landmarks_yolo[16][0]-past_landmarks_yolo[16][0])**2)
        td_ankle_r_y = np.sqrt(abs(landmarks_yolo[16][1]-past_landmarks_yolo[16][1])**2)
        
        td_all = np.array([td_knee_l_y,td_knee_r_y,td_ankle_l_y,td_ankle_r_y])
        # print('td:',td_all)
        if score_x < score_th and score_y <score_th and np.all(td_all<td_th) :# and dirc_flag !=3:
            # print('- Staying -')
            return False

        else:
            # print('+ Moving +')
            return True
            
    
    def save_frames_flag(self,
                         frame,
                         flag,
                         moving_flag,
                         only_going = False,
                         only_backing = False,
                         only_turning = False,
                         ):
        # pdb.set_trace()  
        if moving_flag:
            if only_going:
                if flag == 2:
                    self.videoWriter.write(frame)  # save embedded videos
                    return flag
                else:
                    return None
            elif only_backing:
                if flag == 1:
                    self.videoWriter.write(frame)  # save embedded videos
                    return flag
                else:
                    return None
            elif only_turning:
                if flag == 3:
                    self.videoWriter.write(frame)  # save embedded videos
                    return flag
                else:
                    return None
            else:
                self.videoWriter.write(frame)
                return flag
        else:
            return None
    
    def yolo2blaze(self,
            landmarks_yolo,
            landmarks_blazepose,
            current_frame
            ):
        # if current_frame == 230:
        #     pdb.set_trace()
        if landmarks_blazepose is not None:
            tem = np.zeros((5,2))
            for i in range(tem.shape[1]):
                x = landmarks_blazepose[landmark_indices_blazepose[5:],i]
                y = landmarks_yolo[5:,i]
                slope, intercept = np.polyfit(x,y, 1)
                tem[:,i] =  slope*landmarks_blazepose[0:5,i] + intercept
            
            mask = landmarks_yolo[:5]==0
            new_top_10 = np.where(mask,tem,landmarks_yolo[:5])
        else:
            new_top_10 = landmarks_yolo[:5]
        return new_top_10
    
    def detection(self,
                  max_seq,
                  activate_top_down=True,
                  save_video = False,
                  save_table = False,
                  para_only_going = False,
                  para_only_backing = False,
                  para_only_turning = False,
                  moving_flag = False,
                  initiate = False,
                  good_quality = False,
                  save_blazepose = False,
                  resolution = '1080p',
                  ):
        
        saved_flags_log = []
        past_landmarks_yolo = None
        past_boxes_yolo = None
        totalframerate = int(self.cap.get(cv2.CAP_PROP_FPS))
        if totalframerate>35: # if frame rate > 30 HZ, change max_seq
            max_seq*=2
        totalframecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        TabularFile_path=self.csv_path+'\\' + self.para_dataname+'_{}_'.format(self.person_id)+'step_1.csv'
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            # image_height, image_width, _ = image.shape
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if resolution == '1080p':
                image_width = 1080
                image_height = 1920
            img = cv2.resize(image, (image_width,image_height),interpolation =cv2.INTER_AREA)
            
            ''' ÝoLo pose  '''
            results_yolo = self.Pose_YoloV8model.predict(img,
                                                         conf=0.2,
                                                         iou=0.5,
                                                         verbose=False,
                                                         agnostic_nms=True,
                                                         device=0,
                                                         max_det=5,
                                                         classes=0)

            keypoints = results_yolo[0].keypoints  # Masks object
            boxes_yolov8 = results_yolo[0].boxes  # Masks object
            landmarks_yolov8 = np.squeeze(keypoints.xy.detach().cpu().numpy())  # x, y keypoints (pixels), (num_dets, num_kpts, 2/3), the last dimension can be 2 or 3, depends the model.
            # b = keypoints.xyn  # x, y keypoints (normalized), (num_dets, num_kpts, 2/3)
            # c = keypoints.conf  # confidence score(num_dets, num_kpts) of each keypoint if the last dimension is 3.
            # d = keypoints.data  # raw keypoints tensor, (num_dets, num_kpts, 2/3) 
            # Visualize the results on the frame
            if landmarks_yolov8 is None: #(num_dets, num_kpts, 2/3)
                # pdb.set_trace()
                print('-------- skip -----------' )
                landmarks_yolov8 = None
            else:
                annotated_frame_yolo = results_yolo[0].plot()
                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", rescale_frame(annotated_frame_yolo,40))
            
            ''' Blase Pose '''
            if activate_top_down:
                # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)#cv2.ROTATE_90_COUNTERCLOCKWISE
                  # wid=720,hei=1280,wid=1080,hei=1920
                # To improve performance, optionally mark the image as not writeable to pass by reference.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # make detection
                results_blaze = self.blazepose.process(img)
                # cv2.imshow( 'cropped',rescale_frame(image1,40))
                if results_blaze.pose_landmarks is None:  # People disappear in Video
                    if self.current_frame <= totalframecount:  # Sometimes, people disappeared during test
                        print('Did not detect any information')
                        landmarks_blazepose = None
                    
                else:
                    # Draw the pose annotation on the image.
                    annotated_frame_blaze = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    annotated_frame_blaze.flags.writeable = True
                    mp_drawing.draw_landmarks(
                        annotated_frame_blaze,
                        results_blaze.pose_landmarks,
                        POSE_CONNECTIONS,
                        landmark_drawing_spec= mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    # Show the frame index on saved videos
                    annotated_frame_blaze = cv2.putText(annotated_frame_blaze, str('Frame number:' + str(self.current_frame)),
                                        tuple(np.multiply([0.1, 0.1], [image_height, image_width]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    annotated_frame_blaze.flags.writeable = False
                    # The ratio of good points should be bigger than threshold
                    landmarks_blazepose = np.array([np.array([results_blaze.pose_landmarks.landmark[i].x * image_width,
                                                    results_blaze.pose_landmarks.landmark[i].y * image_height,
                                                    results_blaze.pose_landmarks.landmark[i].z * image_width])
                                          for i in range(self.blaze_total_landmarks)])
                    annotated_frame_blaze = rescale_frame(annotated_frame_blaze,40)
                    cv2.imshow('MediaPipe Pose', annotated_frame_blaze)  # cv2.flip(image, 1)

                
            ''' get flag based on the landmarks '''
            if landmarks_yolov8 is None:
                print('skip this frame:',self.current_frame)
            else:    
                curr_direction_flag,landmarks_yolo,boxes_yolo = self._direction_flags(
                    landmarks_blazepose,
                    landmarks_yolov8,
                    boxes_yolov8,
                    (image_width,image_height))
                
                ''' To check whether target moving '''
                if (past_boxes_yolo is not None and past_landmarks_yolo is not None and
                    landmarks_yolo is not None and curr_direction_flag is not None): 
                    moving_flag = self._time_difference(landmarks_yolo,past_landmarks_yolo,
                                                        boxes_yolo,past_boxes_yolo,
                                                        curr_direction_flag)
                ''' Start to record when people standing near to camera and turning to go
                in this project, record after turning and subject who is standing near camera
                '''
                # print('curr_direction_flag:', curr_direction_flag)
                # pdb.set_trace()
                # print('(landmarks_yolo[15]',landmarks_yolo[15])
                if (curr_direction_flag==3 
                    # and landmarks_yolo[7][0]<landmarks_yolo[8][0] 
                # and landmarks_yolo[9][0]<landmarks_yolo[10][0]):#  \
                    and abs(landmarks_yolo[15][1])>abs(image_height*0.45)):#0.725
                    initiate = True
                # for people go over the initial point,check shoulder
                elif  curr_direction_flag==3 and \
                    abs(landmarks_yolo[15][1])==0 and abs(landmarks_yolo[5][1])> abs(image_height*0.5):
                    initiate = True
                # elif (curr_direction_flag==3 and 
                #     landmarks_yolo[7][0]<landmarks_yolo[8][0] and 
                #     landmarks_yolo[9][0]<landmarks_yolo[10][0]):
                #     # turning and walking away, left_x < right_x
                #     initiate = True
                ''' check good_quality for ensuring that all points are visiable  '''
                if (landmarks_yolo is not None 
                    and np.all(landmarks_yolo[5:] != 0)
                    and landmarks_blazepose is not None):
                    good_quality = True
                
                past_boxes_yolo = boxes_yolo
                past_landmarks_yolo = landmarks_yolo
                # pdb.set_trace()    
                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if (save_video ==True and initiate == True and good_quality == True
                    ):
                    saved_flag = self.save_frames_flag(
                        frame = img,
                        flag = curr_direction_flag,
                        moving_flag = moving_flag,
                        only_going = para_only_going,
                        only_backing = para_only_backing,
                        only_turning = para_only_turning,
                        )
                    if saved_flag is not None:  
                        # pdb.set_trace()
                        saved_flags_log += [saved_flag]    
                        '''
                        save landmarks, only saving table of saved frames;
                        when people walking away, yolo will not estimate the occluded point;
                        so use blazepose coor to estimate, here includes a linear fitting for
                        top[5] landmarks, which are facial info. 
                        '''
                        if save_table == True:
                            if save_blazepose == False:
                                # print('yolo-',landmarks_yolo)
                                if np.count_nonzero(landmarks_yolo[:5] == 0) > 0:
                                    landmarks_yolo[:5] = self.yolo2blaze(landmarks_yolo,landmarks_blazepose,self.current_frame)
                                landmarks_yolo = list(np.append(landmarks_yolo,self.current_frame).flatten())
                            else: # save blazepose
                                # pdb.set_trace()
                                landmarks_yolo = list(np.append(
                                    landmarks_blazepose[landmark_indices_blazepose,:-1],
                                    self.current_frame).flatten())
                            # print('blazeyolo-',landmarks_yolo)
                            # assert len(landmarks_yolov8) == 17*2+1
                            with open(TabularFile_path,mode='a',newline='') as f:
                                # pdb.set_trace()
                                csv_writer=csv.writer(f,delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(landmarks_yolo)
                good_quality=False            
                            
            if cv2.waitKey(5) & 0xFF == ord('q') or len(saved_flags_log) >= max_seq:
                # pdb.set_trace()
                break
            
        self.cap.release()
        self.videoWriter.release()
        cv2.destroyAllWindows()
        print("The total frame rate in this video is ", totalframerate)
        print("The total number of frames in this video is ", totalframecount)
        print('image_height: {}  image_width:{}'.format(image_height, image_width))
        # pdb.set_trace()
        assert len(saved_flags_log) == max_seq, f'{len(saved_flags_log)}'
        return landmarks_yolo
    
'''Create CSV'''
def create_CSV2(para_dataname,person_id,csv_path,num_coords= 17):
    landmarks=[]
    for val in range(1,num_coords+1):
        landmarks += list(['x{}'.format(val),'y{}'.format(val)])
    landmarks += list(['frames'])
    Table_path=csv_path+'\\'+ para_dataname +'_{}_'.format(person_id)+'step_1.csv'
    # print(Table_path)
    with open(Table_path,mode='w',newline='') as f:
        csv_writer=csv.writer(f,delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

def run(para_dataname,
        saved_video_path,
        csv_path,
        video_path,
        max_seq,
        save_blazepose = False,
        ):
    
    directory = os.listdir(video_path)
    # pdb.set_trace()
    print(' Runing...')

    if para_dataname == 'sz' :
        label = pd.read_excel(r"C:\Users\Olive\Desktop\video_retrival\video_retrival\Label_SZgait_4video.xlsx"
                     ,usecols=['File No.','name'])
        label = label.values
        # pdb.set_trace()
        for person_id,file_info in label:
            # print('person id: ',person_id)
            for fname in directory[:]:
                if file_info in fname:
                    print()
                    # print('file_info: ',file_info)
                    t1 = time.time()
                    create_CSV2(para_dataname,person_id,csv_path)
                    print('\n index in table:',fname)
                    vpath = video_path+os.sep+fname
                    print('\n video_name:',vpath)
                    features = posedetection(para_dataname,vpath,saved_video_path,csv_path,person_id)
                    landmarks = features.detection(
                                                    max_seq=max_seq,
                                                    activate_top_down = True,
                                                    save_video = True,
                                                    save_table = True,
                                                    para_only_going = False,
                                                    para_only_backing = False,
                                                    para_only_turning = False,
                                                    save_blazepose = save_blazepose
                                                    )
                    t2 = time.time()
                    print('\n {} Take:{} sec'.format(fname,t2-t1))
    
    if para_dataname == 'dk' :
     label = pd.read_excel(r'C:\Users\Olive\.spyder-py3\Database\Label_DKgait_4video.xlsx'
                  ,usecols=['File No.'])
     label = np.squeeze(label.values)#starting label changed here
     for person_id in label:
          # pdb.set_trace()
         for fname in directory:
             # pdb.set_trace()
             if '_'+str(person_id)+'_' in fname:
                 create_CSV2(para_dataname,person_id,csv_path)
                 print('\n index in table:',fname)
                 vpath = video_path+os.sep+fname
                 print('\n video_name:',vpath)
                 features = posedetection(para_dataname,vpath,saved_video_path,csv_path,person_id)
                 landmarks = features.detection(
                                                max_seq=max_seq,
                                                activate_top_down = True,
                                                save_video = True,
                                                save_table = True,
                                                para_only_going = False,
                                                para_only_backing = False,
                                                para_only_turning = False,
                                                save_blazepose = save_blazepose
                                                )
    if para_dataname == 'other' :
     num_files = len(os.listdir(video_path))
     index_df  = pd.read_excel(r"C:\Users\Olive\Desktop\HKU_PhD\Pui Kiu\筛查日志\2024疑似侧弯名单302.xlsx"
                  ,sheet_name='Index',usecols=['index','File No.'])
     # pdb.set_trace()
     # Get the search codes directly (they're already in the correct format)
     search_codes = index_df['File No.'].tolist()
     # search_index = index_df['index'].tolist()
     
     for person_id, search_code in enumerate(search_codes,1):
        # Construct filename pattern
        print('ID num: ',person_id)
        file_pattern = f"H_{search_code}.mp4"
        matching_file = None
        for fname in os.listdir(video_path):
            if fname == file_pattern:
                matching_file = fname
                break
        if matching_file:
    #  for person_id,fname in zip(range(1,num_files+1),os.listdir(video_path)):
            create_CSV2(para_dataname,person_id,csv_path)
            print('\n index in table:',fname)
            vpath = video_path+os.sep+fname
            print('\n video_name:',vpath)
            features = posedetection(para_dataname,vpath,saved_video_path,csv_path,person_id)
            landmarks = features.detection(
                                            max_seq=max_seq,
                                            activate_top_down = True,
                                            save_video = True,
                                            save_table = True,
                                            para_only_going = False,
                                            para_only_backing = False,
                                            para_only_turning = False,
                                            save_blazepose = save_blazepose
                                            )
    print(' Finished.')
    return 

if __name__ == '__main__':
    '''
    args:
        max_seq: 32 # maximum number of frames
        only_going: False
        only_backing: False
        only_turning: False
        yolo_max_det: 3 # maximum detection of people of yolo
        activate_top_down: True #open blazepose for more accurate location for one of multipersons detection
    
    Notes:
        didnt test the output landmarks with deactivate_top_down
    '''
    
    para_saved_video_path = r"C:\Users\Olive\Desktop\video"
    para_saved_csv_path= r"C:\Users\Olive\Desktop\table"
    para_read_video_path = r"C:\Users\Olive\Desktop\raw"    
    run(
        para_dataname = 'sz',
        saved_video_path = para_saved_video_path,
        video_path = para_read_video_path,
        csv_path = para_saved_csv_path,
        max_seq= 300,
        save_blazepose = False
        )
   

    