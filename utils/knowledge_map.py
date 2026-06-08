# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 13:38:57 2023
Updated for YoLoV8, 17 landmarks

Created on Mar 22 14:07:07 2023

- Update the windows smooth and frame normalization
- Add STP features based on "A gait functional classification of adolescent idiopathic scoliosis (AIS) 
    based on spatio-temporal parameters (STP)"
    
SZ data dk_60 is not good, discard

embed_coors, #34
dis_embedders, # 106
ang_embedders, # 32
gait_phases, #66
@author: Olive
"""


# Get attention score at attention layer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import stats
from scipy import interpolate

from scipy.signal import savgol_filter
from scipy import signal
import math
np.random.seed(0)


class FullBodyPoseEmbedder(object):
  """Converts 3D pose landmarks into 3D embedding."""

  def __init__(self, torso_size_multiplier=2.5):
    # Multiplier to apply to the torso to get minimal body size.
    self._torso_size_multiplier = torso_size_multiplier

    # Names of the landmarks as they appear in the prediction.
    # self._landmark_names = [#33
    #     'nose',
    #     'left_eye_inner', 'left_eye', 'left_eye_outer',
    #     'right_eye_inner', 'right_eye', 'right_eye_outer',
    #     'left_ear', 'right_ear',
    #     'mouth_left', 'mouth_right',
    #     'left_shoulder', 'right_shoulder',
    #     'left_elbow', 'right_elbow',
    #     'left_wrist', 'right_wrist',
    #     'left_pinky_1', 'right_pinky_1',
    #     'left_index_1', 'right_index_1',
    #     'left_thumb_2', 'right_thumb_2',
    #     'left_hip', 'right_hip',
    #     'left_knee', 'right_knee',
    #     'left_ankle', 'right_ankle',
    #     'left_heel', 'right_heel',
    #     'left_foot_index', 'right_foot_index',
    # ]
    self._landmark_names_yolov8 = [#17                
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

  def __call__(self, landmarks):
    """Normalizes pose landmarks and converts to embedding
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances defined in `_get_pose_distance_embedding`.
    """
    assert landmarks.shape[0] == len(self._landmark_names_yolov8), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

    # Get pose landmarks.
    landmarks = np.copy(landmarks)
    
    # Normalize landmarks.
    landmarks = self._normalize_pose_landmarks(landmarks)

    # Get embedding.
    dis_embedding,ang_embedding = self._get_pose_distance_embedding(landmarks)
    return dis_embedding,ang_embedding,landmarks

  def _normalize_pose_landmarks(self, landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)
    
    # Normalize translation.
    pose_center = self._get_pose_center(landmarks)
    landmarks -= pose_center

    # Normalize scale.
    pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
    landmarks /= pose_size
    # pdb.set_trace()
    # # Multiplication by 100 is not required, but makes it eaasier to debug.
    landmarks *= 1e5

    return landmarks

  def _get_pose_center(self, landmarks):
    """Calculates pose center as point between hips."""
    
    left_hip = landmarks[self._landmark_names_yolov8.index('left_hip')]
    right_hip = landmarks[self._landmark_names_yolov8.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_pose_size(self, landmarks, torso_size_multiplier):
    """Calculates pose size.
    
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # This approach uses only 2D landmarks to compute pose size.
    landmarks = landmarks[:,:2]

    # Hips center.
    left_hip = landmarks[self._landmark_names_yolov8.index('left_hip')]
    right_hip = landmarks[self._landmark_names_yolov8.index('right_hip')]
    hips = (left_hip + right_hip) * 0.5

    # Shoulders center.
    left_shoulder = landmarks[self._landmark_names_yolov8.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names_yolov8.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)

    # Max dist to pose center.
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return torso_size* max_dist#max(torso_size * torso_size_multiplier, max_dist)

  def _get_pose_distance_embedding(self, landmarks):
    """Converts pose landmarks into 3D embedding.

    We use several pairwise 2D distances to form pose embedding. All distances
    include X and Y components with sign. We differnt types of pairs to cover
    different pose classes. Feel free to remove some or add new.
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 2).

    Result:
      Numpy array with pose embedding of shape (M, 2) where `M` is the number of
      pairwise distances.
      
    Notes: dis -> 106; ang -> 32
    """
    dis_embedding = np.concatenate([
        self._get_distance(
            self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
            self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),
        
        # starting nose
        self._get_distance_by_names(landmarks, 'nose', 'left_shoulder'),
        self._get_distance_by_names(landmarks, 'nose', 'right_shoulder'),
        self._get_distance_by_names(landmarks, 'nose', 'left_hip'),
        self._get_distance_by_names(landmarks, 'nose', 'right_hip'),
        # starting elbow
        self._get_distance_by_names(landmarks, 'left_elbow', 'left_shoulder'),
        self._get_distance_by_names(landmarks, 'left_elbow', 'left_hip'),
        self._get_distance_by_names(landmarks, 'left_elbow', 'left_knee'),
        self._get_distance_by_names(landmarks, 'left_elbow', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'left_elbow', 'right_shoulder'),
        self._get_distance_by_names(landmarks, 'left_elbow', 'right_hip'),
        self._get_distance_by_names(landmarks, 'left_elbow', 'right_knee'),
        self._get_distance_by_names(landmarks, 'left_elbow', 'right_ankle'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'left_shoulder'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'left_hip'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'left_knee'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'right_shoulder'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'right_hip'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'right_knee'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'right_ankle'),
        # starting shoulder
        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_hip'),
        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_knee'),
        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'left_shoulder', 'right_hip'),
        self._get_distance_by_names(landmarks, 'left_shoulder', 'right_knee'),
        self._get_distance_by_names(landmarks, 'left_shoulder', 'right_ankle'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'left_hip'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'left_knee'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_hip'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_knee'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),
        #starting hip
        self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
        self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'left_hip', 'right_knee'),
        self._get_distance_by_names(landmarks, 'left_hip', 'right_ankle'),
        self._get_distance_by_names(landmarks, 'right_hip', 'left_knee'),
        self._get_distance_by_names(landmarks, 'right_hip', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),
        
        # Cross body.
        self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
        self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
        self._get_distance_by_names(landmarks, 'left_shoulder', 'right_shoulder'),
        self._get_distance_by_names(landmarks, 'left_hip', 'right_hip'),
        self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),
        self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

        self._get_distance(
            self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
            self._get_average_by_names(landmarks, 'right_ankle', 'right_ankle')),
        self._get_distance(
            self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
            self._get_average_by_names(landmarks, 'left_ankle', 'left_ankle')),
        
        # Spine and Trunk (right shoulder remains more advanced relative to the line of progression)
        self._get_distance(
            self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
            self._get_average_by_names(landmarks, 'right_shoulder', 'right_shoulder')),
        self._get_distance(
            self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
            self._get_average_by_names(landmarks, 'left_shoulder', 'left_shoulder')),
        self._get_distance(
            self._get_average_by_names(landmarks, 'left_eye', 'right_eye'),
            self._get_average_by_names(landmarks, 'right_shoulder', 'right_shoulder')),
        self._get_distance(
            self._get_average_by_names(landmarks, 'left_eye', 'right_eye'),
            self._get_average_by_names(landmarks, 'left_shoulder', 'left_shoulder')),
      
    ])
    # pdb.set_trace()
    ang_embedding=np.array([
        #trunk rotation
        self._get_angle_by_names(landmarks,['left_shoulder','right_shoulder'],['left_hip','right_hip']),
        #'left_elbow_shoulder'
        self._get_angle_by_names(landmarks,['left_elbow','left_shoulder'],['right_elbow','right_shoulder']),
        self._get_angle_by_names(landmarks,['left_elbow','left_shoulder'],['right_hip','right_shoulder']),
        self._get_angle_by_names(landmarks,['left_elbow','left_shoulder'],['right_knee','right_hip']),
        self._get_angle_by_names(landmarks,['left_elbow','left_shoulder'],['right_ankle','right_knee']),
        
        self._get_angle_by_names(landmarks,['left_elbow','left_shoulder'],['left_hip','left_shoulder']),
        self._get_angle_by_names(landmarks,['left_elbow','left_shoulder'],['left_knee','left_hip']),
        self._get_angle_by_names(landmarks,['left_elbow','left_shoulder'],['left_ankle','left_knee']),
        #'right_elbow_shoulder'
        self._get_angle_by_names(landmarks,['right_elbow','right_shoulder'],['right_hip','right_shoulder']),
        self._get_angle_by_names(landmarks,['right_elbow','right_shoulder'],['right_knee','right_hip']),
        self._get_angle_by_names(landmarks,['right_elbow','right_shoulder'],['right_ankle','right_knee']),
        
        self._get_angle_by_names(landmarks,['right_elbow','right_shoulder'],['left_hip','left_shoulder']),
        self._get_angle_by_names(landmarks,['right_elbow','right_shoulder'],['left_knee','left_hip']),
        self._get_angle_by_names(landmarks,['right_elbow','right_shoulder'],['left_ankle','left_knee']),
        
        #left_shoulder_hip
        self._get_angle_by_names(landmarks,['left_shoulder','left_hip'],['right_shoulder','right_hip']),
        self._get_angle_by_names(landmarks,['left_shoulder','left_hip'],['right_knee','right_hip']),
        self._get_angle_by_names(landmarks,['left_shoulder','left_hip'],['right_ankle','right_knee']),
        
        self._get_angle_by_names(landmarks,['left_shoulder','left_hip'],['left_knee','left_hip']),
        self._get_angle_by_names(landmarks,['left_shoulder','left_hip'],['left_ankle','left_knee']),
        
        #right_shoulder_hip
        self._get_angle_by_names(landmarks,['right_shoulder','right_hip'],['right_knee','right_hip']),
        self._get_angle_by_names(landmarks,['right_shoulder','right_hip'],['right_ankle','right_knee']),
        
        self._get_angle_by_names(landmarks,['right_shoulder','right_hip'],['left_knee','left_hip']),
        self._get_angle_by_names(landmarks,['right_shoulder','right_hip'],['left_ankle','left_knee']),
        
        #left_hip_knee
        self._get_angle_by_names(landmarks,['left_hip','left_knee'],['right_hip','right_knee']),
        self._get_angle_by_names(landmarks,['left_hip','left_knee'],['right_ankle','right_knee']),
        self._get_angle_by_names(landmarks,['left_hip','left_knee'],['left_ankle','left_knee']),
        
        #right_hip_knee
        self._get_angle_by_names(landmarks,['right_hip','right_knee'],['right_ankle','right_knee']),
        self._get_angle_by_names(landmarks,['right_hip','right_knee'],['left_ankle','left_knee']),
        
        # targets between one points
        # elbow
        self._get_angle_by_names(landmarks,['left_hip','right_elbow'],['left_knee','right_elbow']),
        self._get_angle_by_names(landmarks,['right_hip','right_elbow'],['right_knee','right_elbow']),
        self._get_angle_by_names(landmarks,['right_hip','left_elbow'],['right_knee','left_elbow']),
        self._get_angle_by_names(landmarks,['left_hip','left_elbow'],['left_knee','left_elbow']),
        
        ])

    return dis_embedding,ang_embedding

  def _get_average_by_names(self, landmarks, name_from, name_to):
    lmk_from = landmarks[self._landmark_names_yolov8.index(name_from)]
    lmk_to = landmarks[self._landmark_names_yolov8.index(name_to)]
    return (lmk_from + lmk_to) * 0.5

  def _get_distance_by_names(self, landmarks, name_from, name_to):
    lmk_from = landmarks[self._landmark_names_yolov8.index(name_from)]
    lmk_to = landmarks[self._landmark_names_yolov8.index(name_to)]
    return self._get_distance(lmk_from, lmk_to)

  def _get_distance(self, lmk_from, lmk_to):
      return lmk_to - lmk_from

  def _get_angle_by_names(self,landmarks,v1:list,v2:list):
    v1=landmarks[self._landmark_names_yolov8.index(v1[1])]-landmarks[self._landmark_names_yolov8.index(v1[0])]
    v2=landmarks[self._landmark_names_yolov8.index(v2[1])]-landmarks[self._landmark_names_yolov8.index(v2[0])]
    theta1=math.atan2(v1[1],v1[0])-math.atan2(v2[1],v2[0])
    theta1=theta1*180/np.pi 
    if theta1>180.0:
        theta1=theta1-360
    elif theta1<-180.0:
        theta1=theta1+360
    return theta1


def cross_correlation(ele):
    corr = signal.correlate(ele[0], ele[1],mode='same',method='auto')
    corr /= np.max(corr)
    return corr

def gait_laging(embed_coor):
    structure_data=[]
    embed_coor=np.squeeze(np.array(embed_coor))
    ''' use x,y '''
    r_ankle = embed_coor[:,32:34]
    r_ankle = np.linalg.norm(r_ankle,axis=1)#get norm value of xyz
    l_ankle=embed_coor[:,30:32]
    l_ankle = np.linalg.norm(l_ankle,axis=1)
    
    r_knee=embed_coor[:,28:30]
    r_knee = np.linalg.norm(r_knee,axis=1)#get norm value of xyz
    l_knee=embed_coor[:,26:28]
    l_knee = np.linalg.norm(l_knee,axis=1)
    
    r_hip = embed_coor[:,26:28]
    r_hip = np.linalg.norm(r_hip,axis=1)
    l_hip = embed_coor[:,24:26]
    l_hip = np.linalg.norm(l_hip,axis=1)
   
    r_wrist=embed_coor[:,22:24]
    r_wrist = np.linalg.norm(r_wrist,axis=1)#get norm value of xyz
    l_wrist=embed_coor[:,20:22]
    l_wrist = np.linalg.norm(l_wrist,axis=1)
    
    r_elbow = embed_coor[:,18:20]
    r_elbow = np.linalg.norm(r_elbow,axis=1)#get norm value of xyz
    l_elbow = embed_coor[:,16:18]
    l_elbow = np.linalg.norm(l_elbow,axis=1)#get norm value of xyz
    
    r_shoulder=embed_coor[:,14:16]
    r_shoulder = np.linalg.norm(r_shoulder,axis=1)    
    l_shoulder=embed_coor[:,12:14]
    l_shoulder = np.linalg.norm(l_shoulder,axis=1)

    _var=[[r_wrist,l_wrist],[r_elbow,l_elbow],[r_ankle,l_ankle],[r_shoulder,l_shoulder],
          [r_knee,l_knee],[r_hip,l_hip],
          
          [r_wrist,l_ankle],[r_wrist,r_ankle],[l_wrist,l_ankle],[l_wrist,r_ankle],
          [r_wrist,l_elbow],[r_wrist,r_elbow],[l_wrist,l_elbow],[l_wrist,r_elbow],
          [r_wrist,l_shoulder],[r_wrist,r_shoulder],[l_wrist,l_shoulder],[l_wrist,r_shoulder],
          [r_wrist,l_knee],[r_wrist,r_knee],[l_wrist,l_knee],[l_wrist,r_knee],
          [r_wrist,l_hip],[r_wrist,r_hip],[l_wrist,l_hip],[l_wrist,r_hip],
          
          [r_elbow,l_ankle],[r_elbow,r_ankle],[l_elbow,l_ankle],[l_elbow,r_ankle],
          [r_elbow,l_shoulder], [r_elbow,r_shoulder], [l_elbow,l_shoulder], [l_elbow,r_shoulder],
          [r_elbow,l_knee],[r_elbow,r_knee],[l_elbow,l_knee],[l_elbow,r_knee],
          [r_elbow,l_hip], [r_elbow,r_hip], [l_elbow,l_hip], [l_elbow,r_hip],
          
           [r_ankle,l_shoulder],[r_ankle,r_shoulder],[l_ankle,l_shoulder],[l_ankle,r_shoulder],
           [r_ankle,l_knee],[r_ankle,r_knee],[l_ankle,l_knee],[l_ankle,r_knee],
           [r_ankle,l_hip],[r_ankle,r_hip],[l_ankle,l_hip],[l_ankle,r_hip],
          
           [r_shoulder,l_knee],[r_shoulder,r_knee], [l_shoulder,l_knee], [l_shoulder,r_knee],
           [r_shoulder,l_hip],[r_shoulder,r_hip],[l_shoulder,l_hip],[l_shoulder,r_hip],
          
           [r_knee,l_hip], [r_knee,r_hip], [l_knee,l_hip], [l_knee,r_hip],
          ]
    group=[]
    for ele in _var:
       corr=cross_correlation(ele)
       group+=[corr*100]
    structure_data.append(group)
    # pdb.set_trace()
    return structure_data

def GetAllFeatures(samples):#video_len   
    PoseEmbedder = FullBodyPoseEmbedder() 
    ## data processing ##
    embed_coor,embed_coors= [],[]
    gait_phases=[]
    dis_embedder,ang_embedder=[],[]
    dis_embedders,ang_embedders=[],[]
    
    for row in range(len(samples)):
        # pdb.set_trace()
        landmarks=samples[row,:].reshape(17,2)
        dis_embedding,ang_embedding,ed_coor=PoseEmbedder(landmarks)
        dis_embedder.extend(dis_embedding)
        ang_embedder.extend(ang_embedding)
        embed_coor.extend(ed_coor.reshape(1,-1))
        
    dis_embedders.extend([np.array(dis_embedder).reshape(-1,int(len(dis_embedding)))])
    ang_embedders.extend([np.array(ang_embedder).reshape(-1,int(len(ang_embedding)))])
    
    embed_coors.extend([np.array(embed_coor)])
    '''get supporting phase of embed_coors'''
    gait_phases.extend([np.transpose(np.squeeze(
        np.array(gait_laging(np.array(embed_coors[-1])))))])#
    
    
    embed_coors = np.squeeze(embed_coors)
    embed_coors = (embed_coors-np.mean(embed_coors))/np.std(embed_coors)
    dis_embedders = np.squeeze(dis_embedders)
    dis_embedders = (dis_embedders-np.mean(dis_embedders))/np.std(dis_embedders)
    
    ang_embedders = np.squeeze(ang_embedders)
    ang_embedders = (ang_embedders-np.mean(ang_embedders))/np.std(ang_embedders)
    
    gait_phases = np.squeeze(gait_phases)
    gait_phases = (gait_phases-np.mean(gait_phases))/np.std(gait_phases)

    Inputdata=np.concatenate((embed_coors, #34
                              dis_embedders, # 106
                              ang_embedders, # 32
                              gait_phases, #66
                              ),
                             axis=-1)
    
    return Inputdata






