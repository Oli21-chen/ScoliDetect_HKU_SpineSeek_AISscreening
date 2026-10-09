# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:11:03 2023

@author: Olive
"""

from sklearn.model_selection import KFold
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np 
from numpy import load
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import pdb
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from model_resnet3d import Resnet3DBuilder
from model_r2plus1d import Resnetr2plus1dBuilder
from model_STM import STMBuilder
COBB_TH =15


def plot_metrics(history):
    # metrics =['loss', 'categorical_accuracy','Specificity','Sensitivity','precision_co','precision_pt']
    metrics =['loss', 'binary_accuracy']#,'co_recall','pt_recall'
    plt.figure(figsize=(15,10))
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(3,2,n+1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                  linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
          plt.ylim([0, plt.ylim()[1]+0.1])
        elif metric == 'categorical_accuracy':
          plt.ylim([0,1])
        else:
          plt.ylim([0,1])
    plt.legend()
    plt.tight_layout()
    

''' 'HKUSZ'  '''
sz_vid_raw =load(r'D:\HKUSZH_gaitvideo\video_libs.npy',allow_pickle=True)#SZ_TrimmedVideo,dk_vid_label
sz_cobb_label =load(r'D:\HKUSZH_gaitvideo\label_libs.npy',allow_pickle=True)#SZ_TrimmedVideo_label

SZ_vData , SZ_cobb_labels = [], []
for idx,(i,j) in enumerate(zip(sz_vid_raw,sz_cobb_label)):
    # pdb.set_trace()
    i = np.reshape(i,[-1,224,224,3])[0:32,...]

    
    if np.shape(i)!=(32,224,224,3):#(np.shape(sz_tab_raw)[1],np.shape(sz_tab_raw)[2]):
        print(idx,np.shape(i))
    else:
        SZ_vData += [i]
        SZ_cobb_labels += [j]
        
        
SZ_vData = np.array(SZ_vData)
cobb_labels = np.array(SZ_cobb_labels)
print('Len of data',len(SZ_vData))
print('Len of labels',len(SZ_cobb_labels))

SZ_Labels=np.zeros([len(SZ_cobb_labels),1])
for i,j in enumerate(SZ_cobb_labels):#Label_cobb_sele
    if  max(j)>=COBB_TH:
        SZ_Labels[i]=1
    else:
        SZ_Labels[i]=0
print('num of control:',len(SZ_Labels[SZ_Labels==0]))
print('num of Patient:',len(SZ_Labels[SZ_Labels==1]))

''' 'DKCH'  '''
dk_vid_raw =load(r'D:\DKCH_gaitvideo\video_libs.npy',allow_pickle=True)
dk_cobb_label =load(r'D:\DKCH_gaitvideo\label_libs.npy',allow_pickle=True)
DK_vData, DK_cobb_labels = [], []
for idx,(i,j) in enumerate(zip(dk_vid_raw,dk_cobb_label)):
    # pdb.set_trace()
    i=np.reshape(i,[-1,224,224,3])[0:32,...]

    if np.shape(i)!=(32,224,224,3):#(np.shape(sz_tab_raw)[1],np.shape(sz_tab_raw)[2]):
        print(idx)
    else:
        DK_vData += [i]
        DK_cobb_labels += [j]
        
DK_vData = np.array(DK_vData)
DK_cobb_labels = np.array(DK_cobb_labels)
print('Len of data',len(DK_vData))
print('Len of labels',len(DK_cobb_labels))

DK_Labels=np.zeros([len(DK_cobb_labels),1])
for i,j in enumerate(DK_cobb_labels):#Label_cobb_sele
    if  max(j)>=COBB_TH:
        DK_Labels[i]=1
    else:
        DK_Labels[i]=0
print('num of control:',len(DK_Labels[DK_Labels==0]))
print('num of Patient:',len(DK_Labels[DK_Labels==1]))
video_lib = np.concatenate([DK_vData,SZ_vData])
label_lib = np.concatenate([DK_Labels,SZ_Labels])


kf = KFold(n_splits=5,random_state=1, shuffle=True)
for ki, (train_index, test_index) in enumerate(kf.split(video_lib,label_lib)):
    # pdb.set_trace()
    if ki==0 or ki==1 or ki==2 or ki==3:
        continue
    else:
        # print("TRAIN:", train_index, "TEST:", test_index)
        video_train = np.array([video_lib[i] for i in train_index]).astype('int')
        video_test = np.array([video_lib[j] for j in test_index]).astype('int')
        
        label_train = np.array([label_lib[i] for i in train_index]).astype('uint8')
        label_test = np.array([label_lib[j] for j in test_index] ).astype('uint8')
#%%
SEED = 26
EPOCHS = 50
BATCH_SIZE = 10
AUTO = tf.data.AUTOTUNE




# TRAINING


def run_experiment():
    # Initialize model
    # Resnet = STMBuilder(),Resnetr2plus1dBuilder(),Resnet3DBuilder()
    INPUT_SHAPE = (32,224,224,3)
    Resnet = Resnet3DBuilder()
    # INPUT_SHAPE = (16,224,224,3)
    model = Resnet.build_resnet_18(input_shape=INPUT_SHAPE, num_outputs=1)
    model.summary(line_length =120)


    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.lr
            decay = self.model.optimizer.decay
            iterations = self.model.optimizer.iterations
            lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
            print('learning_rate:',K.eval(lr_with_decay))
            
            
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
        loss = tf.keras.losses.BinaryCrossentropy(),#CategoricalCrossentropy(),#val_categorical_accuracy
        metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')]
    )

    # Train the model.
    checkpointer = ModelCheckpoint(filepath=r'C:\Users\Olive\Desktop\binary_class_model\TestModel',
                                   monitor='val_binary_accuracy',mode='max', verbose=1,save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5, min_lr=1e-15)
    history = model.fit(video_train,
                        label_train,
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS,
                        validation_split=0.2,
                        # validation_data=([x_val],y_val),
                        # validation_data = val_ds,
                        verbose=1,
                        callbacks=[checkpointer,reduce_lr,MyCallback()],
                        )
    plot_metrics(history)
    return model


model = run_experiment()

model = tf.keras.models.load_model(r'C:\Users\Olive\Desktop\binary_class_model\TestModel')# compile = False 
#rfft_data_test
baseline_results = model.evaluate([video_test], label_test,verbose=1,batch_size=BATCH_SIZE)
print(f"Test accuracy: {round(baseline_results[1] * 100, 2)}%")

for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

results_of_test = model.predict([video_test], batch_size=BATCH_SIZE)

import seaborn as sns   
from sklearn.metrics import confusion_matrix
def plot_cm(labels, predictions, draw=True):
    # pdb.set_trace()
    predictions[predictions>0.5]=1
    predictions[predictions<=0.5]=0
    cm = confusion_matrix(labels, predictions)
    if draw==True:
        plt.figure(figsize=(8,8))
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        
# plot_cm(np.argmax(y_test,axis=1),np.argmax(results_of_test,axis=1))
plot_cm(label_test,results_of_test)













