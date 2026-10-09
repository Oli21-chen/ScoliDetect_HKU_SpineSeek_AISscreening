# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:30:08 2023
Summary:
    Looks like Resnet3d can also perform as good as transformer.
    But Res_results has higher False Negative. And hard to learn knowledge with th=11
    
    
# from model_resnet3d import Resnet3DBuilder
# from utili_model_videor2plus1d import Resnetr2plus1dBuilder
# from model_STM import STMBuilder
# from utili_multivit_opticalflow_video import create_model
    - utili_multimodal_timecrossvit: optical flow + video
# from utili_model_videovit import create_vivit_classifier,TubeletEmbedding,PositionalEncoder
# from utili_model_tablevit import create_feature_encoder
# from utili_model_tableItransformer import create_feature_encoder
# from utili_model_maest import spatialtemporal_leaner,PositionalEncoder
   
------------------------------------------------------------------------
# model.layer[3]
# attention_layer = model.get_layer('MHA_3')

def get_attention_scores(model, input_data):
    # Create a new model that outputs both predictions and attention scores
    inputs = model.input
    outputs, attention_scores = model.layers[-1](inputs)
    attention_model = tf.keras.Model(inputs=inputs, outputs=[outputs, attention_scores])
    
    # Predict using this new model
    predictions, att_scores = attention_model.predict(input_data)
    return predictions, att_scores

embed_coors, #34
dis_embedders, # 106
ang_embedders, # 32
gait_phases, #66

---------------------  Set Multi-gpu In tensorflow  ---------------------
import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # list your GPUs

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.MultiWorkerMirroredStrategy()
print('Number of devices:', strategy.num_replicas_in_sync)
    

@author: Olive
"""
import sys
import os
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lr_func import WarmUpCosine,ExponentialDecay
from utils.utili_plots import plot_metrics, plot_cm,roc_auc,plt_kinetic_sig
from conv_models.model_resnet3d import Resnet3DBuilder
from models.utili_model_videor2plus1d import Resnetr2plus1dBuilder
from conv_models.model_STM import STMBuilder
from models.utili_model_videovit import video_vit
from models.utili_model_videovit_demographic import vivit
from models.utili_model_tablevit import table_vit #attscore
from models.utili_model_tableItransformer import create_feature_encoder
from models.utili_model_TimeSFormerV2 import TimeSFormer_model
from utils.utili_dataload import data_processing
# from utili_dataload_upsample import data_processing


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pdb
from sklearn.utils.class_weight import compute_class_weight


# SEED = 128
# tf.random.set_seed(SEED)
# np.random.seed(SEED)
train_lib, test_lib, val_lib = data_processing(
    COBB_TH = 11,
    IMG_SIZE = 128,
    )
                                    

[table_train, video_train, screen_train] = train_lib
[table_test, video_test, screen_test] = test_lib
[table_val, video_val, screen_val] = val_lib

plt_kinetic_sig(table_train,'table_train')
plt_kinetic_sig(table_test, 'table_test')
plt_kinetic_sig(table_val, 'table_val')

training_data = [video_train]
testing_data = [video_test]
val_data = (video_val,screen_val)
screen_train = np.squeeze(screen_train)
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(screen_train), 
                                     y=screen_train)
class_weight_dict = dict(zip(np.unique(screen_train), class_weights))
EPOCHS = 170
BATCH_SIZE = 32


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lr_history = []

    def on_epoch_end(self, epoch, logs=None):
        # lr = self.model.optimizer.lr.read_value().numpy()
        lr = self.model.optimizer.learning_rate(
            self.model.optimizer.iterations).numpy()
        self.lr_history.append(lr)
        # print(f'\n current LR is = {lr} \n')
        
def run_experiment():
    ''' table_vit  '''
    # model = table_vit( #7e-6
    #     inputshape = (32,238),#238 172
    #     mlp_dim = 2048,
    #     att_dim = 6*128,
    #     att_drop = 0.,#small
    #     head = 6,
    #     transform_layers = 5,
    #     drop_out = 0, # 0 is better
    #     if_cls = True,
    #     inverted_emb = None,
    #     init_values = 0.12, #Linear model estimation 0.12
    #     depth = 0.5
    #     )

    '''  vi-vit ''' 
    model = video_vit( # 4e-6
        inputshape = (32,128,128,3),
        vol_size = (2,32,32),
        mlp_dim = 768,
        att_dim = 6*256,
        num_heads = 6,
        drop_out = 0,
        transform_layers = 4,
        if_cls = True,
        vit = True ,
    )
    
    model.summary(line_length = 80)
    tf.keras.utils.plot_model(model,
                              to_file=r'C:\Users\Olive\Desktop\model_fig.png', 
                              show_shapes=True,
                              show_layer_names=True, 
                              expand_nested=True)
    total_steps = int((len(table_train) // BATCH_SIZE) * EPOCHS)
    lr = 4e-6
    scheduled_lrs = WarmUpCosine(
        learning_rate_base = lr, # 9.5e-6,
        total_steps = total_steps,
        warmup_learning_rate = 4e-7,
        warmup_steps = int(total_steps * 0.1),
    )
    lr_logger = LearningRateLogger()
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    # monitor='val_loss', factor = 0.8,
    # patience= 5, min_lr=1e-8,
    # verbose=1
    # )
    
    lrs = [scheduled_lrs(step).numpy() for step in range(total_steps)]
    plt.plot(lrs)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("LR", fontsize=14)
    plt.show()
   
    model.compile(
        optimizer = tf.keras.optimizers.Adam(scheduled_lrs),
        loss =  tf.keras.losses.BinaryCrossentropy(),
        metrics= tf.keras.metrics.BinaryAccuracy(name='screen_accuracy'),
        )

    # Train the model.
    checkpointer = ModelCheckpoint(
        filepath = r'C:\Users\Olive\Desktop\binary_class_model\20250224t_8layer',
        monitor = 'val_screen_accuracy',
        mode = 'max', 
        verbose = 1,
        save_best_only = True)
    
    history = model.fit(
        x = training_data,
        y = screen_train,
        batch_size = BATCH_SIZE, 
        epochs = EPOCHS,
        validation_data = val_data,
        class_weight = class_weight_dict,
        verbose=1,
        callbacks=[
            checkpointer,
            lr_logger,
            # reduce_lr
            ],
    )
    plot_metrics(history)
    print(f'batach size:{BATCH_SIZE}, lr:{lr}')
    learning_rates = lr_logger.lr_history
    plt.plot(learning_rates)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("LR", fontsize=14)
    plt.show()
    return model,history


model,history = run_experiment()

''' Model test  '''
model_pre = model.predict(testing_data)#demographic_test
model_pre_screen = model_pre#[0]

''' Best performance on Validation '''
test_model = tf.keras.models.load_model(
    r'C:\Users\Olive\Desktop\binary_class_model\20250224t_8layer',
    custom_objects={'WarmUpCosine': WarmUpCosine})# compile = False 

test_model_pre = test_model.predict(testing_data)
test_model_pre_screen = test_model_pre#[0]

''' plot results '''
pred_th = 0.5
model_pre_screen = np.where(model_pre_screen<pred_th,0,1)
test_model_pre_screen = np.where(test_model_pre_screen<pred_th,0,1)

acc_model_screen= np.count_nonzero((model_pre_screen==screen_test).astype(int))/len(screen_test)
acc_test_model_screen= np.count_nonzero((test_model_pre_screen==screen_test).astype(int))/len(screen_test)
if acc_model_screen > acc_test_model_screen:
    print('The training model is selected:{}'.format(acc_model_screen))
    plot_cm(screen_test,
            model_pre,
            th = pred_th,
            screen = True,
            severity = False)
    auc = roc_auc(screen_test,model_pre)
    print('auc: ',auc)
    print(classification_report(screen_test, model_pre_screen, target_names=['Control','AIS']))
    model.save(r'C:\Users\Olive\Desktop\binary_class_model\20250224t_8layer')
else:
    print('The checkpoint model is selected:{}'.format(acc_test_model_screen))
    plot_cm(screen_test,
            test_model_pre,
            th = pred_th,
            screen = True,
            severity = False)
    auc = roc_auc(screen_test,test_model_pre)
    print('auc: ',auc)
    print(classification_report(screen_test, test_model_pre_screen, target_names=['Control','AIS']))
    


# acc_model_severity = np.count_nonzero((model_pre_severity==severity_test).astype(int))/len(severity_test)
# acc_test_model_severity= np.count_nonzero((test_model_pre_severity==severity_test).astype(int))/len(severity_test)
# if acc_model_severity>acc_test_model_severity:
#     print('The training model is selected:{}'.format(acc_model_severity))
#     plot_cm(severity_test,
#             model_pre_severity,
#             th=0.5,
#             screen = False,
#             severity=True)
# else:
#     print('The checkpoint model is selected:{}'.format(acc_test_model_severity))
#     plot_cm(severity_test,
#             test_model_pre_severity,
#             th=0.5,
#             screen = False,
#             severity=True)




