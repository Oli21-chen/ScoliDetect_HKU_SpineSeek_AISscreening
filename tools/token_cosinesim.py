'''
To achive this, you should have STTokenFormer baseline model
and model without TokenNorm. To compare the ST tokens similarity
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from utili_dataload_upsample import data_processing
from scipy.spatial import distance

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)
        
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        if self.learning_rate_base < self.warmup_learning_rate:
            raise ValueError("Learning_rate_base must be larger or equal to warmup_learning_rate.")

    def __call__(self, step):
        cos_annealed_lr = tf.cos(
            self.pi*(tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where( step < self.warmup_steps, warmup_rate, learning_rate )

        learning_rate = tf.where(step > self.total_steps, 0.0, learning_rate, name="learning_rate")
        return learning_rate
    def get_config(self):
       return {
           'learning_rate_base': self.learning_rate_base,
           'total_steps': self.total_steps,
           'warmup_learning_rate':self.warmup_learning_rate,
           'warmup_steps':self.warmup_steps,
       }    
   
train_lib, test_lib, val_lib = data_processing(
    COBB_TH = 11,
    IMG_SIZE = 128,
    )
                                    
[table_train, video_train, screen_train] = train_lib
[table_test, video_test, screen_test] = test_lib
[table_val, video_val, screen_val] = val_lib

model_raw = tf.keras.models.load_model(
    r'C:\Users\Olive\Desktop\binary_class_model\20241217-02',
    custom_objects={'WarmUpCosine': WarmUpCosine})# compile = False 

model_raw.summary()
sub_model1 = tf.keras.Model(inputs=model_raw.input, outputs=model_raw.layers[2].output)
# Display the summary of the sub-model
sub_model1.summary()

sub_output1 = sub_model1.predict(table_train)
tem_l_patches1 = sub_output1[:,:30,:] #token2
fea_n_patches1 = sub_output1[:,30:,:] #token1


model_sec = tf.keras.models.load_model(
    r'C:\Users\Olive\Desktop\binary_class_model\20241217-01',
    custom_objects={'WarmUpCosine': WarmUpCosine})# compile = False 
model_sec.summary()
sub_model2 = tf.keras.Model(inputs=model_sec.input, outputs=model_sec.layers[2].output)
# Display the summary of the sub-model
sub_model2.summary()

sub_output2 = sub_model2.predict(table_train)
tem_l_patches2 = sub_output2[:,:30,:] #token2
fea_n_patches2 = sub_output2[:,30:,:] #token1

val1,val2 = [],[]
for i in range(len(table_train)):
    tokens11 = fea_n_patches1[i]
    tokens12 = tem_l_patches1[i]
    similarity_matrix1 = distance.cdist(tokens11, tokens12, 'cosine')

    mean_val1 = np.mean(similarity_matrix1)
    val1 += [mean_val1]
    
    tokens21 = fea_n_patches2[i]
    tokens22 = tem_l_patches2[i]
    similarity_matrix2 = distance.cdist(tokens21, tokens22, 'cosine')
    mean_val2 = np.mean(similarity_matrix2)
    val2 += [mean_val2]

plt.figure(figsize=(8, 6))
plt.scatter(val1, val2, alpha=0.7)
plt.plot([min(val1 + val2), max(val1 + val2)], 
         [min(val1 + val2), max(val1 + val2)], 
         'r--', label='y = x')  # Reference line
plt.title("Pairwise Comparison of Token Similarity",fontsize=16)
plt.xlabel("Baseline Model",fontsize=14)
plt.ylabel("Model w/o STT",fontsize=14)
plt.legend(fontsize=14)
plt.show()