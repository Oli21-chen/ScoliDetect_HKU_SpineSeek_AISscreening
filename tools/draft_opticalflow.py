import tensorflow as tf
import cv2
from numpy import load
import matplotlib.pyplot as plt

sz_vid_raw =load(r'D:\HKUSZH_gaitvideo\video_libs.npy',allow_pickle=True)#\log
sz_vid_raw = sz_vid_raw[:5,...]

class OpticalFlowLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OpticalFlowLayer, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     super(OpticalFlowLayer, self).build(input_shape)

    def call(self, inputs):
        # Extract previous and current frames from input tensor
        batch_size, time_steps, height, width, channels = inputs.shape
        prev_frames = inputs[:, :-1, :, :, :]
        curr_frames = inputs[:, 1:, :, :, :]
        # Convert frames to grayscale
        prev_frames_gray = tf.image.rgb_to_grayscale(prev_frames)
        print("here2:",prev_frames_gray.shape) 
        curr_frames_gray = tf.image.rgb_to_grayscale(curr_frames)
        
        def compute_flow(prev_frame, curr_frame):
            print("here1:",prev_frame.shape) 
            flow = cv2.calcOpticalFlowFarneback(
                prev = prev_frame,
                next = curr_frame,
                flow = None,
                pyr_scale = 0.5,
                levels = 3,
                winsize = 3,
                iterations = 3,
                poly_n = 5,
                poly_sigma = 1.2,
                flags = 0
            )
            return flow
      
        op_flow_timestep = tf.TensorArray(tf.float32, size=1,dynamic_size=True,
                                          element_shape = [time_steps-1, height, width, 2])
        for t in range(0,time_steps-1):
            print('here1.1',prev_frames_gray[:,t,...])
            flow = tf.numpy_function(func=compute_flow, inp=[prev_frames_gray[:,t,...], curr_frames_gray[:,t,...]], Tout=tf.float32)
            op_flow_timestep.write(t,flow)
        op_flow_timestep = op_flow_timestep.stack()
        print('here3.1:',op_flow_timestep)
        # op_flow_timestep = tf.squeeze(op_flow_timestep,0)
        # print('here3.2:',op_flow_timestep)
        # op_flow = tf.reshape(op_flow_timestep, [-1,time_steps-1,height, width, 2])
        # print("op_flow shape2:",op_flow.shape)  # Print the shape of flow
        return op_flow_timestep

model = tf.keras.models.Sequential([
    OpticalFlowLayer(),
    # tf.keras.layers.Lambda(lambda x: tf.image.convert_image_dtype(x[0], tf.uint8)),
    # tf.keras.layers.Lambda(lambda x: tf.image.encode_png(x)),
])
# Call the model to compute the output and print the result
output = model(sz_vid_raw)
a = output.numpy()
a1 = a[0]
for i in range(len(a1)):
    plt.imshow(a1[i,...,1])
    plt.show()
# image = tf.io.decode_png(output[0])
# plt.imshow(image)
# plt.show()