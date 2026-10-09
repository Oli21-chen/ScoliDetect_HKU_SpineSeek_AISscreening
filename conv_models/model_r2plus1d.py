# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:02:24 2023

@author: Olive
"""

"""A vanilla 3D resnet implementation.

Based on Raghavendra Kotikalapudi's 2D implementation
keras-resnet (See https://github.com/raghakot/keras-resnet.)
"""
import pdb
import tensorflow as tf
import six
from math import ceil
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from tensorflow.keras.layers import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization,ZeroPadding3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def conv1x3x3(filters, kernel_size,strides=1):
    return Conv3D(filters,
                kernel_size = kernel_size,
                strides = strides,
                padding='same',
                use_bias=False)



def conv3x1x1(filters, kernel_size,strides=1):
    return Conv3D(filters,
                kernel_size= kernel_size,
                strides = strides,
                padding='same',
                use_bias=False)


def conv1x1x1(filters, strides=1):
    return Conv3D(filters,
                kernel_size=1,
                strides = strides,
                use_bias=False)


def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu2plus1d(filters1,filters2,kernel_size1,kernel_size2,stride1,stride2):

    def f(input):
        x = conv1x3x3(filters1,kernel_size1,stride1)(input)
        x = _bn_relu(x)
        x = conv3x1x1(filters2,kernel_size1,stride2)(x)
        x = BatchNormalization(axis=CHANNEL_AXIS)(x)
        return x

    return f


def _bn_relu_conv2plus1d(filters1,filters2,kernel_size1,kernel_size2,strides1,strides2):
    """Helper to build a  BN -> relu -> convr2plus1d block."""
    def f(input):
        x = _bn_relu(input)
        x = conv1x3x3(filters1,kernel_size1,strides1)(x)
        x = _bn_relu(x)
        x = conv3x1x1(filters2,kernel_size2,strides2)(x)
        return x
    return f


def _shortcutr2plus1d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = ceil(input.shape[DIM1_AXIS] \
        / residual.shape[DIM1_AXIS])
    stride_dim2 = ceil(input.shape[DIM2_AXIS] \
        / residual.shape[DIM2_AXIS])
    stride_dim3 = ceil(input.shape[DIM3_AXIS] \
        / residual.shape[DIM3_AXIS])
    equal_channels = residual.shape[CHANNEL_AXIS] \
        == input.shape[CHANNEL_AXIS]

    # print(stride_dim1, stride_dim2, stride_dim3)
    shortcut = input
    
    # pdb.set_trace()
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = Conv3D(
            filters=residual.shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal", padding="valid",
            kernel_regularizer=l2(1e-4)
            )(input)
        shortcut = BatchNormalization(axis=CHANNEL_AXIS)(shortcut)
        
    return add([shortcut, residual])




def basic_block(filters1,filters2, strides1=(1, 1, 1),strides2=(1, 1, 1), 
                kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:

            conv1 = conv1x3x3(filters1,(1, 3, 3), strides=1)(input)#stride=1
            conv1 = _bn_relu(conv1)
            conv1 = conv3x1x1(filters2,(3, 1, 1), strides=1)(conv1)
        else:
            # pdb.set_trace()
            conv1 = _bn_relu_conv2plus1d(
                filters1 = filters1,
                filters2 = filters2,
                kernel_size1 = (1, 3, 3),
                kernel_size2 = (3, 1, 1),
                strides1 = strides1,
                strides2 = strides2
                # kernel_regularizer=kernel_regularizer
                )(input)

        residual = _bn_relu_conv2plus1d(
            filters1 = filters1,
            filters2 = filters2,
            kernel_size1 = (1, 3, 3),
            kernel_size2 = (3,1,1),
            strides1 = 1,
            strides2 = 1
            # kernel_regularizer=kernel_regularizer
            )(conv1)
        return _shortcutr2plus1d(input, residual)

    return f

def _residual_block_r2plus1d(block_function, 
                             filters1,filters2,
                             strides1,strides2,
                             kernel_regularizer, repetitions,
                      is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides1 = (1, 1, 1)
            strides2 = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides1 = (1, 2, 2)
                strides2 = (2, 1, 1)
            # pdb.set_trace()
            input = block_function(filters1=filters1,filters2=filters2, 
                                   strides1=strides1,strides2=strides2,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0)
                                   )(input)
        return input

    return f

# def bottleneck(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
#                is_first_block_of_first_layer=False):
#     """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
#     def f(input):
#         if is_first_block_of_first_layer:
#             # don't repeat bn->relu since we just did bn->relu->maxpool
#             conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
#                               strides=strides, padding="same",
#                               kernel_initializer="he_normal",
#                               kernel_regularizer=kernel_regularizer
#                               )(input)
#         else:
#             conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
#                                        strides=strides,
#                                        kernel_regularizer=kernel_regularizer
#                                        )(input)

#         conv_3_3 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
#                                    kernel_regularizer=kernel_regularizer
#                                    )(conv_1_1)
#         residual = _bn_relu_conv3d(filters=filters * 4, kernel_size=(1, 1, 1),
#                                    kernel_regularizer=kernel_regularizer
#                                    )(conv_3_3)

#         return _shortcut3d(input, residual)

#     return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class Resnetr2plus1dBuilder(object):
    """ResNet3D."""

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, reg_factor):
        """Instantiate a vanilla ResNet3D keras model.

        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        # block_fn = _get_block(block_fn)
        input = Input(shape=input_shape)
        # first conv
        conv1 = _conv_bn_relu2plus1d(filters1=45,
                                     filters2=64,
                                     kernel_size1 = (1, 7, 7),
                                     kernel_size2 = (3, 1, 1),
                                     stride1=(1,2,2),
                                     stride2=(1,1,1)
                                     )(input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                              padding="same")(conv1)
        # repeat blocks
        block = pool1
        filters = 64
        # pdb.set_trace()
        for i, r in enumerate(repetitions):
            block = _residual_block_r2plus1d(block_fn,
                                             filters1=filters,
                                             filters2=filters,
                                              strides1= 1,
                                              strides2= 1,
                                       kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            filters *= 2

        # # last activation
        block_output = _bn_relu(block)

        # average poll and classification
        pool2 = AveragePooling3D(pool_size=(block.shape[DIM1_AXIS],
                                            block.shape[DIM2_AXIS],
                                            block.shape[DIM3_AXIS]),
                                  strides=(1, 1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        if num_outputs > 1:
            dense = Dense(units=num_outputs,
                          kernel_initializer="he_normal",
                          activation="softmax",
                          kernel_regularizer=l2(reg_factor))(flatten1)
        else:
            dense = Dense(units=num_outputs,
                          kernel_initializer="he_normal",
                          activation="sigmoid",
                          kernel_regularizer=l2(reg_factor))(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 18."""
        return Resnetr2plus1dBuilder.build(input_shape, num_outputs, basic_block,
                                     [2,2,2,2], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 34."""
        return Resnetr2plus1dBuilder.build(input_shape, num_outputs, basic_block,
                                     [3, 4, 6, 3], reg_factor=reg_factor)

    # @staticmethod
    # def build_resnet_50(input_shape, num_outputs, reg_factor=1e-4):
    #     """Build resnet 50."""
    #     return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
    #                                  [3, 4, 6, 3], reg_factor=reg_factor)

    # @staticmethod
    # def build_resnet_101(input_shape, num_outputs, reg_factor=1e-4):
    #     """Build resnet 101."""
    #     return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
    #                                  [3, 4, 23, 3], reg_factor=reg_factor)

    # @staticmethod
    # def build_resnet_152(input_shape, num_outputs, reg_factor=1e-4):
    #     """Build resnet 152."""
    #     return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
    #                                  [3, 8, 36, 3], reg_factor=reg_factor)
    
# Resnet = Resnetr2plus1dBuilder()
# INPUT_SHAPE = (32,224,224,3)
# model = Resnet.build_resnet_18(input_shape=INPUT_SHAPE, num_outputs=1)
# model.summary()
# tf.keras.utils.plot_model(
#     model,
#     to_file=r'C:\Users\Olive\Desktop\r2plus1d.png')