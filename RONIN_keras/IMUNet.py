
"""

IMUnet implementation for keras
Created on Wed Jun 22 15:31:04 2022

@author: behnam
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:59:00 2022

@author: behnam
"""

from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import ELU, Dense, SeparableConv1D, Dropout, Conv1D,  Input, MaxPool1D, Flatten, GlobalAveragePooling1D,  BatchNormalization, Layer, Add,DepthwiseConv1D
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from keras import models

def MobileResnetBlock(x , channels , down_sample):
    """
    A standard resnet block.
    """
    res = x
    strides = [2, 1] if down_sample else [1, 1]
    KERNEL_SIZE = (3)
    # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
    INIT_SCHEME = "he_normal"

    x = DepthwiseConv1D(strides=strides[0],kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = SeparableConv1D(channels, strides=strides[1],kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)(x)
    x = BatchNormalization()(x)
    

    if down_sample:
        # perform down sampling using stride of 2, according to [1].
        res = SeparableConv1D( channels, strides=2, kernel_size=(1), kernel_initializer=INIT_SCHEME, padding="same")(res)
        res = BatchNormalization()(res)
    x = Add()([x, res])
    x = ELU()(x)
    return x

def IMUNet(input_shape=None, n_classes=2):
    
    
    input = Input(shape = input_shape)
    x = Conv1D(filters = 64, kernel_size = 7, strides = 2, padding = 'same', kernel_initializer="he_normal")(input)
    x = BatchNormalization()(x)
    x =  MaxPool1D(pool_size=(3), strides=2, padding="same")(x)
    
    x = MobileResnetBlock(x , 64 , False)
    x = MobileResnetBlock(x , 64 , False)
    x = MobileResnetBlock( x , 128, down_sample=True)
    x = MobileResnetBlock(x , 128 , False)
    x = MobileResnetBlock(x , 256, down_sample=True)
    x = MobileResnetBlock(x ,256 , False)
    x = MobileResnetBlock(x ,512, down_sample=True)
    x = MobileResnetBlock(x ,512 , False)
    
    x = Conv1D(128, (1), strides=1)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    
    
    x =  Dropout(0.5)(x)
    x = Dense(512)(x)
    
    x =  Dropout(0.5)(x)
    output = Dense(n_classes)(x)
    model = Model(inputs=input, outputs=output)
    # model = models.Model(imu_input, x, name='ResNet')
    return model

def get_flops(model, batch_size=None):
    if batch_size is None:
        batch_size = 1

    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops
if __name__ == '__main__':
    input_shape =   (200, 6)
    network  =  IMUNet(input_shape=( 200, 6),  n_classes= 2)
    network.summary()
    flops = get_flops(network, batch_size=1)
    print(flops)
    print(f"FLOPS: {flops / 10 ** 6:.03} M")
