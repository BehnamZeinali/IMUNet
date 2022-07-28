
"""
Created on Wed Jun 22 14:59:00 2022

@author: behnam
Keras implementation of https://github.com/Sachini/ronin/blob/master/source/model_resnet1d.py
"""

from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout,ReLU, Conv1D,  Input, MaxPool1D, Flatten, GlobalAveragePooling1D,  BatchNormalization, Layer, Add,DepthwiseConv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from keras import models

def ResnetBlock(x , channels , down_sample):
    """
    A standard resnet block.
    """
    res = x


    
    strides = [2, 1] if down_sample else [1, 1]

    KERNEL_SIZE = (3)
    # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
    INIT_SCHEME = "he_normal"

    x = Conv1D(channels, strides=strides[0],kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(channels, strides=strides[1],kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)(x)
    x = BatchNormalization()(x)
    

    if down_sample:
        # perform down sampling using stride of 2, according to [1].
        res = Conv1D( channels, strides=2, kernel_size=(1), kernel_initializer=INIT_SCHEME, padding="same")(res)
        res = BatchNormalization()(res)
    x = Add()([x, res])
    x = ReLU()(x)

    return x

def ResNet18(input_shape=None, n_classes=2):
    
    imu_input = Input(shape=input_shape)
    input = Input(shape = input_shape)
    x = Conv1D(filters = 64, kernel_size = 7, strides = 2, padding = 'same', kernel_initializer="he_normal")(input)
    x = BatchNormalization()(x)
    x =  MaxPool1D(pool_size=(3), strides=2, padding="same")(x)
    
    x = ResnetBlock(x , 64 , False)
    x = ResnetBlock(x , 64 , False)
    x = ResnetBlock( x , 128, down_sample=True)
    x = ResnetBlock(x , 128 , False)
    x = ResnetBlock(x , 256, down_sample=True)
    x = ResnetBlock(x ,256 , False)
    x = ResnetBlock(x ,512, down_sample=True)
    x = ResnetBlock(x ,512 , False)
    
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
    input_shape =   (6, 200)
    network  =  ResNet18(input_shape=( 6, 200),  n_classes= 2)
    network.summary()
    flops = get_flops(network, batch_size=1)
    print(flops)
    print(f"FLOPS: {flops / 10 ** 6:.03} M")
