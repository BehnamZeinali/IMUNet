"""
This is the 1-D  version of MobileNet
Original paper is "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
Link: https://arxiv.org/abs/1704.04861

The implementation in https://towardsdatascience.com/building-mobilenet-from-scratch-using-tensorflow-ad009c5dd42c

"""

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

#import all necessary layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, BatchNormalization, DepthwiseConv1D
from tensorflow.keras.layers import ReLU, GlobalAveragePooling1D, Flatten, Dense
from tensorflow.keras import Model

# MobileNet block
def mb_block (x, filters, strides):
    
    x = DepthwiseConv1D(kernel_size = 3, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv1D(filters = filters, kernel_size = 1, strides = 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x
def MobileNet (input_shape , class_number):
    input = Input(shape = input_shape)
    x = Conv1D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    
    # main part of the model
    x = mb_block(x, filters = 64, strides = 1)
    x = mb_block(x, filters = 128, strides = 2)
    x = mb_block(x, filters = 128, strides = 1)
    x = mb_block(x, filters = 256, strides = 2)
    x = mb_block(x, filters = 256, strides = 1)
    x = mb_block(x, filters = 512, strides = 2)
    for _ in range (5):
         x = mb_block(x, filters = 512, strides = 1)
         
    x = mb_block(x, filters = 1024, strides = 2)
    x = mb_block(x, filters = 1024, strides = 1)
    x = GlobalAveragePooling1D ()(x)
    x = Flatten()(x)
    output = Dense (units = class_number)(x)
    model = Model(inputs=input, outputs=output)
    model.summary()
    
    return model
    # #plot the model
    # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True
    #                           , show_dtype=False,show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

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
    network  = MobileNet(input_shape , 2)
    network.summary()
    flops = get_flops(network, batch_size=1)
    print(flops)
    print(f"FLOPS: {flops / 10 ** 6:.03} M")


