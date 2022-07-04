"""
This is the 1-D  version of EfficientNetB0
Original paper is "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
Link: https://arxiv.org/abs/1905.11946

The implementation in https://github.com/AnjieCheng/MnasNet-PyTorch/blob/master/MnasNet.py has been modified.
A simple code has been added to calculate the number of FLOPs and parameters
from https://github.com/1adrianb/pytorch-estimate-flops.
"""

from tensorflow.keras import  backend
import keras
from keras import layers
from keras import models
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _conv_block(inputs, strides, filters, kernel=3):
    """
    Adds an initial convolution layer (with batch normalization and relu6).
    """
    x = layers.Conv1D(filters, kernel, padding='same', use_bias=False, strides=strides, name='Conv1')(inputs)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv1_bn')(x)
    
    print(x.name, inputs.shape, x.shape)
    
    # return layers.ReLU(6., name='Conv1_relu')(x)
    return layers.Activation(get_swish(), name='Conv1_activation')(x)
def get_swish():
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    def swish(x):
        """Swish activation function: x * sigmoid(x).
        Reference: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
        """

        if backend.backend() == 'tensorflow':
            try:
                # The native TF implementation has a more
                # memory-efficient gradient implementation
                return backend.tf.nn.swish(x)
            except AttributeError:
                pass

        return x * backend.sigmoid(x)

    return swish

    
def _inverted_res_block(inputs, kernel, expansion, alpha, filters, block_id, stride=1):
    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)
    
    if block_id:
        x = layers.Conv1D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_bn')(x)
        # x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
        x = layers.Activation(get_swish(), name= prefix + 'expand_activation')(x)
        
    else:
        prefix = 'expanded_conv_'

    x = layers.DepthwiseConv1D(kernel_size=kernel,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_bn')(x)

    # x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)
    x = layers.Activation(get_swish(), name= prefix + 'depthwise_activation')(x)

    x = layers.Conv1D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_bn')(x)

    print(x.name, inputs.shape, x.shape)
    
    if in_channels == pointwise_filters and stride == 1:
        print("Adding %s" % x.name)
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x
    
def EfficientNetB0(input_shape=None, alpha=1.0, depth_multiplier=1, pooling=None, nb_classes=10):
    img_input = layers.Input(shape=input_shape)
    
    x = _conv_block(img_input, strides=2, filters=32)


    x = _inverted_res_block(x, kernel=3, expansion=1, stride=1, alpha=alpha, filters=16, block_id=1)
    
    
    x = _inverted_res_block(x, kernel=3, expansion=6, stride=2, alpha=alpha, filters=24, block_id=2)
    x = _inverted_res_block(x, kernel=3, expansion=6, stride=2, alpha=alpha, filters=24, block_id=3)
    
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=40, block_id=4)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=40, block_id=5)
    
    
    x = _inverted_res_block(x, kernel=3, expansion=6, stride=2, alpha=alpha, filters=80, block_id=6)
    x = _inverted_res_block(x, kernel=3, expansion=6, stride=2, alpha=alpha, filters=80, block_id=7)
    x = _inverted_res_block(x, kernel=3, expansion=6, stride=2, alpha=alpha, filters=80, block_id=8)

    x = _inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=112, block_id=9)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=112, block_id=10)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=112, block_id=11)
    
   
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=192, block_id=12)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=192, block_id=13)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=192, block_id=14)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=192, block_id=15)
    
    x = _inverted_res_block(x, kernel=3, expansion=6, stride=1, alpha=alpha, filters=320, block_id=16)
    
    
    x = layers.Conv1D(1280, 1,
                      padding='same',
                      use_bias=False,
                      name='top_conv')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='top_bn')(x)
    # x = layers.ReLU(6., name='top_relu')(x)
    x = layers.Activation(get_swish(), name='top_activation')(x)
    
    if pooling == 'avg':
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = layers.GlobalMaxPooling1D()(x)
        
    x = layers.Dense(nb_classes, use_bias=True, name='proba')(x)
    inputs = img_input
    
    model = models.Model(inputs, x, name='EfficientNetB0')
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
    network  =  EfficientNetB0(input_shape=( 6, 200), pooling='avg', nb_classes= 2)
    network.summary()
    flops = get_flops(network, batch_size=1)
    print(flops)
    print(f"FLOPS: {flops / 10 ** 6:.03} M")