"""
This is the 1-D  version of MobileNetV2
Original paper is "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
Link: https://arxiv.org/abs/1801.04381

The implementation in https://medium.com/analytics-vidhya/creating-mobilenetsv2-with-tensorflow-from-scratch-c85eb8605342

"""

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, BatchNormalization, DepthwiseConv1D, Dropout
from tensorflow.keras.layers import ReLU, GlobalAveragePooling1D, Flatten, Dense, add
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


def expansion_block(x,t,filters,block_id):    
    prefix = 'block_{}_'.format(block_id)
    total_filters = t*filters
    x = Conv1D(total_filters,1,padding='same',use_bias=False, name =    prefix +'expand')(x)    
    x = BatchNormalization(name=prefix +'expand_bn')(x)
    x = ReLU(6,name = prefix +'expand_relu')(x)
    return x

def depthwise_block(x,stride,block_id):    
    prefix = 'block_{}_'.format(block_id)
    x = DepthwiseConv1D(3,strides=stride,padding ='same', use_bias = False, name = prefix + 'depthwise_conv')(x)    
    x = BatchNormalization(name=prefix +'dw_bn')(x)
    x = ReLU(6,name = prefix +'dw_relu')(x)
    return x

def projection_block(x,out_channels,block_id):    
    prefix = 'block_{}_'.format(block_id)
    x = Conv1D(filters=out_channels,kernel_size = 1,   padding='same',use_bias=False,name= prefix + 'compress')(x)    
    x = BatchNormalization(name=prefix +'compress_bn')(x)
    return x


def Bottleneck(x,t,filters, out_channels,stride,block_id):
    y = expansion_block(x,t,filters,block_id)
    y = depthwise_block(y,stride,block_id)
    y = projection_block(y, out_channels,block_id)
    if y.shape[-1]==x.shape[-1]:
       y = add([x,y])
    return y
def MobileNetV2(input_shape=(224,224,3),n_classes = 1000): 
   input = Input (input_shape)
   x = Conv1D(32,3,strides=2,padding='same', use_bias=False)(input)
   x = BatchNormalization(name='conv1_bn')(x)
   x = ReLU(6, name='conv1_relu')(x)    # 17 Bottlenecks
   
   x = depthwise_block(x,stride=1,block_id=1)
   x = projection_block(x, out_channels=16,block_id=1)
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 2,block_id = 2)
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 1,block_id = 3)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 2,block_id = 4)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 5)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 6)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 2,block_id = 7)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 8)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 9)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 10)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 11)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 12)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 13)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 2,block_id = 14)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 15)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 16)    
   x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 320, stride = 1,block_id = 17)
   
   x = Conv1D(filters = 1280,kernel_size = 1,padding='same',use_bias=False, name = 'last_conv')(x)
   x = BatchNormalization(name='last_bn')(x)
   x = ReLU(6,name='last_relu')(x)  
   x = Dropout(0.4)(x)
   x = GlobalAveragePooling1D(name='global_average_pool')(x)
   output = Dense(n_classes)(x)    
   model = Model(input, output)
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
    network  = MobileNetV2(input_shape , 2)
    network.summary()
    flops = get_flops(network, batch_size=1)
    print(flops)
    print(f"FLOPS: {flops / 10 ** 6:.03} M")   
