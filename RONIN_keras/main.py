"""
This is the keras implementation of  https://github.com/Sachini/ronin/blob/master/source/ronin_resnet.py
More datasets and machine learning models have been added to the code

Herath, S., Yan, H. and Furukawa, Y., 2020, May. RoNIN: Robust Neural Inertial Navigation in the Wild:
Benchmark, Evaluations, & New Methods. In 2020 IEEE International Conference on Robotics and Automation
(ICRA) (pp. 3146-3152). IEEE.


"""

import os
import time
from os import path as osp

import numpy as np
import json

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from tensorboardX import SummaryWriter
from utils import *
# from data_glob_speed import *
# from transformations import *
from metric import compute_ate_rte, compute_absolute_trajectory_error
import tensorflow as tf
from math import pi, cos


import keras.backend as K
from model_resnet1d import ResNet18
from MobileNetV2 import MobileNetV2_1D_Arch
from MobileNet import MobileNetV1_1D
from MnasNet import MnasNet

from EfficientnetB0 import EfficientNetB0
from  IMUNet import IMUNet


def get_model(arch):
    arch = args.arch
    n_class = 2
    if (args.dataset == 'px4'):
        n_class = 3
    input_shape =   (6, 200)
    if arch == 'ResNet':
        
        network = ResNet18(n_class)   
    elif arch == 'MobileNetV2':
        network = MobileNetV2_1D_Arch(input_shape , n_class)
    elif arch == 'MobileNet':
        input_shape =   ( 6, 200)
        network  = MobileNetV1_1D(input_shape , n_class)
    elif arch == 'MnasNet':
        input_shape =   ( 6, 200)
        network  = MnasNet(input_shape=( 6, 200), pooling='avg', nb_classes= n_class)
    elif arch == 'EfficientNet':
        network  =EfficientNetB0(input_shape=( 6, 200), pooling='avg', nb_classes= n_class)
    elif arch == 'IMUNet':
       network  =IMUNet(n_class)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network
 

def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)
    
    
def get_dataset(root_dir, data_list, args, **kwargs):
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True

    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    elif args.dataset == 'ridi':
        seq_type = RIDIGlobSpeedSequence
    elif args.dataset == 'px4':
        seq_type = PX4Sequence
    elif args.dataset == 'oxiod':
        seq_type = OXIODSequence
    elif args.dataset == 'proposed':
        seq_type = ProposedSequence
    dataset = CustomDataGen(
        seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size, args.batch_size , 
        random_shift=random_shift, transform=transforms,
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    mode = kwargs.get('mode', 'train')
    if args.dataset == 'px4':
        if (mode == 'train'):
            root_dir = root_dir + '/training'
            data_list = os.listdir(root_dir)
        else:
            root_dir = root_dir + '/validation'
            data_list = os.listdir(root_dir)
    elif args.dataset == 'oxiod':
        if (mode == 'train'):
            root_dir = root_dir + '/train'
            data_list = os.listdir(root_dir)
        else:
            root_dir = root_dir + '/validation'
            data_list = os.listdir(root_dir)
    else:
        with open(list_path) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)

class LR_Restart(tf.keras.callbacks.Callback):
    '''SGDR learning rate scheduler'''
    def __init__(self, lr_max, lr_min, restart_epoch  ,verbose):
        super(LR_Restart, self).__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.restart_epoch = restart_epoch
        self.cycle = 0
        self.verbose = verbose
        self.lr = lr_max

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch % self.restart_epoch) == 0:
            self.cycle = epoch / self.restart_epoch
        print (self.lr)   
        self.curr_epoch = epoch - (self.cycle * self.restart_epoch)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        curr_epoch = self.curr_epoch + (batch/self.params['steps'])
        self.lr = self.lr_min + (0.5 * (self.lr_max - self.lr_min) * (1 + cos((curr_epoch / self.restart_epoch) * pi)))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


# def get_flops():
#     run_meta = tf.compat.v1.RunMetadata()
#     opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

#     # We use the Keras session graph in the call to the profiler.
#     flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
#                                 run_meta=run_meta, cmd='op', options=opts)

#     return flops.total_float_ops  # Prints the "flops" of the model.

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

def train(args, **kwargs):
    
    # start_t = time.time()
    print(args.root_dir)
    train_generator = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    val_generator = get_dataset_from_list(args.root_dir, args.val_list, args, mode='val')
    
    # def get_flops():
    #     session = tf.compat.v1.Session()
    #     graph = tf.compat.v1.get_default_graph()

    #     with graph.as_default():
    #         with session.as_default():
    #             input_shape =   ( 6, 200)
    #             if (args.arch == 'Resnet18'):
    #                 input_shape =   ( None , 6, 200)
    #             model = get_model(args.arch)
    #             # model = model_object.getModel(input_shape[1:])
    #             model.build(input_shape = input_shape)
    #             run_meta = tf.compat.v1.RunMetadata()
    #             opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    #             # Optional: save printed results to file
    #             # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
    #             # opts['output'] = 'file:outfile={}'.format(flops_log_path)

    #             # We use the Keras session graph in the call to the profiler.
    #             flops = tf.compat.v1.profiler.profile(graph=graph,
    #                                                   run_meta=run_meta, cmd='op', options=opts)

    #     tf.compat.v1.reset_default_graph()

    #     return flops.total_float_ops
    input_shape =   ( 6, 200)
    if (args.arch == 'ResNet' or args.arch == 'IMUNet' ):
        input_shape =   ( None , 6, 200)
    model = get_model(args.arch)
    # model = model_object.getModel(input_shape[1:])
    model.build(input_shape = input_shape)
    
    # save_model(model,'124446.model')
    # model = load_model('124446.model', compile=False)
    model.summary()
   
    def custom_loss_function(y_true, y_pred):
        return  K.mean((y_true - y_pred) ** 2, axis=0)
    
    from keras.callbacks import LearningRateScheduler

    # This is a sample of a scheduler I used in the past
    
   
    # model.compile(sgd, loss=custom_loss_function)
    # history = model.fit_generator(train_generator, steps_per_epoch=1000, epochs=args.epochs, validation_data=val_generator,
    #                         max_queue_size=100, workers=4, verbose=1, use_multiprocessing=True, callbacks=[sgdr])

    # model.compile(optimizer = "adam",loss='categorical_crossentropy') 
    # model.summary()
    
    model.compile(optimizer='adam', loss=custom_loss_function)
    print('--------------------------------------------------------------------------------')
    
    # flops = get_flops(model, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")
    print('--------------------------------------------------------------------------------')
    checkpoint_path = args.out_dir + '/' + args.arch +  '.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=True, save_weights_only=True, verbose=1)
    # path = args.model_path + args.arch + '/'
    # if (path) is not None and not osp.isdir(path):
    #     os.makedirs(path)
    
    history = model.fit(train_generator,epochs=args.epochs,validation_data=val_generator , callbacks=[cp_callback
                        ,LR_Restart(args.lr , args.lr*1e-6 , 10 , verbose = 1)])
     
    model.load_weights(checkpoint_path)
    model.save_weights(args.out_dir + '/' + args.arch + '.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter=True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(args.out_dir + '/' + args.arch + '.tflite', 'wb') as f:
      f.write(tflite_model)
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    with open(args.out_dir + '/' + 'train_' + args.arch + '.npy', 'wb') as f:
        np.save(f, train_loss)
    with open(args.out_dir + '/' + 'val_' + args.arch + '.npy', 'wb') as f:
        np.save(f, val_loss)
    # json_txt = model.to_json()
    # json_object = json.loads(json_txt)
    # with open('model_resnet_18.json', 'w') as f:
    #     json.dump(json_object, f, indent=2)
    
    # model.save_weights('weights_resnet18.h5')
    
    return history

def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int32)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    print ('--------------------------------------------------------')
    print (pos[0])
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    
    # new_value = ts_ext[0:pos.shape[0]]
    # new_value[-1] = ts[-1]
    pos = interp1d( ts_ext , pos, axis=0)(ts)
    return pos

def run_test(model, data_generator, batch_size):
    
    
    # preds_all = model.predict(data_generator)
    
    # targs  = data_generator.all_targets[0:-1]
    # targs = np.array(targs)
    # targets_all = np.reshape(targs, [targs.shape[0]*targs.shape[1],2])
    length_test = len(data_generator.index_map) // batch_size
    targets_all = []
    preds_all = []
    for item in range(0,length_test):
        feat_list = []
        target_list = []
        for i in range(item*batch_size, (item*batch_size)+batch_size):
            seq_id, frame_id = data_generator.index_map[i][0], data_generator.index_map[i][1]
           
            feat = data_generator.features[seq_id][frame_id:frame_id + args.window_size]
            targ = data_generator.targets[seq_id][frame_id]
            feat_list.append(feat.T)
            target_list.append(targ)
            
        
        feat = np.array(feat_list)
        targ = np.array(target_list)
        pred = model.predict(feat)
        
        targets_all.append(targ)
        preds_all.append(pred)
    
    
    feat_list = []
    target_list = []
    for i in range((item*batch_size)+batch_size, len(data_generator.index_map)):
        seq_id, frame_id = data_generator.index_map[i][0], data_generator.index_map[i][1]
       
        feat = data_generator.features[seq_id][frame_id:frame_id + args.window_size]
        targ = data_generator.targets[seq_id][frame_id]
        feat_list.append(feat.T)
        target_list.append(targ)
        
    
    feat = np.array(feat_list)
    targ = np.array(target_list)
    if feat.size > 0 : 
        pred = model.predict(feat)
        targets_all.append(targ)
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    
    preds_all = np.concatenate(preds_all, axis=0)
    
    
    return targets_all, preds_all
  
    
    
def test_sequence(args):
    if args.dataset != 'px4' and args.dataset != 'oxiod':
        out_dir = args.out_dir + '/'
        if args.test_path is not None:
            if args.test_path[-1] == '/':
                args.test_path = args.test_path[:-1]
            root_dir = osp.split(args.test_path)[0]
            test_data_list = [osp.split(args.test_path)[1]]
        elif args.test_list is not None:
            root_dir = args.root_dir
            with open(args.test_list) as f:
                test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
        else:
            raise ValueError('Either test_path or test_list must be specified.')
    else:
        out_dir = args.out_dir
        root_dir = args.root_dir + '/validation'
        test_data_list = os.listdir(root_dir)
        if args.dataset == 'px4':
            args.batch_size = 1
    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    
    

    # if (args.out_dir) is not None and not osp.isdir(out_dir):
    #     os.makedirs(out_dir)
        
        
    # test_generator = get_dataset_from_list(args.root_dir, args.test_list, args, mode='test')
    
    input_shape =   ( 6, 200)
    if (args.arch == 'ResNet' or args.arch == 'IMUNet'):
        input_shape =   ( None , 6, 200)
    # model = ResNet18(2)
    model = get_model(args.arch)
    # model = model_object.getModel(input_shape[1:])
    model.build(input_shape = input_shape)
    
    # checkpoint_path = args.model_path + args.arch + '/' + args.arch +  '.ckpt'
    # model.load_weights(checkpoint_path)
    
    
    # model_path = args.model_path + args.arch + '/' + args.arch + '.h5'
    # model.load_weights(model_path)
    
    model.load_weights(args.model_path)
    
    if (args.dataset == 'proposed'):
        tflite_path = current_dir + '/RONIN_keras/Test_out/proposed/' +\
                                args.arch + '/tflite/' + args.arch + '.tflite'
        if not osp.exists(tflite_path):
            model._set_inputs(tf.random.uniform((1,6,200)))
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.experimental_new_converter=True
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            os.makedirs(current_dir + '/RONIN_keras/Test_out/proposed/' +\
                                    args.arch + '/tflite/')
            with open(tflite_path, 'wb') as f:
              f.write(tflite_model)
    
    # model = Model()
    # with open('/home/behnam/Desktop/IMU_paper/Code/NAS/model_json.json', 'r') as f:
    #         model_json = json.load(f)
    # model = model_from_config(model_json)
    # model.load_weights('/home/behnam/Desktop/IMU_paper/Code/NAS/model_weights.h5')
    
    # model.save('model_res.h5')

    # converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('model_res.h5')
    # tfmodel = converter.convert() 
    # open ('model_res.tflite' , "wb") .write(tfmodel)
    # print('TFLite is saved --- segmented')
    
    model.summary()
    
    # model = load_model ('/home/behnam/Desktop/IMU_paper/Code/Ronin/model_nas.h5')
    # model.load_weights('weights_resnet18.h5')
    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []
    test_batch_size = 256
    if (args.dataset == 'px4'):
        test_batch_size = 10
    pred_per_min = 200 * 60
    for data in test_data_list:
        test_generator = get_dataset(root_dir, [data], args, mode='test')
        ind = np.array([i[1] for i in test_generator.index_map if i[0] == 0], dtype=np.int32)
        
            
        targets, preds = run_test(model, test_generator, test_batch_size)
       
        losses = np.mean((targets - preds) ** 2, axis=0)
        # losses = K.mean((targets - preds) ** 2, axis=0)
        preds_seq.append(preds)
        targets_seq.append(targets)
        losses_seq.append(losses)
        
        if args.dataset != 'px4':
            

            pos_pred = recon_traj_with_preds(test_generator, preds)[:, :2]
            pos_gt = test_generator.gt_pos[0][:, :2]
    
            traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
            ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
            ate_all.append(ate)
            rte_all.append(rte)
            pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)
    
            print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))
        
            # Plot figures
            kp = preds.shape[1]
            if kp == 2:
                targ_names = ['vx', 'vy']
            elif kp == 3:
                targ_names = ['vx', 'vy', 'vz']
    
            plt.figure('{}'.format(data), figsize=(16, 9))
            plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
            plt.plot(pos_pred[:, 0], pos_pred[:, 1])
            plt.plot(pos_gt[:, 0], pos_gt[:, 1])
            plt.title(data)
            plt.axis('equal')
            plt.legend(['Predicted', 'Ground truth'])
            plt.subplot2grid((kp, 2), (kp - 1, 0))
            plt.plot(pos_cum_error)
            plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
            for i in range(kp):
                plt.subplot2grid((kp, 2), (i, 1))
                plt.plot(ind, preds[:, i])
                plt.plot(ind, targets[:, i])
                plt.legend(['Predicted', 'Ground truth'])
                plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
            plt.tight_layout()
    
            if args.show_plot:
                plt.show()
    
            if (out_dir) is not None and osp.isdir(out_dir):
                np.save(osp.join((out_dir), data + '_gsn.npy'),
                        np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
                plt.savefig(osp.join((out_dir), data + '_gsn.png'))
    
            plt.close('all')
        else:

            pos_pred = np.cumsum(preds, axis=0)
            # pos_gt = seq_dataset.gt_pos[0][args.window_size:, ] 
            pos_gt = np.cumsum(targets, axis=0)

            ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
            ate_all.append(ate)
            pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

            print('Sequence {}, loss {} / {}, ate {:.6f}'.format(data, losses, np.mean(losses), ate))

            # Plot figures
            kp = preds.shape[1]
            if kp == 2:
                targ_names = ['vx', 'vy']
            elif kp == 3:
                targ_names = ['vx', 'vy', 'vz']

            plt.figure('{}'.format(data), figsize=(16, 9))

            plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1, projection='3d')
            plt.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2])
            plt.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2])
            plt.title(data)
            # plt.axis('equal')
            plt.legend(['Predicted', 'Ground truth'])
            plt.subplot2grid((kp, 2), (kp - 1, 0))
            plt.plot(pos_cum_error)
            plt.legend(['ATE:{:.3f}'.format(ate_all[-1])])
            for i in range(kp):
                plt.subplot2grid((kp, 2), (i, 1))
                plt.plot(ind, preds[:, i])
                plt.plot(ind, targets[:, i])
                plt.legend(['Predicted', 'Ground truth'])
                plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
            plt.tight_layout()

            if args.show_plot:
                plt.show()

            if args.out_dir is not None and osp.isdir(args.out_dir):
                np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                        np.concatenate([pos_pred, pos_gt], axis=1))
                plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

            plt.close('all')
        

    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)
    if args.dataset != 'px4':
    # Export a csv file
        if (out_dir) is not None and osp.isdir(out_dir):
            with open(osp.join((out_dir), 'losses.csv'), 'w') as f:
                if losses_seq.shape[1] == 2:
                    f.write('seq,vx,vy,avg,ate,rte\n')
                else:
                    f.write('seq,vx,vy,vz,avg,ate,rte\n')
                for i in range(losses_seq.shape[0]):
                    f.write('{},'.format(test_data_list[i]))
                    for j in range(losses_seq.shape[1]):
                        f.write('{:.6f},'.format(losses_seq[i][j]))
                    f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))
    
        print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}'.format(
            np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all)))
    else:
        if args.out_dir is not None and osp.isdir(args.out_dir):
            with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
                if losses_seq.shape[1] == 2:
                    f.write('seq,vx,vy,avg,ate\n')
                else:
                    f.write('seq,vx,vy,vz,avg,ate\n')
                for i in range(losses_seq.shape[0]):
                    f.write('{},'.format(test_data_list[i]))
                    for j in range(losses_seq.shape[1]):
                        f.write('{:.6f},'.format(losses_seq[i][j]))
                    f.write('{:.6f},{:6f}\n'.format(losses_avg[i], ate_all[i]))

        print('----------\nOverall loss: {}/{}, avg ATE:{}'.format(
            np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all)))
    return losses_avg
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', default='', type=str)
    parser.add_argument('--val_list', type=str, default='')
    parser.add_argument('--test_list', type=str, default='')
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default='', help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='px4',
                        choices=['ronin', 'ridi', 'proposed', 'oxiod', 'px4'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--arch', type=str, default='IMUNet',
             choices=['ResNet', 'MobileNet', 'MobileNetV2','MnasNet', 'EfficientNet', 'IMUNet'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')
    parser.add_argument('--test_status', type=str, default='seen', choices=['seen', 'unseen'])
    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')

    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})
    dataset = args.dataset
    import os

    # Get the current working directory
    current_dir = os.getcwd()
    
    from pathlib import Path
    path = Path(current_dir)
    current_dir = str(path.parent.absolute())

    # Print the current working directory
    print("Current working directory: {0}".format(current_dir))

    if args.mode == 'train':
        if dataset == 'ronin':
            args.train_list = current_dir +'/Datasets/ronin/list_train.txt'
            args.val_list = current_dir + '/Datasets/ronin/list_val.txt'
            args.root_dir = current_dir + '/Datasets/ronin/train_dataset_1'
            args.out_dir = current_dir + '/RONIN_keras/Train_out/' + args.arch + '/ronin'
            
        elif dataset == 'ridi':
            args.train_list = current_dir + '/Datasets/ridi/data_publish_v2/list_train_publish_v2.txt'
            args.val_list = current_dir + '/Datasets/ridi/data_publish_v2/list_test_publish_v2.txt'
            args.root_dir = current_dir + '/Datasets/ridi/data_publish_v2'
            args.out_dir = current_dir + '/RONIN_keras/Train_out/' + args.arch + '/ridi'
        elif dataset == 'proposed':
            args.train_list = current_dir +'/Datasets/proposed/list_train.txt'
            args.val_list = current_dir +'/Datasets/proposed/list_test.txt'
            args.root_dir = current_dir + '/Datasets/proposed'
            args.out_dir = current_dir +'/RONIN_keras/Train_out/' + args.arch + '/proposed'
        elif dataset == 'oxiod':
            args.train_list = ''
            args.val_list = ''
            args.root_dir = current_dir + '/Datasets/oxiod'
            args.out_dir = current_dir +'/RONIN_keras/Train_out/' + args.arch + '/oxiod'
        elif dataset == 'px4':
            args.step_size = 1
            args.train_list = ''
            args.val_list = ''
            args.root_dir = current_dir + '/Datasets/px4'
            args.out_dir = current_dir +'/RONIN_keras/Train_out/' + args.arch + '/px4'
        train(args)
    elif args.mode == 'test':
        if args.test_status == 'unseen':
            if dataset != 'ronin':
                raise ValueError('Undefined mode')
        if dataset == 'ronin':
            args.model_path = current_dir + '/RONIN_keras/Train_out/' + args.arch + \
                              '/ronin/' + args.arch +  '.ckpt'
            
            
                            
                              
            
            if args.test_status == 'seen':
                args.root_dir =  current_dir + '/Datasets/ronin/seen_subjects_test_set'
                # args.root_dir = current_dir + '/Datasets/ronin/seen_subjects_test_set'       
                
                args.test_list = current_dir + '/Datasets/ronin/list_test_seen.txt'
                args.out_dir = current_dir + '/RONIN_keras/Test_out/ronin/seen/'  + args.arch
            else:
                args.root_dir = current_dir + '/Datasets/ronin/unseen_subjects_test_set'
                # args.root_dir = current_dir + '/Datasets/ronin/unseen_subjects_test_set'       
                
                args.test_list = current_dir + '/Datasets/ronin/list_test_unseen.txt'
                args.out_dir = current_dir + '/RONIN_keras/Test_out/ronin/unseen/'  + args.arch

        elif dataset == 'ridi':
            args.model_path = current_dir + '/RONIN_keras/Train_out/' + args.arch + \
                              '/ridi/'+ args.arch +  '.ckpt'
            args.test_list = current_dir + '/Datasets/ridi/data_publish_v2/list_test_publish_v2.txt'
            args.root_dir = current_dir + '/Datasets/ridi/data_publish_v2'
            args.out_dir = current_dir + '/RONIN_keras/Test_out/ridi/' + args.arch
        elif dataset == 'proposed':
            args.model_path = current_dir + '/RONIN_keras/Train_out/' + args.arch + '/proposed/'+ args.arch +  '.ckpt'
            args.test_list = current_dir + '/Datasets/proposed/list_test.txt'
            args.root_dir = current_dir + '/Datasets/proposed'
            args.out_dir = current_dir + '/RONIN_keras/Test_out/proposed/' + args.arch
        elif dataset == 'oxiod':
            args.model_path = current_dir + '/RONIN_keras/Train_out/' + args.arch + '/oxiod/'+ args.arch +  '.ckpt'
            args.test_list = current_dir + '/Datasets/oxiod/'
            args.root_dir = current_dir + '/Datasets/oxiod'
            args.out_dir = current_dir + '/RONIN_keras/Test_out/oxiod/' + args.arch
        elif dataset == 'px4':
            args.model_path = current_dir + '/RONIN_keras/Train_out/' + args.arch + '/px4/'+ args.arch +  '.ckpt'
            args.test_list = current_dir +'/Datasets/px4/'
            args.root_dir = current_dir +'/Datasets/px4'
            args.out_dir = current_dir + '/RONIN_keras/Test_out/px4/' + args.arch
        test_sequence(args)
        # onnx_convertor()
    else:
        raise ValueError('Undefined mode')
