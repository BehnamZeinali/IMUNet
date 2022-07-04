"""
This is the Edge device simulation implementation using the Tensorflow library. 
Tensorflow lite models and the tensorflow lite interpreter habe been used.

First it uploads the .oonx file from each model in a proposed dataset convert the model to .tflite
and simulate the edge implementation. Same method has been implemented for Adnroid devices


"""

import os

import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import quaternion
import pandas
import numpy as np
import matplotlib.pyplot as plt
from metric import compute_ate_rte


    
def run_simulation(tflite_path , list_path):
    
    with open(list_path + 'list_test.txt') as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    
        
    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []

    pred_per_min = 200 * 60
    for data in data_list:
        
        data_dir = list_path + data + '/processed/data.csv'
        
        data_dir = '/home/behnam/PycharmProjects/IMUNet_Android/app/src/main/res/raw/behnam_28__s10.csv'
        imu_all = pandas.read_csv(data_dir)
        ts = imu_all[['time']].values / 1e09
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
        tango_pos = imu_all[['pos_x', 'pos_y', 'pos_z']].values
        init_tango_ori = quaternion.quaternion(*imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values[0])
        game_rv = quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)
        init_rotor = init_tango_ori * game_rv[0].conj()
        init_pos = tango_pos[0][:2]
        tango_pos = tango_pos[:,:2]
        
        
        ## load the tensorflow lite model
        interpreter = tf.lite.Interpreter(tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    
        input_vector = np.zeros([6,200],dtype=np.float32)
        pos_result = []
        pos_result.append(init_pos)
        zero = np.zeros([1,])
        pos = np.zeros([gyro.shape[0] + 2, 2])
        previous_pos = 0
        result_index = 0
        last_time = ts[0]
        time_sum = 0
        for i in range(gyro.shape[0]):
            
            if (i>0):
                tempp = ts[i]-last_time
                time_sum = time_sum + tempp
                dts = time_sum/i
                last_time = ts[i]
            
       
            gm_rv = game_rv[i]
            ori = init_rotor * gm_rv
            # ori = gm_rv
            gyro_temp = gyro[i]
            gyro_temp_q = quaternion.from_float_array(np.concatenate([zero, gyro_temp], axis=0))
            oriented_gyro = quaternion.as_float_array(ori * gyro_temp_q * ori.conj())[1:]
            
            acce_temp = acce[i]
            acce_temp_q = quaternion.from_float_array(np.concatenate([zero, acce_temp], axis=0))
            oriented_acce = quaternion.as_float_array(ori * acce_temp_q * ori.conj())[1:]
            features = np.concatenate([oriented_gyro,  oriented_acce], axis=0)
            if (i < 200):
                input_vector[:,i] = features
            else:
                input_vector[:,0:-1] = input_vector[:,1:200]
                input_vector[:,-1] = features
            if (i > 198):
            # dts = np.mean(ts[1:i] - ts[:i-1])
            
            
                input_temp = np.reshape(input_vector , (1,6,200))
                interpreter.set_tensor(input_details[0]['index'], input_temp)
                interpreter.invoke()
                out_result = interpreter.get_tensor(output_details[0]['index'])
                
                non_integrated = out_result*dts
                if ( i == 199):
                    previous_pos = non_integrated
                    temp = non_integrated + init_pos
                    pos_result.append(temp)
                else:
                    previous_pos = non_integrated + previous_pos
                    temp = previous_pos + init_pos
                    pos_result.append(temp)
                if (i%500 == 0):
                    print('Pos: ',out_result)
                    print('j= ',i)
        
        pos_result_np = np.array(pos_result)    
    
                
        result_last = np.zeros([pos_result_np.shape[0], 2])            
        for i in range  (pos_result_np.shape[0]):
             result_last[i] = pos_result_np[i]  
             
        losses = np.mean((tango_pos[198:,] - result_last) ** 2, axis=0)
        preds_seq.append(result_last)
        targets_seq.append(tango_pos)
        losses_seq.append(losses)
        
        ate, rte = compute_ate_rte(result_last, tango_pos[198:,], pred_per_min)
        ate_all.append(ate)
        rte_all.append(rte)
        pos_cum_error = np.linalg.norm(result_last - tango_pos[198:,], axis=1)

        print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))

        
        
        
        kp =2
        plt.figure('{}'.format(data), figsize=(16, 9))
        plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
        plt.plot(result_last[:, 0], result_last[:, 1])
        plt.plot(tango_pos[:, 0], tango_pos[:, 1])
        plt.title(data)
        plt.axis('equal')
        plt.legend(['Predicted', 'Ground truth'])
        plt.subplot2grid((kp, 2), (kp - 1, 0))
        plt.plot(pos_cum_error)
        plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
        # for i in range(kp):
        #     plt.subplot2grid((kp, 2), (i, 1))
        #     plt.plot(ind, preds[:, i])
        #     plt.plot(ind, tango_pos[:, i])
        #     plt.legend(['Predicted', 'Ground truth'])
        #     plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
        plt.tight_layout()

        plt.show()

        # if args.out_dir is not None and osp.isdir(args.out_dir):
        #     np.save(osp.join(args.out_dir, data + '_gsn.npy'),
        #             np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
        #     plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

        plt.close('all')
        
        # # ate, rte = compute_ate_rte(result_last, tango_pos, pred_per_min)
        # plt.figure('Models', figsize=(16, 9))
        # # plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
        # plt.plot(result_last[:, 0], result_last[:, 1])
        # plt.plot(tango_pos[:, 0], tango_pos[:, 1])
        # plt.title('Track 1')
        # plt.axis('equal')
        # plt.legend(['Ground truth','predicted'])
        # plt.show()
    # plt.legend(['ATE:{:.3f}'.format(ate)])
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
   
    parser.add_argument('--arch', type=str, default='IMUNet',
             choices=['ResNet', 'MobileNet', 'MobileNetV2','MnasNet', 'EfficientNet', 'IMUNet'])
   
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()

    # Get the current working directory
    current_dir = os.getcwd()
    
    from pathlib import Path
    path = Path(current_dir)
    current_dir = str(path.parent.absolute())
    
    
    tflite_path =  current_dir + '/RONIN_keras/Test_out/proposed/' +\
                            args.arch + '/tflite/' + args.arch + '.tflite'
    if os.path.exists(tflite_path):
        
            
        list_path = current_dir + '/Datasets/proposed/'
        run_simulation (tflite_path , list_path)
    else:
        raise ValueError('Could not find the model')

    
