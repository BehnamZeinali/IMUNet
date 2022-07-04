
"""
Created on Wed Jun 15 00:04:08 2022

@author: behnam
"""
import os
import pandas 
import numpy as np
import scipy
import quaternion
from scipy.ndimage.filters import gaussian_filter1d
import scipy.interpolate
import math













def readOXIODData(root_dir, file_name , mode):
   
    new_data_list = []
    ox_mode = 'raw'
   
    if mode == 'train':
        temp_list = root_dir + '/' + file_name + '/Train.txt' 
    else:
        temp_list = root_dir + '/' + file_name + '/Test.txt' 
    if  file_name != 'test': 
        with open(temp_list) as f:
            temp_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
        
    if file_name == 'handheld':
        for temp_file in temp_data_list:
            new_data_root = root_dir + '/' + file_name + '/' + temp_file + '/' + ox_mode 
            data_list_handheld = os.listdir(new_data_root)
            for new_file in data_list_handheld:
              if ('imu' in new_file):
                  new_data = new_data_root + '/' + new_file
                  new_data_list.append(new_data)
    # elif file_name == 'large scale':
    #     new_data_root = root_dir + '/' + file_name 
    #     data_list_ls = os.listdir(new_data_root)
    #     for new_file_ in data_list_ls:
    #         if not('floor' in new_file_):
    #             continue
    #         new_data_root_ = new_data_root +  '/' + new_file_ + '/'+ ox_mode
    #         data_list_ls_ = os.listdir(new_data_root_)
    #         for new_file__ in data_list_ls_:
    #           if ('imu' in new_file__):
    #               new_data = new_data_root_ + '/' + new_file__
    #               new_data_list.append(new_data)
    elif  file_name == 'test':
        new_data_root = root_dir + '/' + file_name 
        data_list_test = os.listdir(new_data_root)
        for file in data_list_test:
            root = new_data_root + '/' + file
            if not('DS' in root):
                list_1 = os.listdir(root)
                for file_1 in list_1:
                    root_1 = root + '/' + file_1
                    if not('DS' in root_1):
                        list_2 = os.listdir(root_1)
                        for file_2 in list_2:
                            if ('imu' in file_2):
                                new_data = root_1 + '/' + file_2
                                new_data_list.append(new_data)
            
        
        
        
            
    else:             
         for temp_file in temp_data_list:
            split_file = temp_file.split('/')
            new_data = root_dir + '/' + file_name + '/' + split_file[0] + '/' + ox_mode + '/' + split_file[-1]
            new_data_list.append(new_data)

    return new_data_list





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', default='/home/behnam/Downloads/Oxford Inertial Odometry Dataset_2.0/Oxford Inertial Odometry Dataset/list_train.txt' , type=str)
    parser.add_argument('--test_list', type=str, default='/home/behnam/Downloads/Oxford Inertial Odometry Dataset_2.0/Oxford Inertial Odometry Dataset/list_test.txt')
    parser.add_argument('--root_dir', type=str, default='/home/behnam/Downloads/Oxford Inertial Odometry Dataset_2.0/Oxford Inertial Odometry Dataset')
    args = parser.parse_args()
    modes = ['train' , 'validation']
    for mode in modes:
        if mode == 'train':
            txt_path = args.train_list
        else:
            txt_path = args.test_list
        
        
        with open(txt_path) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
            
        data_number = 0
        new_dataset_path = args.root_dir+ '/' + mode
        if not os.path.isdir(new_dataset_path):
            os.makedirs(new_dataset_path)
        for data in data_list:
               
                data_path = readOXIODData(args.root_dir, data , mode)
           
                for path in data_path:
                    data_path_ = args.root_dir+ '/' + mode +  '/data_'+ data + '_' + str(data_number) + '.csv'
                    if os.path.exists(data_path_):
                        data_number =  data_number + 1
                        print ('Data already saved')
                        continue
                    
                    imu_all = pandas.read_csv(path)
                    
                    if ('trolley/data2' in path):
                        target_path = path.replace('imu' , 'hand')
                    elif ('floor' in path):
                        if ('imu11' in path):
                            continue
                        target_path = path.replace('imu' , 'tango')
                        target_path = target_path.replace('raw' , 'tango')
                        
                    else:
                        target_path = path.replace('imu' , 'vi')
                    target_cvs = pandas.read_csv(target_path)
                    
                    
                    ts = imu_all.iloc[:,0] 
                    ts_ = target_cvs.iloc[:,0]
                    
                    if ('nexus' in path):
                        ts_ = ts_ / 1e6
                    elif ('floor' in path):
                        # n = len(str(ts_[0]).split('.')[0])
                        # ts = ts % math.pow(10,n)
                        ts_ = target_cvs.iloc[:,1]/ 1e3
                    else:
                        ts_ = ts_ / 1e9
                    
                    if (ts_[ts_.size-1] > ts_[0]):
                        # continue
                        
                        # print (ts[0])
                        # print ('------------------------------------')
                        # print (ts_[0])
                        
                        # print (ts[ts.size-1])
                        # print ('------------------------------------')
                        # print (ts_[ts_.size-1])
                        sync_time = []
                        sync_index = []
                        for i, v in enumerate(ts_):
                            if v >= ts[0] and v <= ts[ts.size-1]:
                                sync_time.append(v)
                                sync_index.append(i)
                            
                                
                        if len(sync_time)>0:
                            # print (' data is read')
                            # print (path)
                            sync_time = np.array(sync_time)
                            
                            
                            gyro = imu_all.iloc[:, 4:7]
                            acce = imu_all.iloc[:, 10:13]
                            
                            if acce.size == 0:
                                continue
                            
                            func = scipy.interpolate.interp1d(ts, gyro, axis=0)
                            output_gyro = func(sync_time)
                            
                            func = scipy.interpolate.interp1d(ts, acce, axis=0)
                            output_acce = func(sync_time)
                            
                           
                            target_pos = target_cvs.iloc[:,2:5]
                            
                            target_pos = target_pos[sync_index[0]:sync_index[-1]+1]
                            orientation = target_cvs.iloc[:,5:9]
                            orientation = orientation[sync_index[0]:sync_index[-1]+1]
                            
                            column_list = 'time,gyro_x,gyro_y,gyro_z,acce_x'.split(',') + \
                                      'acce_y,acce_z'.split(',') + \
                                      'pos_x,pos_y,pos_z,rv_x,rv_y,rv_z,rv_w'.split(',')
                            sync_time = np.reshape (sync_time , [sync_time.size,1])
                            data_mat = np.concatenate([sync_time, output_gyro,
                                                       output_acce,
                                                       target_pos,
                                                       orientation], axis=1)
                
                            data_pandas = pandas.DataFrame(data_mat, columns=column_list)
                            
                            data_pandas.to_csv(data_path_)
                            data_number = data_number + 1
                            print('Dataset written to ' + data_path_)
                            
                        else:
                            print ('length problem')
                            # 
                    
                    else:
                        print ('ts problem')
                        # print (len(ts))
                        # print (path)
               
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
