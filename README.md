# IMUNet: Efficient Regression Architecture for IMU Navigation and Positioning

This is the Python implementation for the Paper: [IMUNet](https://arxiv.org/abs/2208.00068)

A new Architecture called IMUNet which is appropriate for edge-device implementation has been proposed for processing IMU measurements and performing inertial navigation. 

In this repository, the data-driven method for inertial navigation proposed in [here](https://github.com/Sachini/ronin) for the ResNet18 model has been modified.
Other than ResNet18, three state-of-the-art CNN models that have been designed for IoT device implementation have been reimplemented for inertial navigation and IMU sensor processing. 

# Architectures
Other than IMUNet, MobileNet, MnasNet, and EfficientNetB0 models have been re-implemented to work with one-dimensional IMU mesurements. 

# Dataset
Four datasets have been used in the paper.
* A new method for collecting a dataset using Android cellphones that uses [ARCore API](https://developers.google.com/ar/reference) for collecting the ground truth trajectory has been proposed and a dataset using this method along with the method proposed in [RIDI](https://github.com/higerra/ridi_imu) using a Lenovo Tango device for collecting the ground truth trajectory has been collected. A preprocessing step has been added to read and prepare the data. The collected dataset can be downloaded from [IMUNet_dataset](https://drive.google.com/file/d/1A49YZ1G8vkEJPIb51n-MFITropK4pIhu/view?usp=drive_link). Also, you can download the pre-trained models in Pytorch from [pre-trained models](https://drive.google.com/file/d/1NGwBhvh-KjVg0CpMeFtYI72G4J0LOPWE/view?usp=sharing). 

Other datasets are:

1- RONIN which is available at [here](https://ronin.cs.sfu.ca/) 

2- RIDI which is available at [DropBox](https://www.dropbox.com/s/9zzaj3h3u4bta23/ridi_data_publish_v2.zip?dl=0)  

3- OxIOD: The Dataset for Deep Inertial Odometry which is available at [OxIOD](http://deepio.cs.ox.ac.uk/ )   


# Keras Implementation
The data-driven method for inertial navigation proposed in [RONIN](https://github.com/Sachini/ronin) for the ResNet18 model with all the new architectures and datasets as well as the proposed architecture have been implemented in Tensorflow-Keras. 

# Android
The Android Application for collecting a new dataset is available at [Android](https://github.com/BehnamZeinali/IMUNet_Android). 


