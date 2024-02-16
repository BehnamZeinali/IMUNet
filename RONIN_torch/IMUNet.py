

# This is the networks script
import torch
import torch.nn as nn

from collections import OrderedDict

class DSConv(nn.Module):
    expansion = 1
    def __init__(self, f_3x3, f_1x1, kernel_size, stride=1, dilation=1, downsample=False, padding=1 , inplace = True):
        super(DSConv, self).__init__()
        self.relu = nn.ELU()
        
        
        self.depth_wise = nn.Conv1d(f_3x3, f_3x3,kernel_size=kernel_size,groups=f_3x3,stride=stride, padding = padding ,
                            bias=False)
                            
        
        self.bn_1 = nn.BatchNorm1d(f_3x3 , eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.point_wise = nn.Conv1d(f_3x3,f_1x1,kernel_size=1 ,bias=False)
        
        self.bn_2 = nn.BatchNorm1d(f_1x1 , eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       
        self.downsample_ = nn.Sequential( nn.Conv1d(f_3x3, f_1x1, kernel_size= 1, stride=stride, bias=False),
                                        nn.BatchNorm1d(f_1x1))
        self.downsample = downsample
        
        # self.elu_ = nn.ELU()
       
    def forward(self, x):
        residual = x
        # out = self.feature(x)
        out = self.depth_wise(x)
        out = self.bn_1 (out)
        out = self.relu (out)
        out = self.point_wise(out)
        out = self.bn_2 (out)
        out = self.relu(out)
        if self.downsample:
            residual = self.downsample_(x)

        out += residual.clone()
        out = self.relu(out)
        return out
    
class DSConv_Regular(nn.Module):
    expansion = 1
    def __init__(self, f_3x3, f_1x1, kernel_size, stride=1, dilation=1, downsample=False, padding=1 , inplace = True):
        super(DSConv_Regular, self).__init__()
        self.relu = nn.ELU()
        
        
        self.depth_wise = nn.Conv1d(f_3x3, f_3x3,kernel_size=kernel_size,groups=f_3x3,stride=stride, padding = padding ,
                            bias=False)
                            
        
        self.bn_1 = nn.BatchNorm1d(f_3x3 , eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.point_wise = nn.Conv1d(f_3x3,f_1x1,kernel_size=1 ,bias=False)
        
        self.bn_2 = nn.BatchNorm1d(f_1x1 , eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
       
    def forward(self, x):
        residual = x
        # out = self.feature(x)
        out = self.depth_wise(x)
        out = self.bn_1 (out)
        out = self.relu (out)
        out = self.point_wise(out)
        out = self.bn_2 (out)
        out = self.relu(out)
      

        out += residual.clone()
        out = self.relu(out)
        return out
class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(1, 1200))  # Learnable parameter with matching shape
        self.b = nn.Parameter(torch.zeros(1, 1200))  # Learnable parameter with matching shape

    def forward(self, TensorA, TensorB):
        return TensorA - self.W * TensorB + self.b
    
    
class IMUNet(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))
    
    
    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(IMUNet, self).__init__()
        
       

        self.noise = CustomLayer()
        
        self.input_mult = 64
   
        
        self.input_block = nn.Sequential(
            nn.Conv1d(6, self.input_mult , kernel_size=7, stride=2, padding= 3, bias=False),
            nn.BatchNorm1d(self.input_mult, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1)
        )


        self.conv_1_1 = DSConv_Regular(self.input_mult, 64, 3 , downsample= False , padding=1,  stride = 1)
        self.conv_1_2 = DSConv_Regular( 64, 64, 3 , downsample= False , padding = 1)
       
        
        self.conv_3_1 = DSConv(64, 64, 3 , downsample= True , padding = 1 ,  stride = 1)
        self.conv_3_2 = DSConv_Regular( 64, 64, 3 , downsample= False , padding = 1 , stride = 1)
        
        
        self.conv_4_1 = DSConv(64, 128, 3 , downsample= True , padding = 1  , stride = 2)
        self.conv_4_2 = DSConv_Regular( 128, 128, 3 , downsample= False , padding = 1 , stride = 1)
        
        self.conv_5_1 = DSConv(128, 256, 3 , downsample= True , padding = 1,  stride = 2)
        self.conv_5_2 = DSConv_Regular( 256, 256, 3 , downsample= False , padding = 1 , stride = 1)
        
        
        self.conv_6_1 = DSConv(256, 512, 3 , downsample= True , padding = 1,  stride = 2)
        self.conv_6_2 = DSConv_Regular( 512, 512, 3 , downsample= False , padding = 1 , stride = 1)
        
        self.conv_7_1 = DSConv(512, 1024, 3 , downsample= True , padding = 1,  stride = 2)
        self.conv_7_2 = DSConv_Regular( 1024, 1024, 3 , downsample= False , padding = 1)
        
       
        
        self.relu = nn.ELU()
        
      
        
        self.output_block = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=400,
                      kernel_size=2, stride=1),
            nn.BatchNorm1d(400),
            
          )
        
        
        
        
        
        
        self.fc = nn.Sequential(
            nn.Linear(1200, 2)
            
            
        )
        
        
        

    def forward(self, x):
      
        
        input_val = x.view(x.size(0), -1)
        x = self.input_block(x)
       
        
       
        y = self.conv_1_1(x)
     
        y = self.conv_1_2(y)
        
        x = y
        y = self.conv_3_1(x)
       
        y = self.conv_3_2(y)
        
       
        y = self.conv_4_1(y)
        
        y = self.conv_4_2(y)
       
        y = self.conv_5_1(y)
        y = self.conv_5_2(y)
        
        y = self.conv_6_1(y)
        y = self.conv_6_2(y)
        
        y = self.conv_7_1(y)
        y = self.conv_7_2(y)
        
      
       
        out = self.output_block(y)
        
        out = out.view(out.size(0), -1)
        out = self.noise(out,input_val)
        out = self.relu(out)
        
        out = self.fc(out)
        
        return out
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == '__main__':
    from torch.autograd import Variable
    x_image = Variable(torch.randn(1,6, 200))
    net = IMUNet(num_classes= 2, input_size= (6,200) ,sampling_rate= 200, num_T = 32 , num_S = 64 , hidden = 64, dropout_rate = 0.5)
    print (net)
    
   
    y = net(x_image)
    print(y)
    
    inp = torch.rand(1,6, 200)
    from pthflops import count_ops
    
    # Count the number of FLOPs
    count_ops(net, inp)
    print(net.get_num_params())



    

