import numpy as np
import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import Parameter


class Conv2d_X(nn.Module):
    ''' 
    kernel_size: int of kernel size
    kernel_pos: list of positions that does convolution. Value should be 0 if skipped and 1 if does convolution. Length of the list should be kernel_size**2
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,kernel_pos = [[0,1,0],[1,1,1],[0,1,0]], stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7, device = 'cuda'):

        super().__init__()
        try:
            self.kernel_pos = torch.flatten(torch.Tensor(kernel_pos)).int()
        except:
            exit()
        assert len(self.kernel_pos) == kernel_size**2, f'Expected to get total number of {kernel_size**2} elements for position list, instead got {len(self.kernel_pos)}'
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, torch.sum(self.kernel_pos,dtype= int)), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).to(device)
        if theta < 0:
            self.theta = torch.sigmoid(Parameter(torch.zeros(1), requires_grad= True)).to(device)
        else:
            self.theta = theta
        self.device = device

        self.conv_weights_reashaped = self._reshape_conv_weights()

    def forward(self, x):
        
        out_normal = F.conv2d(input=x, weight=self.conv_weights_reashaped, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding,dilation=self.conv.dilation, groups=self.conv.groups)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv_weights_reashaped.shape
            pad = self.conv.padding[0]
            dilate = self.conv.dilation[0]
            neg_pad = (kernel_size // 2)*(dilate) - pad  

            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            if neg_pad < 1:
                out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=-neg_pad, groups=self.conv.groups)
            else:
                out_diff = F.conv2d(input=x[:,:,neg_pad: -neg_pad,neg_pad: -neg_pad], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,dilation=1, groups=self.conv.groups)


            return out_normal - self.theta * out_diff

    def _reshape_conv_weights(self):
        [C_out,C_in,H_k,W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0).to(self.device)
        # conv_weight_list = [tensor_zeros if self.kernel_pos[i] == 0 else self.conv.weight[:,:,:,i] for i in range(self.kernel_size*self.kernel_size)]
        conv_weight_list = []
        j = 0
        for i in range(self.kernel_size*self.kernel_size):
            if self.kernel_pos[i] == 0:
                conv_weight_list.append(tensor_zeros)
            else:
                conv_weight_list.append(self.conv.weight[:,:,:,j])
                j += 1
        conv_weight = torch.cat(conv_weight_list, 2)
        # conv_weight = torch.cat((tensor_zeros, self.conv.weight[:,:,:,0], tensor_zeros, self.conv.weight[:,:,:,1], self.conv.weight[:,:,:,2], self.conv.weight[:,:,:,3], tensor_zeros, self.conv.weight[:,:,:,4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, self.kernel_size, self.kernel_size)

        return conv_weight

def GlobalPooling(x: torch.Tensor):
    return nn.AdaptiveAvgPool2d((1,1))(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16):
        super().__init__()
    
        self.global_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size= 1 , bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size= 1, bias = False)
        )
        self.sigmoid = torch.sigmoid


    def forward(self, x):

        out = self.global_pooling(x)
        # print(out.shape)
        out = self.se(out)
        out = self.sigmoid(out)
        return torch.mul(x,out)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel = 3):
        super(SpatialAttention, self).__init__()


        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x)

class CDCNv2(nn.Module):

    def __init__(self, basic_conv=Conv2d_X, theta=0.7, se=False, **kwarg):   
        super(CDCNv2, self).__init__()
        
        if 'padding' in kwarg.keys():
            padding = kwarg['padding']
        else:
            padding = 1
        if 'stride' in kwarg.keys():
            padding = kwarg['stride']
        else:
            stride = 1
        if 'bias' in kwarg.keys():
            bias = kwarg['bias']
        else:
            bias = False
        if 'dilation' in kwarg.keys():
            dilation = kwarg['dilation']
        else:
            dilation = 1
        if 'groups' in kwarg.keys():
            groups = kwarg['groups']
        else:
            groups = 1

        
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(128),
            SEBlock(128) if se else Identity(),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(128),
            SEBlock(128) if se else Identity(),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(128),
            SEBlock(128) if se else Identity(),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        
        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, kernel_pos= [1,1,1,1,1,1,1,1,1], stride=stride, padding=padding, bias=bias, theta= theta, dialtion = dilation, groups = groups),
            nn.ReLU(),    
        )
        
        
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]  
        
        x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	  
        x_Block2_32x32 = self.downsample32x32(x_Block2)   # x [128, 32, 32]  
        
        x_Block3 = self.Block3(x_Block2)	    # x [128, 32, 32]  	
        x_Block3_32x32 = self.downsample32x32(x_Block3)   # x [128, 32, 32]  
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]  
        
        #pdb.set_trace()
        
        x = self.lastconv1(x_concat)    # x [128, 32, 32] 
        x = self.lastconv2(x)    # x [64, 32, 32] 
        x = self.lastconv3(x)    # x [1, 32, 32] 
        
        map_x = x.squeeze(1)
        
        return map_x , x_concat #, x_Block1, x_Block2, x_Block3, x_input





