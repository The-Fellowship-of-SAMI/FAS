import torch 
import torch.nn as nn
import math
import torch.nn.functional as F

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

def GlobalPooling(x: torch.Tensor):
    _, c, h, w = x.shape
    # return nn.Conv2d(c,c,w, groups= c)(x)
    # %timeit x.sum(2).sum(2).unsqueeze(-1).unsqueeze(-1)/(w*h)
    # %timeit nn.AvgPool2d(kernel_size = (w,h),stride = 1)(x)   
    # %timeit nn.AdaptiveAvgPool2d((1,1))(x)
    return nn.AdaptiveAvgPool2d((1,1))(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16):
        super(SEBlock, self).__init__()
    
        self.global_pooling = GlobalPooling
        self.se = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size= 1 , bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size= 1, bias = False)
        )
        self.sigmoid = torch.sigmoid


    def forward(self, x):

        out = self.global_pooling(x)
#         print(out.shape)
        out = self.se(out)
        
        out = self.sigmoid(out)
        
        
        return torch.mul(x,out)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
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



class CDCN(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):   
        super(CDCN, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        
        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
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
        
        return map_x, x_concat#, x_Block1, x_Block2, x_Block3, x_input

class CDCNpp(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7 ):   
        super(CDCNpp, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
            
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            
            basic_conv(128, int(128*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.6)),
            nn.ReLU(),  
            basic_conv(int(128*1.6), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            basic_conv(128, int(128*1.4), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.4)),
            nn.ReLU(),  
            basic_conv(int(128*1.4), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Original
        
        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        
      
        self.sa1 = SpatialAttention(kernel = 7)
        self.sa2 = SpatialAttention(kernel = 5)
        self.sa3 = SpatialAttention(kernel = 3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   
        
        x_Block1 = self.Block1(x)	    	    	
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)   
        
        x_Block2 = self.Block2(x_Block1)	    
        attention2 = self.sa2(x_Block2)  
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)  
        
        x_Block3 = self.Block3(x_Block2)	    
        attention3 = self.sa3(x_Block3)  
        x_Block3_SA = attention3 * x_Block3	
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)   
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    
        
        #pdb.set_trace()
        
        map_x = self.lastconv1(x_concat)
        
        map_x = map_x.squeeze(1)
        
        return map_x, x_concat#, attention1, attention2, attention3, x_input

class SE_CDCN(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7, se=False, sa=False):   
        super(SE_CDCN, self).__init__()
        
        self.se= se
        self.sa= sa
        
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            SEBlock(128) if se else Identity(),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            SEBlock(128) if se else Identity(),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            SEBlock(128) if se else Identity(),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        
        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        if self.sa:
            self.sa1 = SpatialAttention(kernel = 7)
            self.sa2 = SpatialAttention(kernel = 5)
            self.sa3 = SpatialAttention(kernel = 3)
        
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        if self.sa:
            attention1 = self.sa1(x_Block1)
            x_Block1_SA = attention1 * x_Block1
            x_Block1_32x32 = self.downsample32x32(x_Block1_SA)   # x [128, 32, 32]  
        else:
            x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]  
        
        x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	  
        if self.sa:
            attention2 = self.sa2(x_Block2)
            x_Block2_SA = attention2 * x_Block2
            x_Block2_32x32 = self.downsample32x32(x_Block2_SA)   # x [128, 32, 32]  
        else:
            x_Block2_32x32 = self.downsample32x32(x_Block2)   # x [128, 32, 32]  
         
        
        x_Block3 = self.Block3(x_Block2)	    # x [128, 32, 32]  	
        if self.sa:
            attention3 = self.sa3(x_Block3)
            x_Block3_SA = attention3 * x_Block3
            x_Block3_32x32 = self.downsample32x32(x_Block3_SA)   # x [128, 32, 32]  
        else:
            x_Block3_32x32 = self.downsample32x32(x_Block3)   # x [128, 32, 32]  
        
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]  
        
        #pdb.set_trace()
        
        x = self.lastconv1(x_concat)    # x [128, 32, 32] 
        x = self.lastconv2(x)    # x [64, 32, 32] 
        x = self.lastconv3(x)    # x [1, 32, 32] 
        
        map_x = x.squeeze(1)
        
        return map_x , x_concat#, x_Block1, x_Block2, x_Block3, x_input

class ATT_CDCN(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7, se=False, sa=False, pos= 'first'):   
        super(ATT_CDCN, self).__init__()
        assert pos in ['first','last']
        self.se= se
        self.sa= sa
        self.pos = pos
        
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             SEBlock(128) if se else Identity(),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             SEBlock(128) if se else Identity(),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             SEBlock(128) if se else Identity(),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        
        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        if self.sa:
            self.sa1 = SpatialAttention(kernel = 7)
            self.sa2 = SpatialAttention(kernel = 5)
            self.sa3 = SpatialAttention(kernel = 3)
        
        if self.se:
            self.se1 = SEBlock(128)
            self.se2 = SEBlock(128)
            self.se3 = SEBlock(128)
        
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   # x [64, 256, 256]
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]  
        
        if self.se:
            x_Block1_SE = self.se1(x_Block1)
        else:
            x_Block1_SE = x_Block1
        if self.sa:
            attention1 = self.sa1(x_Block1_SE)
            x_Block1_SA = attention1 * x_Block1_SE
            x_Block1_32x32_enhanced = self.downsample32x32(x_Block1_SA)   # x [128, 32, 32]  
        else:
            x_Block1_32x32_enhanced = self.downsample32x32(x_Block1_SE)   # x [128, 32, 32]  
            x_Block1_SA = x_Block1_SE
        
        
        if self.pos == 'first':
            x_Block2 = self.Block2(x_Block1_SA)	    # x [128, 64, 64]	  
        else:
            x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	 
            
        x_Block2_32x32 = self.downsample32x32(x_Block2)
        
        if self.se:
            x_Block2_SE = self.se2(x_Block2)
        else:
            x_Block2_SE = x_Block2
        if self.sa:
            attention2 = self.sa2(x_Block2_SE)
            x_Block2_SA = attention2 * x_Block2_SE
            x_Block2_32x32_enhanced = self.downsample32x32(x_Block2_SA)   # x [128, 32, 32]  
        else:
            x_Block2_32x32_enhanced = self.downsample32x32(x_Block2_SE)   # x [128, 32, 32]  
            x_Block2_SA = x_Block2_SE
         
        
        if self.pos == 'first':
            x_Block3 = self.Block2(x_Block2_SA)	    # x [128, 32, 32]	  
        else:
            x_Block3 = self.Block2(x_Block2)	    # x [128, 32, 32]	    	
        
        x_Block3_32x32 = self.downsample32x32(x_Block3)
        if self.se:
            x_Block3_SE = self.se3(x_Block3)
        else:
            x_Block3_SE = x_Block3
        if self.sa:
            attention3 = self.sa3(x_Block3_SE)
            x_Block3_SA = attention3 * x_Block3_SE
            x_Block3_32x32_enhanced = self.downsample32x32(x_Block3_SA)   # x [128, 32, 32]  
        else:
            x_Block3_32x32_enhanced = self.downsample32x32(x_Block3_SE)   # x [128, 32, 32]  
        
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]
        
        x_concat_enhanced = torch.cat((x_Block1_32x32_enhanced,x_Block2_32x32_enhanced,x_Block3_32x32_enhanced), dim=1)    # x [128*3, 32, 32]
        
        #pdb.set_trace()
        
        x = self.lastconv1(x_concat)    # x [128, 32, 32] 
        x = self.lastconv2(x)    # x [64, 32, 32] 
        x = self.lastconv3(x)    # x [1, 32, 32] 
        
        map_x = x.squeeze(1)
        
        return map_x , x_concat_enhanced#, x_Block1, x_Block2, x_Block3, x_input


class SATT_CDCN(ATT_CDCN):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7, se=False, sa=False, pos= 'first'):   
        super(SATT_CDCN, self).__init__(basic_conv, theta, se,sa,pos = 'last')
        '''
        SE for softmax head
        SA for depth head
        '''

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   # x [64, 256, 256]
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        
        
        if self.se:
            x_Block1_SE = self.se1(x_Block1)
        else:
            x_Block1_SE = x_Block1
        x_Block1_32x32_cls = self.downsample32x32(x_Block1_SE)   # x [128, 32, 32]  
            
        if self.sa:
            attention1 = self.sa1(x_Block1)
            x_Block1_SA = attention1 * x_Block1 
        else:
            x_Block1_SA = x_Block1
        x_Block1_32x32_depth = self.downsample32x32(x_Block1_SA)   # x [128, 32, 32]  
        
        if self.pos == 'first':
            x_Block2 = self.Block2(x_Block1_SA)	    # x [128, 64, 64]	  
        else:
            x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	 
            
        x_Block2_32x32 = self.downsample32x32(x_Block2)
        
        if self.se:
            x_Block2_SE = self.se2(x_Block2)
        else:
            x_Block2_SE = x_Block2
        x_Block2_32x32_cls = self.downsample32x32(x_Block2_SE)    
        
        
        if self.sa:
            attention2 = self.sa2(x_Block2_SE)
            x_Block2_SA = attention2 * x_Block2_SE
        else: 
            x_Block2_SA = x_Block2
        x_Block2_32x32_depth= self.downsample32x32(x_Block2_SA)   # x [128, 32, 32]  
        
        
        
        if self.pos == 'first':
            x_Block3 = self.Block2(x_Block2_SA)	    # x [128, 32, 32]	  
        else:
            x_Block3 = self.Block2(x_Block2)	    # x [128, 32, 32]	    	
        
        
        if self.se:
            x_Block3_SE = self.se3(x_Block3)
        else:
            x_Block3_SE = x_Block3
        x_Block3_32x32_cls = self.downsample32x32(x_Block3_SE)    
            
        
        if self.sa:
            attention3 = self.sa3(x_Block3_SE)
            x_Block3_SA = attention3 * x_Block3_SE
            
        else:
            x_Block3_SA = x_Block3
        x_Block3_32x32_depth = self.downsample32x32(x_Block3_SA)   # x [128, 32, 32]      
        
        
        x_concat_depth = torch.cat((x_Block1_32x32_depth,x_Block2_32x32_depth,x_Block3_32x32_depth), dim=1)    # x [128*3, 32, 32]
        
        x_concat_cls = torch.cat((x_Block1_32x32_cls,x_Block2_32x32_cls,x_Block3_32x32_cls), dim=1)    # x [128*3, 32, 32]
        
        #pdb.set_trace()
        
        x = self.lastconv1(x_concat_depth)    # x [128, 32, 32] 
        x = self.lastconv2(x)    # x [64, 32, 32] 
        x = self.lastconv3(x)    # x [1, 32, 32] 
        
        map_x = x.squeeze(1)
        
        return map_x , x_concat_cls#, x_Block1, x_Block2, x_Block3, x_input



class DATT_CDCN(ATT_CDCN):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7, se=False, sa=False, pos= 'first'):   
        super(DATT_CDCN, self).__init__(basic_conv, theta, se,sa,pos = 'last')
        '''
        No attention for softmax head
        SE_SA for depth head
        '''

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   # x [64, 256, 256]
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]  
        
        if self.se:
            x_Block1_SE = self.se1(x_Block1)
        else:
            x_Block1_SE = x_Block1
        if self.sa:
            attention1 = self.sa1(x_Block1_SE)
            x_Block1_SA = attention1 * x_Block1_SE
            x_Block1_32x32_enhanced = self.downsample32x32(x_Block1_SA)   # x [128, 32, 32]  
        else:
            x_Block1_32x32_enhanced = self.downsample32x32(x_Block1_SE)   # x [128, 32, 32]  
            x_Block1_SA = x_Block1_SE
        
        
        if self.pos == 'first':
            x_Block2 = self.Block2(x_Block1_SA)	    # x [128, 64, 64]	  
        else:
            x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	 
            
        x_Block2_32x32 = self.downsample32x32(x_Block2)
        
        if self.se:
            x_Block2_SE = self.se2(x_Block2)
        else:
            x_Block2_SE = x_Block2
        if self.sa:
            attention2 = self.sa2(x_Block2_SE)
            x_Block2_SA = attention2 * x_Block2_SE
            x_Block2_32x32_enhanced = self.downsample32x32(x_Block2_SA)   # x [128, 32, 32]  
        else:
            x_Block2_32x32_enhanced = self.downsample32x32(x_Block2_SE)   # x [128, 32, 32]  
            x_Block2_SA = x_Block2_SE
         
        
        if self.pos == 'first':
            x_Block3 = self.Block2(x_Block2_SA)	    # x [128, 32, 32]	  
        else:
            x_Block3 = self.Block2(x_Block2)	    # x [128, 32, 32]	    	
        
        x_Block3_32x32 = self.downsample32x32(x_Block3)
        if self.se:
            x_Block3_SE = self.se3(x_Block3)
        else:
            x_Block3_SE = x_Block3
        if self.sa:
            attention3 = self.sa3(x_Block3_SE)
            x_Block3_SA = attention3 * x_Block3_SE
            x_Block3_32x32_enhanced = self.downsample32x32(x_Block3_SA)   # x [128, 32, 32]  
        else:
            x_Block3_32x32_enhanced = self.downsample32x32(x_Block3_SE)   # x [128, 32, 32]  
        
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]
        
        x_concat_enhanced = torch.cat((x_Block1_32x32_enhanced,x_Block2_32x32_enhanced,x_Block3_32x32_enhanced), dim=1)    # x [128*3, 32, 32]
        
        #pdb.set_trace()
        
        x = self.lastconv1(x_concat_enhanced)    # x [128, 32, 32] 
        x = self.lastconv2(x)    # x [64, 32, 32] 
        x = self.lastconv3(x)    # x [1, 32, 32] 
        
        map_x = x.squeeze(1)
        
        return map_x , x_concat#, x_Block1, x_Block2, x_Block3, x_input




class BATT_CDCN(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7,**kwarg):   
        super(BATT_CDCN, self).__init__()

        
        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             SEBlock(128) if se else Identity(),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             SEBlock(128) if se else Identity(),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),  
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             SEBlock(128) if se else Identity(),
            nn.BatchNorm2d(128),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.lastconv1 = nn.Sequential(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),    
        )
        
        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.lastconv3 = nn.Sequential(
            basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        
        self.sa1 = SpatialAttention(kernel = 7)
        self.sa2 = SpatialAttention(kernel = 5)
        self.sa3 = SpatialAttention(kernel = 3)

        self.sa1_cls = SpatialAttention(kernel = 7)
        self.sa2_cls = SpatialAttention(kernel = 5)
        self.sa3_cls = SpatialAttention(kernel = 3)
        
        
        self.se1 = SEBlock(128)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(128)

        self.se1_cls = SEBlock(128)
        self.se2_cls = SEBlock(128)
        self.se3_cls = SEBlock(128)
        
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   # x [64, 256, 256]
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        
        x_Block1_depth = self.se1(x_Block1)
        attention1 = self.sa1(x_Block1_depth)
        x_Block1_depth = attention1 * x_Block1_depth
        x_Block1_32x32_depth = self.downsample32x32(x_Block1_depth)   # x [128, 32, 32]  
        
        x_Block1_cls = self.se1_cls(x_Block1)
        attention1_cls = self.sa1_cls(x_Block1_cls)
        x_Block1_cls = attention1_cls * x_Block1_cls
        x_Block1_32x32_cls = self.downsample32x32(x_Block1_cls)   # x [128, 32, 32] 
        
        
        x_Block2 = self.Block2(x_Block1)	    	    	# x [128, 64, 64]
        
        x_Block2_depth = self.se2(x_Block2)
        attention2 = self.sa2(x_Block2_depth)
        x_Block2_depth = attention2 * x_Block2_depth
        x_Block2_32x32_depth = self.downsample32x32(x_Block2_depth)   # x [128, 32, 32] 
        
        x_Block2_cls = self.se2_cls(x_Block2)
        attention2_cls = self.sa2_cls(x_Block2_cls)
        x_Block2_cls = attention2_cls * x_Block2_cls
        x_Block2_32x32_cls = self.downsample32x32(x_Block2_cls)   # x [128, 32, 32] 
        
        
        
        x_Block3 = self.Block3(x_Block1)	    	    	# x [128, 64, 64]
        
        x_Block3_depth = self.se3(x_Block3)
        attention3 = self.sa3(x_Block3_depth)
        x_Block3_depth = attention3 * x_Block3_depth
        x_Block3_32x32_depth = self.downsample32x32(x_Block3_depth)   # x [128, 32, 32] 
        
        x_Block3_cls = self.se3_cls(x_Block3)
        attention3_cls = self.sa3_cls(x_Block3_cls)
        x_Block3_cls = attention3_cls * x_Block3_cls
        x_Block3_32x32_cls = self.downsample32x32(x_Block3_cls)   # x [128, 32, 32] 
        
        

        x_concat_depth = torch.cat((x_Block1_32x32_depth,x_Block2_32x32_depth,x_Block3_32x32_depth), dim=1)    # x [128*3, 32, 32]
        
        x_concat_cls = torch.cat((x_Block1_32x32_cls,x_Block2_32x32_cls,x_Block3_32x32_cls), dim=1)    # x [128*3, 32, 32]
        
        #pdb.set_trace()
        
        x = self.lastconv1(x_concat_depth)    # x [128, 32, 32] 
        x = self.lastconv2(x)    # x [64, 32, 32] 
        x = self.lastconv3(x)    # x [1, 32, 32] 
        
        map_x = x.squeeze(1)
        
        return map_x , x_concat_cls#, x_Block1, x_Block2, x_Block3, x_input




class BSATT_CDCN(ATT_CDCN):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7, se=True, sa=True, pos= 'first'):   
        super(BSATT_CDCN, self).__init__(basic_conv, theta, se,sa,pos = 'last')
        '''
        No attention for 
        same SE_SA for depth head and softmax head
        '''

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   # x [64, 256, 256]
        
        x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)   # x [128, 32, 32]  
        
        if self.se:
            x_Block1_SE = self.se1(x_Block1)
        else:
            x_Block1_SE = x_Block1
        if self.sa:
            attention1 = self.sa1(x_Block1_SE)
            x_Block1_SA = attention1 * x_Block1_SE
            x_Block1_32x32_enhanced = self.downsample32x32(x_Block1_SA)   # x [128, 32, 32]  
        else:
            x_Block1_32x32_enhanced = self.downsample32x32(x_Block1_SE)   # x [128, 32, 32]  
            x_Block1_SA = x_Block1_SE
        
        
        if self.pos == 'first':
            x_Block2 = self.Block2(x_Block1_SA)	    # x [128, 64, 64]	  
        else:
            x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	 
            
        x_Block2_32x32 = self.downsample32x32(x_Block2)
        
        if self.se:
            x_Block2_SE = self.se2(x_Block2)
        else:
            x_Block2_SE = x_Block2
        if self.sa:
            attention2 = self.sa2(x_Block2_SE)
            x_Block2_SA = attention2 * x_Block2_SE
            x_Block2_32x32_enhanced = self.downsample32x32(x_Block2_SA)   # x [128, 32, 32]  
        else:
            x_Block2_32x32_enhanced = self.downsample32x32(x_Block2_SE)   # x [128, 32, 32]  
            x_Block2_SA = x_Block2_SE
         
        
        if self.pos == 'first':
            x_Block3 = self.Block2(x_Block2_SA)	    # x [128, 32, 32]	  
        else:
            x_Block3 = self.Block2(x_Block2)	    # x [128, 32, 32]	    	
        
        x_Block3_32x32 = self.downsample32x32(x_Block3)
        if self.se:
            x_Block3_SE = self.se3(x_Block3)
        else:
            x_Block3_SE = x_Block3
        if self.sa:
            attention3 = self.sa3(x_Block3_SE)
            x_Block3_SA = attention3 * x_Block3_SE
            x_Block3_32x32_enhanced = self.downsample32x32(x_Block3_SA)   # x [128, 32, 32]  
        else:
            x_Block3_32x32_enhanced = self.downsample32x32(x_Block3_SE)   # x [128, 32, 32]  
        
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]
        
        x_concat_enhanced = torch.cat((x_Block1_32x32_enhanced,x_Block2_32x32_enhanced,x_Block3_32x32_enhanced), dim=1)    # x [128*3, 32, 32]
        
        #pdb.set_trace()
        
        x = self.lastconv1(x_concat_enhanced)    # x [128, 32, 32] 
        x = self.lastconv2(x)    # x [64, 32, 32] 
        x = self.lastconv3(x)    # x [1, 32, 32] 
        
        map_x = x.squeeze(1)
        
        return map_x , x_concat_enhanced#, x_Block1, x_Block2, x_Block3, x_input




## ATT_CDCN
# class ATT_CDCN(nn.Module):

#     def __init__(self, basic_conv=Conv2d_cd, theta=0.7, se=False, sa=False, pos= 'first'):   
#         super(ATT_CDCN, self).__init__()
#         assert pos in ['first','last']
#         self.se= se
#         self.sa= sa
#         self.pos = pos
        
#         self.conv1 = nn.Sequential(
#             basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),    
#         )
        
#         self.Block1 = nn.Sequential(
#             basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),   
#             basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.BatchNorm2d(196),
#             nn.ReLU(),  
#             basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
# #             SEBlock(128) if se else Identity(),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),   
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
#         )
        
#         self.Block2 = nn.Sequential(
#             basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),   
#             basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.BatchNorm2d(196),
#             nn.ReLU(),  
#             basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
# #             SEBlock(128) if se else Identity(),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),  
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )
        
#         self.Block3 = nn.Sequential(
#             basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),   
#             basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.BatchNorm2d(196),
#             nn.ReLU(),  
#             basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
# #             SEBlock(128) if se else Identity(),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),   
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         )
        
#         self.lastconv1 = nn.Sequential(
#             basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),    
#         )
        
#         self.lastconv2 = nn.Sequential(
#             basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),    
#         )
        
#         self.lastconv3 = nn.Sequential(
#             basic_conv(64, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
#             nn.ReLU(),    
#         )
#         if self.sa:
#             self.sa1 = SpatialAttention(kernel = 7)
#             self.sa2 = SpatialAttention(kernel = 5)
#             self.sa3 = SpatialAttention(kernel = 3)
        
#         if self.se:
#             self.se1 = SEBlock(128)
#             self.se2 = SEBlock(128)
#             self.se3 = SEBlock(128)
        
#         self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
#     def forward(self, x):	    	# x [3, 256, 256]
        
#         x_input = x
#         x = self.conv1(x)		   # [64, 256, 256]
        
#         x_Block1 = self.Block1(x)	    	    	# x [128, 128, 128]
#         if self.se:
#             x_Block1_SE = self.se1(x_Block1)
#         else:
#             x_Block1_SE = x_Block1
#         if self.sa:
#             attention1 = self.sa1(x_Block1_SE)
#             x_Block1_SA = attention1 * x_Block1_SE
#             x_Block1_32x32 = self.downsample32x32(x_Block1_SA)   # x [128, 32, 32]  
#         else:
#             x_Block1_32x32 = self.downsample32x32(x_Block1_SE)   # x [128, 32, 32]  
#             x_Block1_SA = x_Block1_SE
        
#         if self.pos == 'first':
#             x_Block2 = self.Block2(x_Block1_SA)	    # x [128, 64, 64]	  
#         else:
#             x_Block2 = self.Block2(x_Block1)	    # x [128, 64, 64]	  
#         if self.se:
#             x_Block2_SE = self.se2(x_Block2)
#         else:
#             x_Block2_SE = x_Block2
#         if self.sa:
#             attention2 = self.sa2(x_Block2_SE)
#             x_Block2_SA = attention2 * x_Block2_SE
#             x_Block2_32x32 = self.downsample32x32(x_Block2_SA)   # x [128, 32, 32]  
#         else:
#             x_Block2_32x32 = self.downsample32x32(x_Block2_SE)   # x [128, 32, 32]  
#             x_Block2_SA = x_Block2_SE
         
        
#         if self.pos == 'first':
#             x_Block3 = self.Block2(x_Block2_SA)	    # x [128, 32, 32]	  
#         else:
#             x_Block3 = self.Block2(x_Block2)	    # x [128, 32, 32]	    	
            
#         if self.se:
#             x_Block3_SE = self.se3(x_Block3)
#         else:
#             x_Block3_SE = x_Block3
#         if self.sa:
#             attention3 = self.sa3(x_Block3_SE)
#             x_Block3_SA = attention3 * x_Block3_SE
#             x_Block3_32x32 = self.downsample32x32(x_Block3_SA)   # x [128, 32, 32]  
#         else:
#             x_Block3_32x32 = self.downsample32x32(x_Block3_SE)   # x [128, 32, 32]  
        
        
#         x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    # x [128*3, 32, 32]  
        
#         #pdb.set_trace()
        
#         x = self.lastconv1(x_concat)    # x [128, 32, 32] 
#         x = self.lastconv2(x)    # x [64, 32, 32] 
#         x = self.lastconv3(x)    # x [1, 32, 32] 
        
#         map_x = x.squeeze(1)
        
#         return map_x , x_concat#, x_Block1, x_Block2, x_Block3, x_input