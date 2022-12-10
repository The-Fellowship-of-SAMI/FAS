import torch
import torch.nn as nn
from model.C_CDN import C_CDN, DC_CDN
from model.CDNv2 import CDCNv2, Conv2d_X, SEBlock



class Concat_depth(nn.Module):
    def __init__(self, device = 'cuda') -> None:
        super().__init__()
        self.depth_map = torch.zeros(1,1,32,32).to(device)

    def forward(self,x):
        return torch.cat([self.depth_map,x], dim = 1)


class Finetune_model(nn.Module):
    def __init__(self, depth_model = DC_CDN(), depth_weights = None, cls_weights = './checkpoints/checkpoint_cls.pth'):
        super().__init__()
        self.net = depth_model
        if depth_weights is not None:
            self.net.load_state_dict(torch.load(depth_weights), lambda s, l: s)
        self.cls = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                nn.Flatten(1),
                                nn.Linear(16*16,1),
                                nn.Sigmoid())
        
        if cls_weights is not None:
            self.cls.load_state_dict(torch.load(cls_weights), lambda s, l: s)
    
    def forward(self, x):
        return self.cls(self.net(x))

        
class Finetune_modelv2(nn.Module):
    def __init__(self, depth_model = CDCNv2(), depth_weights = None, cls_weights = None, weights = None, device = 'cuda') -> None:
        super().__init__()
        self.depth = depth_model.to(device)
        

        sample = torch.rand(1,3,256,256).to(device) # Input image size
        _, feature = self.depth(sample)
        [_,C_out,_,_] = feature.shape # C_out should be 384

        self.cls = nn.Sequential(SEBlock(C_out),
                                 Conv2d_X(C_out,3,kernel_size=3, kernel_pos=[1 for i in range(9)], groups = 3),
                                 Concat_depth(),
                                 nn.Flatten(1),
                                 nn.Linear(4*32*32,1),
                                 nn.Sigmoid()).to(device)
        if weights is not None:
            self.load_state_dict(torch.load(weights), lambda s, l: s)
        if depth_weights is not None:
            self.depth.load_state_dict(torch.load(depth_weights), lambda s, l: s)
        if cls_weights is not None:
            self.cls.load_state_dict(torch.load(cls_weights), lambda s, l: s)
        

    def forward(self,x):
        depth, feature = self.depth(x)
        self.cls[2].depth_map = depth.unsqueeze(1)
        # return self.cls[3].depth_map
        out = self.cls(feature)
        return depth, out, feature


