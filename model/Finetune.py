import torch
import torch.nn as nn
from model.C_CDN import C_CDN, DC_CDN
from model.CDCNv2 import CDCNv2, Conv2d_X, SEBlock



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
                                nn.Linear(16*16,2),
                                nn.Softmax())
        
        if cls_weights is not None:
            self.cls.load_state_dict(torch.load(cls_weights), lambda s, l: s)
    
    def forward(self, x):
        depth, feat = self.net(x)
        return depth, self.cls(depth)

        
class Finetune_modelv2(nn.Module):
    def __init__(self, depth_model, depth_weights = None, cls_weights = None, weights = None, device = 'cuda') -> None:
        super().__init__()
        self.net = depth_model.to(device)
        

        sample = torch.rand(1,3,256,256).to(device) # Input image size
        _, feature = self.net(sample)
        [_,C_out,_,_] = feature.shape # C_out should be 384

        self.cls = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                 nn.Conv2d(384,3,kernel_size=3,padding= 1),
                                 nn.Flatten(1),
                                 nn.Linear(3*16*16,2),
                                 nn.Softmax()).to(device)
        if weights is not None:
            self.load_state_dict(torch.load(weights), lambda s, l: s)
        if depth_weights is not None:
            self.net.load_state_dict(torch.load(depth_weights), lambda s, l: s)
        if cls_weights is not None:
            self.cls.load_state_dict(torch.load(cls_weights), lambda s, l: s)
        

    def forward(self,x):
        depth, feature = self.net(x)
        out = self.cls(feature)
        return depth, out#, feature


class WrapperClassifier(nn.Module):
    '''
    This class only wrap around a model that produce a depth map and use the depth map to classify output. Inference only.
    '''
    def __init__(self, backbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.cls = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                nn.Flatten(1),
                                nn.Linear(16*16,2),
                                nn.Softmax())
        self.cls.load_state_dict(torch.load('./checkpoints/checkpoint_cls_sm.pth'), lambda s, l: s)

    def forward(self,x):
        depth, *sm = self.backbone(x)
        out = self.cls(depth)
        return depth, out#, feature