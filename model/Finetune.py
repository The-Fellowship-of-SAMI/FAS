import torch
import torch.nn as nn
from model.C_CDN import C_CDN, DC_CDN


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