import numpy as np
import torch 
from torchvision import transforms
import torch.nn as nn
from torch.utils.data  import Dataset, DataLoader
import torch.optim as optim
from model.utils import Contrast_depth_loss
from model.C_CDN import C_CDN, DC_CDN
from model. CDCN import CDCN
import torchmetrics
import pytorch_lightning as pl




class pl_train(pl.LightningModule):
    def __init__(self, model : C_CDN, runs = None, ckpt_out = None, train ="all", **kwarg):
        super().__init__()
        assert train in ['all','depth','classifier','none'], "Training mode not supported, please select in the list ['all','depth','classifier','none']"
        if kwarg['lr'] is not None:
            self.lr = kwarg['lr']
        else:
            self.lr = .0001
        
        if kwarg['wd'] is not None:
            self.wd = kwarg['wd']
        else:
            self.wd = .00005

        self.net = model.to(self.device)
        self.cls = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                nn.Flatten(1),
                                nn.Linear(16*16,1),
                                nn.Sigmoid()).to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.contrastive_loss =  Contrast_depth_loss().to(self.device)
        self.BCE_loss = nn.BCELoss().to(self.device)
        self.run = runs
        self.ckpt = ckpt_out
        # self.device_t = device
        # self.theta = torch.nn.Parameter(torch.zeros([3,1]),requires_grad= True if train != 'none' else False).to(self.device)

        self.sigmoid = torch.sigmoid
        
        if train == 'all':
            for param in self.parameters():
                param.requires_grad = True
        elif train == 'classifier':
            for param in self.net.parameters():
                param.requires_grad = False
        elif train == 'depth':
            for param in self.cls.parameters():
                param.requires_grad = False
        elif train == 'none':
            for param in self.parameters():
                param.requires_grad = False
            
        
    def forward(self, x):
        map_x = self.net(x)
        score_x = self.cls(map_x)
        return map_x, score_x

    # def on_train_epoch_start(self) -> None:
    #     self.score = torch.Tensor().to(self.device)
    #     self.label = torch.Tensor().to(self.device)
    #     return 
    
    def training_step(self, batch, batch_idx):
        inputs, map_label, spoof_label = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].float().view(-1,1).to(self.device)
        map_x, score_x =  self(inputs)
        # loss = self.sigmoid(self.theta[0])*self.MSE_loss(map_label,map_x) + self.sigmoid(self.theta[1])*self.contrastive_loss(map_label,map_x) + self.sigmoid(self.theta[2])*self.BCE_loss(score_x, spoof_label)
        loss = self.MSE_loss(map_label,map_x) + self.contrastive_loss(map_label,map_x) + self.BCE_loss(score_x, spoof_label)
        loss = loss.mean()
        # self.score = torch.cat([self.score, loss], dim = 1)
        # self.label = torch.cat([self.label, spoof_label], dim = 1)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run["train/loss"].log(loss)
        return {'loss': loss, 'pred': score_x, 'target': spoof_label}
        
    def training_epoch_end(self, training_step_outputs):
        if self.run is not None:
            acc = torchmetrics.Accuracy()(training_step_outputs['pred'],training_step_outputs['target'])
            loss = torch.mean(training_step_outputs['loss'])
            self.run['train/acc'].upload(acc)
            self.run['train/epoch_loss'].upload(loss)
        if self.ckpt is not None:
            torch.save(self.net.state_dict(), self.ckpt)
            if self.run is not None:
                self.run["model"].upload(self.ckpt)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr= self.lr, weight_decay = self.wd)

    # def on_validation_epoch_start(self) -> None:
    #     self.score = torch.Tensor().to(self.device)
    #     self.label = torch.Tensor().to(self.device)
    #     return 
    
    def validation_step(self, batch, batch_idx):
        inputs, map_label, spoof_label = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].float().view(-1,1).to(self.device)
        map_x, score_x =  self(inputs)
        # loss = self.sigmoid(self.theta[0])*self.MSE_loss(map_label,map_x) + self.sigmoid(self.theta[1])*self.contrastive_loss(map_label,map_x) + self.sigmoid(self.theta[2])*self.BCE_loss(score_x, spoof_label)
        loss = self.MSE_loss(map_label,map_x) + self.contrastive_loss(map_label,map_x) + self.BCE_loss(score_x, spoof_label)
        loss = loss.mean()
        # self.score = torch.cat([self.score, loss], dim = 1)
        # self.label = torch.cat([self.label, spoof_label], dim = 1)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run["val/loss"].log(loss)
        
        return {'loss': loss, 'pred': score_x, 'target': spoof_label}
        
    def validation_epoch_end(self, outputs) -> None:
        if self.run is not None:
            acc = torchmetrics.Accuracy()(outputs['pred'],outputs['target'])
            loss = torch.mean(outputs['loss'])
            self.run['val/acc'].upload(acc)
            self.run['val/epoch_loss'].upload(loss)
        
        