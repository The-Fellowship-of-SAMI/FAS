import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from model.utils import Contrast_depth_loss
from model.C_CDN import C_CDN, DC_CDN
from model.CDCN import CDCN
from model.Finetune import Finetune_modelv2
import torchmetrics
import pytorch_lightning as pl




class pl_train(pl.LightningModule):
    def __init__(self, model : C_CDN, runs = None, ckpt_out = None, train ="all", **kwarg):
        super().__init__()
        assert train in ['all','depth','classifier','none'], "Training mode not supported, please select in the list ['all','depth','classifier','none']"
        if 'lr' in kwarg.keys():
            self.lr = kwarg['lr']
        else:
            self.lr = .0001
        
        if 'wd' in kwarg.keys():
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

        self.sigmoid = torch.sigmoid
        self.cls_train = 1
        self.depth_train = 1
        if train == 'all':
            for param in self.parameters():
                param.requires_grad = True
        elif train == 'classifier':
            for param in self.net.parameters():
                param.requires_grad = False
                self.depth_train = 0
        elif train == 'depth':
            for param in self.cls.parameters():
                param.requires_grad = False
                self.cls_train = 0
        elif train == 'none':
            for param in self.parameters():
                param.requires_grad = False
        
        self.accuracy = torchmetrics.Accuracy(threshold= .7, task= 'binary')
        self.f1 = torchmetrics.F1Score(num_classes= 2, threshold= .7)
        


    def forward(self, x):
        
        map_x, *feat = self.net(x)
        score_x = self.cls(map_x)
        return map_x, score_x, feat
    
    def training_step(self, batch, batch_idx):
        inputs, map_label, spoof_label = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].float().view(-1,1).to(self.device)
        map_x, score_x, *_ =  self(inputs)
        loss = self.depth_train*(self.MSE_loss(map_label,map_x) + self.contrastive_loss(map_label,map_x) ) + self.cls_train*self.BCE_loss(score_x, spoof_label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.accuracy(score_x, spoof_label), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run["train/loss"].log(loss)
        return {'loss': loss, 'pred': score_x, 'target': spoof_label}
        
    def training_epoch_end(self, training_step_outputs):
        if self.run is not None:
            acc = torchmetrics.Accuracy()(training_step_outputs['pred'],training_step_outputs['target'])
            loss = torch.mean(training_step_outputs['loss'])
            self.run['train/epoch_acc'].upload(acc)
            self.run['train/epoch_loss'].upload(loss)
        if self.ckpt is not None:
            torch.save(self.net.state_dict(), self.ckpt)
            if self.run is not None:
                self.run["model"].upload(self.ckpt)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr= self.lr, weight_decay = self.wd)
    
    def validation_step(self, batch, batch_idx):
        inputs, map_label, spoof_label = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].float().view(-1,1).to(self.device)
        map_x, score_x, *_ =  self(inputs)
        loss = self.depth_train*(self.MSE_loss(map_label,map_x) + self.contrastive_loss(map_label,map_x) ) + self.cls_train*self.BCE_loss(score_x, spoof_label)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.accuracy(score_x, spoof_label), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run["val/loss"].log(loss)
        
        return {'loss': loss, 'pred': score_x, 'target': spoof_label}
        
    def validation_epoch_end(self, outputs) -> None:
        if self.run is not None:
            acc = self.accuracy(outputs['pred'],outputs['target'])
            f1 = self.f1(outputs['pred'],outputs['target'])
            loss = torch.mean(outputs['loss'])
            self.run['val/epoch_acc'].upload(acc)
            self.run['val/epoch_loss'].upload(loss)
            self.run['val/epoch_f1'].upload(f1)
        
class pl_trainv2(pl.LightningModule):
    def __init__(self, model : Finetune_modelv2, runs = None, ckpt_out = None, train ="all", **kwarg):
        super().__init__()
        assert train in ['all','depth','classifier','none'], "Training mode not supported, please select in the list ['all','depth','classifier','none']"
        if 'lr' in kwarg.keys():
            self.lr = kwarg['lr']
        else:
            self.lr = .0001
        
        if 'wd' in kwarg.keys():
            self.wd = kwarg['wd']
        else:
            self.wd = .00005

        self.model = model.to(self.device)

        self.MSE_loss = nn.MSELoss().to(self.device)
        self.contrastive_loss =  Contrast_depth_loss().to(self.device)
        self.BCE_loss = nn.BCELoss().to(self.device)
        self.run = runs
        self.ckpt = ckpt_out

        self.sigmoid = torch.sigmoid
        self.cls_train = 1
        self.depth_train = 1
        if train == 'all':
            for param in self.parameters():
                param.requires_grad = True
        elif train == 'classifier':
            for param in self.model.net.parameters():
                param.requires_grad = False
                self.depth_train = 0
        elif train == 'depth':
            for param in self.model.cls.parameters():
                param.requires_grad = False
                self.cls_train = 0
        elif train == 'none':
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.accuracy = torchmetrics.Accuracy(threshold= .7, task= 'binary')
        self.f1 = torchmetrics.F1Score(num_classes= 2, threshold= .7)
        


    def forward(self, x):
        
        map_x,score_x, *feat = self.model(x)
        return map_x, score_x, feat
    
    def training_step(self, batch, batch_idx):
        inputs, map_label, spoof_label = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].float().view(-1,1).to(self.device)
        map_x, score_x, *_ =  self(inputs)
        loss = self.depth_train*(self.MSE_loss(map_label,map_x) + self.contrastive_loss(map_label,map_x) ) + self.cls_train*self.BCE_loss(score_x, spoof_label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.accuracy(score_x, spoof_label), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run["train/loss"].log(loss)
        return {'loss': loss, 'pred': score_x, 'target': spoof_label}
        
    def training_epoch_end(self, training_step_outputs):
        if self.run is not None:
            acc = torchmetrics.Accuracy()(training_step_outputs['pred'],training_step_outputs['target'])
            loss = torch.mean(training_step_outputs['loss'])
            self.run['train/epoch_acc'].upload(acc)
            self.run['train/epoch_loss'].upload(loss)
        if self.ckpt is not None:
            torch.save(self.model.state_dict(), self.ckpt)
            if self.run is not None:
                self.run["model"].upload(self.ckpt)
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr= self.lr, weight_decay = self.wd)
    
    def validation_step(self, batch, batch_idx):
        inputs, map_label, spoof_label = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].float().view(-1,1).to(self.device)
        map_x, score_x, *_ =  self(inputs)
        loss = self.depth_train*(self.MSE_loss(map_label,map_x) + self.contrastive_loss(map_label,map_x) ) + self.cls_train*self.BCE_loss(score_x, spoof_label)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.accuracy(score_x, spoof_label), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run["val/loss"].log(loss)
        
        return {'loss': loss, 'pred': score_x, 'target': spoof_label}
        
    def validation_epoch_end(self, outputs) -> None:
        if self.run is not None:
            acc = self.accuracy(outputs['pred'],outputs['target'])
            f1 = self.f1(outputs['pred'],outputs['target'])
            loss = torch.mean(outputs['loss'])
            self.run['val/epoch_acc'].upload(acc)
            self.run['val/epoch_loss'].upload(loss)
            self.run['val/epoch_f1'].upload(f1)



