import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.CDCN import CDCN, ATT_CDCN
import torchmetrics
import numpy as np



def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        
        
        criterion_MSE = nn.MSELoss().cuda()
    
        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)
    
        return loss




class pl_model(pl.LightningModule):
    def __init__(self, model : ATT_CDCN, runs = None, ckpt_out = None, train ="all", **kwarg):
        super(pl_model,self).__init__()
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
        self.cls = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                 nn.Conv2d(384,3,kernel_size=3,padding= 1),
                                 nn.Flatten(1),
                                 nn.Linear(3*16*16,2),
                                 nn.Softmax())
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.contrastive_loss =  Contrast_depth_loss()
        self.CE_loss = nn.CrossEntropyLoss().to(self.device)
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
        
        self.accuracy = torchmetrics.Accuracy(threshold= .5, task= 'binary')
        self.f1 = torchmetrics.F1Score(num_classes= 2, threshold= .5, task= 'binary')
        


    def forward(self, x):
        
        map_x, feat = self.net(x)
        score_x = self.cls(feat)
        return map_x, score_x, feat
    
    def training_step(self, batch, batch_idx):
        inputs, map_label, spoof_label = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].float().to(self.device)
        spoof_label = F.one_hot(spoof_label.to(torch.int64),2).float()
        map_x, score_x, *_ =  self(inputs)
        loss = 50*self.depth_train*(self.MSE_loss(map_label,map_x) + self.contrastive_loss(map_label,map_x) ) + self.cls_train*self.CE_loss(score_x, spoof_label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_acc", self.accuracy(score_x, spoof_label), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run["train/loss"].log(loss)
        return {'loss': loss, 'pred': score_x, 'target': spoof_label}
        
    def training_epoch_end(self, training_step_outputs):
        
        pred = torch.cat([x['pred'] for x in training_step_outputs],dim= 0)
        target = torch.cat([x['target'] for x in training_step_outputs],dim= 0)
        acc = self.accuracy((pred.T)[1],(target.T)[1])
        loss = torch.mean(torch.Tensor([x['loss'] for x in training_step_outputs]))
#         self.log("train_epoch_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run['train/epoch_acc'].log(acc)
            self.run['train/epoch_loss'].log(loss)
        if self.ckpt is not None:
            torch.save(self.state_dict(), self.ckpt)
            if self.run is not None:
                self.run["model/last"].upload(self.ckpt)
                
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr= self.lr, weight_decay = self.wd)
    
    def validation_step(self, batch, batch_idx):
        inputs, map_label, spoof_label = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].float().to(self.device)
        spoof_label = F.one_hot(spoof_label.to(torch.int64),2).float()
        map_x, score_x, _ =  self(inputs)
#         print(score_x.shape, spoof_label.shape)
        loss = 50*self.depth_train*(self.MSE_loss(map_label,map_x) + self.contrastive_loss(map_label,map_x) ) + self.cls_train*self.CE_loss(score_x, spoof_label)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log("val_acc", self.accuracy(score_x, spoof_label), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run["val/loss"].log(loss)
        
        return {'loss': loss, 'pred': score_x, 'target': spoof_label}
        
    def validation_epoch_end(self, outputs) -> None:
        # print(outputs[0]['pred'])
        
        # print([x['pred'] for x in outputs])
        pred = torch.cat([x['pred'] for x in outputs],dim= 0)
        target = torch.cat([x['target'] for x in outputs],dim= 0)
        acc = self.accuracy((pred.T)[1],(target.T)[1])
        f1 = self.f1((pred.T)[1],(target.T)[1])
        loss = torch.mean(torch.Tensor([x['loss'] for x in outputs]))
#         self.log("val_epoch_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        if self.run is not None:
            self.run['val/epoch_acc'].log(acc)
            self.run['val/epoch_loss'].log(loss)
            self.run['val/epoch_f1'].log(f1)