import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2


def contrast_depth_conv(input, device='cuda'):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0],
                                             [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]
         ], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1,
                                                         0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(
        kernel_filter.astype(np.float)).float().to(device)
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)

    input = input.unsqueeze(dim=1).expand(
        input.shape[0], 8, input.shape[1], input.shape[2])

    contrast_depth = F.conv2d(
        input, weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth


# Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
class Contrast_depth_loss(nn.Module):
    def __init__(self,device='cuda'):
        super(Contrast_depth_loss, self).__init__()
        self.device= device

    def forward(self, out, label,):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''

        contrast_out = contrast_depth_conv(out,device=self.device)
        
        contrast_label = contrast_depth_conv(label,device=self.device)

        criterion_MSE = nn.MSELoss().to(self.device)

        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)

        return loss


class CSDataset(Dataset):
    def __init__(self, root_dir='./sample',  transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess()

    def __len__(self):
        with open('./Imfile.txt', 'r+') as f:
            len_ds = len(f.readlines())
        return len_ds-1

    def __getitem__(self, idx):
        with open('./Imfile.txt', 'r+') as f:
            lines = f.readlines()

        dir, filename = lines[idx].strip().split(" ")
        # print(dir+'/color'+filename)

        rgb_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/color'+filename)[:, :, ::-1], (256, 256))).permute(2, 0, 1)/255
        depth_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/depth'+filename, 0), (256, 256))).unsqueeze(-1).permute(2, 0, 1)/255

        label = 0 if 'fake' in dir else 1

        sample = torch.cat((rgb_im, depth_im), dim=0)

        if self.transform is not None:
            sample = self.transform(sample)

        return (sample[:3], torch.Tensor(cv2.resize(sample[3].numpy(), (32, 32))), label, dir+filename)

    def preprocess(self):
        with open(f'./Imfile.txt', 'w+') as f:
            # self.data = {'color':[] , 'depth':[]}
            for class_name in os.listdir(self.root_dir):
                dir = self.root_dir + f'/{class_name}'
                for batch_name in os.listdir(dir):
                    dir2 = dir + f'/{batch_name}'
                    for name in os.listdir(dir2):

                        dir3 = dir2 + f'/{name}'
                        for imtype in ['color']:
                            dir4 = dir3 + f'/{imtype}'

                            for imfile in os.listdir(dir4):
                                # dir5 = dir4 + f'/{imfile}'
                                dir5 = dir3 + f' /{imfile}'
                                # print(torch.Tensor(cv2.resize(cv2.imread(dir5)[:,:,::-1], (256, 256) )) )
                                # self.data[imtype].append((  np.moveaxis(cv2.resize(cv2.imread(dir5)[:,:,::-1], (256, 256) if imtype=='color' else (32,32) ), -1, 0)   ,f'{class_name}_{batch_name}_{name}_{imfile}'))
                                f.writelines(dir5+'\n')


class CFASDDataset(Dataset):
    def __init__(self, root_dir='./train_img', mode='train',  transform=None, preload=True):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.preprocess()
        if preload:
            with open(f'./CFASD_{self.mode}_Imfile.txt', 'r+') as f:
                lines = f.readlines()
                self.lines = [path.strip().split(" ") for path in lines]
            os.remove(f'./CFASD_{self.mode}_Imfile.txt')
        else:
            self.lines = None

    def __len__(self):
        if self.lines is not None:
            return len(self.lines)

        with open(f'./CFASD_{self.mode}_Imfile.txt', 'r+') as f:
            len_ds = len(f.readlines())
        return len_ds-1

    def __getitem__(self, idx):
        if self.lines is not None:
            dir, filename = self.lines[idx][0], self.lines[idx][1]
        else:
            with open(f'./CFASD_{self.mode}_Imfile.txt', 'r+') as f:
                lines = f.readlines()
            dir, filename = lines[idx].strip().split(" ")

        rgb_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/color'+filename)[:, :, ::-1], (256, 256))).permute(2, 0, 1)/255
        depth_im = torch.Tensor(cv2.imread(
            dir+'/depth'+filename, 0)).unsqueeze(-1).permute(2, 0, 1)/255

        label = 0 if 'fake' in dir+filename else 1

        sample = torch.cat((rgb_im, depth_im), dim=0)

        if self.transform is not None:
            sample = self.transform(sample)
            sample[:3] = transforms.ColorJitter(brightness=.2)(sample[:3])

        return (sample[:3], torch.Tensor(cv2.resize(sample[3].numpy(), (32, 32))), label, dir+filename)

    def preprocess(self):
        with open(f'./CFASD_{self.mode}_Imfile.txt', 'w+')as f:
            # self.data = {'color':[] , 'depth':[]}
            dir = self.root_dir + r'/color'
            for filename in os.listdir(dir):
                f.writelines(self.root_dir + f' /{filename}'+'\n')


class ZaloDataset(Dataset):
    def __init__(self, root_dir='./zalo_data',  transform=None, preload=True):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess()
        if preload:
            with open(f'./ZaloImfile.txt', 'r+') as f:
                lines = f.readlines()
                self.lines = [path.strip().split(" ") for path in lines]
            os.remove(f'./ZaloImfile.txt')
        else:
            self.lines = None

    def __len__(self):
        if self.lines is not None:
            return len(self.lines)
        with open(f'./ZaloImfile.txt', 'r+') as f:
            len_ds = len(f.readlines())
        return len_ds-1

    def __getitem__(self, idx):
        if self.lines is not None:
            dir, filename = self.lines[idx][0], self.lines[idx][1]
        else:
            with open(f'./ZaloImfile.txt', 'r+') as f:
                lines = f.readlines()
            dir, filename = lines[idx].strip().split(" ")
        # print(dir+'/color'+filename)

        rgb_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/color'+filename)[:, :, ::-1], (256, 256))).permute(2, 0, 1)/255
        depth_im = torch.Tensor(cv2.imread(
            dir+'/depth'+filename, 0)).unsqueeze(-1).permute(2, 0, 1)/255

        label = 0 if 'fake' in dir+filename else 1

        sample = torch.cat((rgb_im, depth_im), dim=0)

        if self.transform is not None:
            sample = self.transform(sample)
            sample[:3] = transforms.ColorJitter(brightness=.2)(sample[:3])

        return (sample[:3], torch.Tensor(cv2.resize(sample[3].numpy(), (32, 32))), label, dir+filename)

    def preprocess(self):
        with open(f'./ZaloImfile.txt', 'w+') as f:
            # self.data = {'color':[] , 'depth':[]}
            dir = self.root_dir + r'/color'
            for filename in os.listdir(dir):
                f.writelines(self.root_dir + f' /{filename}'+'\n')


class CFASD_ZaloDataset(Dataset):
    def __init__(self, root_dirCSFASD='./train_img', root_dirZalo='./train',  transform=None):
        self.root_dir1 = root_dirCSFASD
        self.root_dir2 = root_dirZalo
        self.transform = transform
        self.preprocess()

    def __len__(self):
        with open(f'./CFASD_Zalo_Imfile.txt', 'r+') as f:
            len_ds = len(f.readlines())
        return len_ds-1

    def __getitem__(self, idx):
        with open(f'./CFASD_Zalo_Imfile.txt', 'r+') as f:
            lines = f.readlines()

        dir, filename = lines[idx].strip().split(" ")
        # print(dir+'/color'+filename)

        rgb_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/color'+filename)[:, :, ::-1], (256, 256))).permute(2, 0, 1)/255
        depth_im = torch.Tensor(cv2.imread(
            dir+'/depth'+filename, 0)).unsqueeze(-1).permute(2, 0, 1)/255

        label = 0 if 'fake' in dir+filename else 1

        sample = torch.cat((rgb_im, depth_im), dim=0)

        if self.transform is not None:
            sample = self.transform(sample)
            sample[:3] = transforms.ColorJitter(brightness=.2)(sample[:3])

        return (sample[:3], torch.Tensor(cv2.resize(sample[3].numpy(), (32, 32))), label, dir+filename)

    def preprocess(self):
        with open(f'./CFASD_Zalo_Imfile.txt', 'w+') as f:
            # self.data = {'color':[] , 'depth':[]}
            dir1 = self.root_dir1 + r'/color'
            for filename in os.listdir(dir1):
                f.writelines(self.root_dir1 + f' /{filename}'+'\n')

            dir2 = self.root_dir2 + r'/color'
            for filename in os.listdir(dir2):
                f.writelines(self.root_dir2 + f' /{filename}'+'\n')


class NUAADataset(Dataset):
    def __init__(self, root_dir='./data/processed/NUAA/train',  transform=None, preload=True, **kwarg):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess()
        if preload:
            with open(f'./NUAAImfile.txt', 'r+') as f:
                lines = f.readlines()
                self.lines = [path.strip().split(" ") for path in lines]
            os.remove(f'./NUAAImfile.txt')
        else:
            self.lines = None

    def __len__(self):
        if self.lines is not None:
            return len(self.lines)

        with open(f'./NUAAImfile.txt', 'r+') as f:
            len_ds = len(f.readlines())
        return len_ds-1

    def __getitem__(self, idx):
        if self.lines is not None:
            dir, filename = self.lines[idx][0], self.lines[idx][1]
        else:
            with open(f'./NUAAImfile.txt', 'r+') as f:
                lines = f.readlines()
            dir, filename = lines[idx].strip().split(" ")
        # print(dir+'/color'+filename)

        rgb_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/color'+filename)[:, :, ::-1], (256, 256))).permute(2, 0, 1)/255
        depth_im = torch.Tensor(cv2.resize(cv2.imread(dir+'/depth'+filename, 0),
                                (256, 256), interpolation=cv2.INTER_CUBIC)).unsqueeze(-1).permute(2, 0, 1)/255

        label = 0 if 'fake' in dir+filename else 1

        sample = torch.cat((rgb_im, depth_im), dim=0)

        if self.transform is not None:
            sample = self.transform(sample)
            # sample[:3] = transforms.ColorJitter(brightness=.2)(sample[:3])
            sample[:3] = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(sample[:3])
        return (sample[:3], torch.Tensor(cv2.resize(sample[3].numpy(), (32, 32))), label, dir+filename)

    def preprocess(self):
        with open(f'./NUAAImfile.txt', 'w+') as f:
            # self.data = {'color':[] , 'depth':[]}
            dir = self.root_dir + r'/color'
            for filename in os.listdir(dir):
                f.writelines(self.root_dir + f' /{filename}'+'\n')


class StandardDataset(Dataset):
    def __init__(self, root_dir='',  transform=None, preload=True, **kwarg):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess()
        if preload:
            with open(f'./{self.root_dir.upper()}Imfile.txt', 'r+') as f:
                lines = f.readlines()
                self.lines = [path.strip().split(" ") for path in lines]
            os.remove(f'./{self.root_dir.upper()}Imfile.txt')
        else:
            self.lines = None

    def __len__(self):
        if self.lines is not None:
            return len(self.lines)

        with open(f'./{self.root_dir.upper()}Imfile.txt', 'r+') as f:
            len_ds = len(f.readlines())
        return len_ds-1

    def __getitem__(self, idx):
        if self.lines is not None:
            dir, filename = self.lines[idx][0], self.lines[idx][1]
        else:
            with open(f'./{self.root_dir.upper()}Imfile.txt', 'r+') as f:
                lines = f.readlines()
            dir, filename = lines[idx].strip().split(" ")

        rgb_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/color'+filename)[:, :, ::-1], (256, 256))).permute(2, 0, 1)/255
        depth_im = torch.Tensor(cv2.resize(cv2.imread(dir+'/depth'+filename, 0),
                                (256, 256), interpolation=cv2.INTER_CUBIC)).unsqueeze(-1).permute(2, 0, 1)/255

        label = 0 if 'fake' in dir+filename else 1

        sample = torch.cat((rgb_im, depth_im), dim=0)

        if self.transform is not None:
            sample = self.transform(sample)
            # sample[:3] = transforms.ColorJitter(brightness=.2)(sample[:3])
            sample[:3] = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(sample[:3])
        return (sample[:3], torch.Tensor(cv2.resize(sample[3].numpy(), (32, 32))), label, dir+filename)

    def preprocess(self):
        with open(f'./{self.root_dir.upper()}Imfile.txt', 'w+') as f:
            # self.data = {'color':[] , 'depth':[]}
            dir = self.root_dir + r'/color'
            for filename in os.listdir(dir):
                f.writelines(self.root_dir + f' /{filename}'+'\n')


class CombinedDataset(Dataset):
    def __init__(self, root_dir:list[str]= [''],  transform=None, preload=True, **kwarg):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess()
        if preload:
            with open(f'./{self.filename.upper()}Imfile.txt', 'r+') as f:
                lines = f.readlines()
                self.lines = [path.strip().split(" ") for path in lines]
            os.remove(f'./{self.filename.upper()}Imfile.txt')
        else:
            self.lines = None

    def __len__(self):
        if self.lines is not None:
            return len(self.lines)

        with open(f'./{self.filename.upper()}Imfile.txt', 'r+') as f:
            len_ds = len(f.readlines())
            return len_ds-1

    def __getitem__(self, idx):
        if self.lines is not None:
            dir, filename = self.lines[idx][0], self.lines[idx][1]
        else:
            with open(f'./{self.filename.upper()}Imfile.txt', 'r+') as f:
                lines = f.readlines()
            dir, filename = lines[idx].strip().split(" ")

        rgb_im = torch.Tensor(cv2.resize(cv2.imread(
            dir+'/color'+filename)[:, :, ::-1], (256, 256))).permute(2, 0, 1)/255
        depth_im = torch.Tensor(cv2.resize(cv2.imread(dir+'/depth'+filename, 0),
                                (256, 256), interpolation=cv2.INTER_CUBIC)).unsqueeze(-1).permute(2, 0, 1)/255

        label = 0 if 'fake' in dir+filename else 1

        sample = torch.cat((rgb_im, depth_im), dim=0)

        if self.transform is not None:
            sample = self.transform(sample)
            # sample[:3] = transforms.ColorJitter(brightness=.2)(sample[:3])
            sample[:3] = transforms.ColorJitter(
                brightness=0.4, contrast=0.3, saturation=0.2, hue=0.1)(sample[:3])
        return (sample[:3], torch.Tensor(cv2.resize(sample[3].numpy(), (32, 32))), label, dir+filename)

    def preprocess(self):

        self.filename = ''
        for dir in self.root_dir:
            self.filename += dir.replace('/','_')
        with open(f'./{self.filename.upper()}Imfile.txt', 'w+') as f:
            for dir in self.root_dir:
                dir = dir + r'/color'
                for file in os.listdir(dir):
                    f.writelines(dir + f' /{file}'+'\n')
