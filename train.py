import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from model.utils import CFASDDataset, CSDataset, ZaloDataset, CFASD_ZaloDataset
from torchvision import transforms
from train_model import pl_train
from model.C_CDN import C_CDN, DC_CDN
from model.CDCN import CDCN
import pytorch_lightning as pl
from argparse import ArgumentParser
import neptune.new as neptune
from tqdm import tqdm

transform = transforms.Compose([ transforms.RandomRotation(.15), transforms.RandomHorizontalFlip()])
map_data_to_dataset ={'sample': CSDataset, 'train_img': CFASDDataset, 'test_img': CFASDDataset, 'zalo_data': ZaloDataset}
map_input_to_model = {'C_CDN': C_CDN, 'DC_CDN': DC_CDN, 'CDCN': CDCN}
map_equalize_to_bool = {'true': True, 't': True, 'yes': True, 'y': True}

if __name__ == '__main__':
    parser = ArgumentParser(description= "Train the model using Pytorch Lightning.")
    parser.add_argument('--model', default= C_CDN, help= 'Specify what model to use.')
    parser.add_argument('--train_data', nargs= '+', default='./data/processed/train_img', help= 'Specify the path to the training folder.')
    parser.add_argument('--val_data', required= False, help= 'Specify the path to the validation folder.')
    parser.add_argument('--device', default= 'cuda', help= 'Specify what hardware to use for training.')
    parser.add_argument('--lr', default= .0001, type= float, help= 'Specify learning rate for the optimizer.')
    parser.add_argument('--wd', default= .00005, type= float, help= 'Specify the weights decay for the optimizer.')
    parser.add_argument('--batch_size', default= 2, type= int, help= 'Define the batch size for the training data.')
    parser.add_argument('--val_batch_size', default= 4,required= False, type= int, help= 'Define the batch size for the validation data.')
    parser.add_argument('--max_epochs', default= 50, type= int, help= 'Specify how many epochs used in training.')
    parser.add_argument('--equalize_data', default= True, help= 'Decide to equalize the class distribution of the data or not. Take longer time to prepare.')
    parser.add_argument('--checkpoint_in', required= False, help= 'Pass the previous checkpoint to the model for transfer learning.')
    parser.add_argument('--checkpoint_out', required= False, help= 'Save the trained model .')
    parser.add_argument('--neptune', required= False, help= 'Use neptune-ai.')
    # parser.add_argument()
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model = map_input_to_model[args.model] 
    sampler = None
    shuffle = True
    run = None

    # print(args.train_data)
    if isinstance(args.train_data, list):
        data_path =''
        for path in args.train_data:
            for key in map_data_to_dataset.keys():
                if key in path:
                    data_path += f'{key}+'
        # print(data_path)
        if data_path == 'train_img+zalo_data+' or data_path == 'zalo_data+train_img+':
            train_dataset = CFASD_ZaloDataset('./data/processed/train_img', './data/processed/zalo_data', transform= transform)
    else:
        for key in map_data_to_dataset.keys():
            if key in args.train_data:
                dataset = map_data_to_dataset[key]
                train_dataset = dataset(args.train_data, transform= transform)

    if  map_equalize_to_bool.get(str(args.equalize_data).lower(), False):
        n_class = {}
        cls_weight = {}
        ds_weight = []
        for i in tqdm(range(len(train_dataset))):
            if str(train_dataset[i][2]) in n_class.keys():
                n_class[str(train_dataset[i][2])] += 1
            else:
                n_class[str(train_dataset[i][2])] = 0
        for i in n_class:
            cls_weight[i] = 1/(n_class[i]+1)

        ds_weight = [cls_weight[str(train_dataset[i][2])] for i in range(len(train_dataset)) ]
        sampler = WeightedRandomSampler(ds_weight,len(train_dataset))
        shuffle = False

    train_loader = DataLoader(train_dataset, sampler= sampler, batch_size= args.batch_size, shuffle= shuffle)

    if args.val_data is not None:
        val_sampler = None
        test_dataset = CFASDDataset(args.val_data, mode = 'val', transform= None)
        
        if map_equalize_to_bool.get(str(args.equalize_data).lower(), False):
            n_class = {}
            cls_weight = {}
            ds_weight = []
            for i in tqdm(range(len(test_dataset))):
                if str(test_dataset[i][2]) in n_class.keys():
                    n_class[str(test_dataset[i][2])] += 1
                else:
                    n_class[str(test_dataset[i][2])] = 0
            for i in n_class:
                cls_weight[i] = 1/(n_class[i]+1)

            ds_weight = [cls_weight[str(test_dataset[i][2])] for i in range(len(test_dataset)) ]
            val_sampler = WeightedRandomSampler(ds_weight,len(test_dataset))
        
        val_loader = DataLoader(test_dataset, sampler= val_sampler, batch_size= args.val_batch_size, shuffle= False)

    
    if args.neptune is not None:
        if args.neptune == 'new':
            run = neptune.init_run(project="minhnguyen/FAS",
                                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNThiYzVhMy0xYzM4LTQ5ZmItOWFkZC00YmIzODljNzM1MmUifQ==",)
        else:
            try:
                run = neptune.init_run(project="minhnguyen/FAS",
                                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwNThiYzVhMy0xYzM4LTQ5ZmItOWFkZC00YmIzODljNzM1MmUifQ==",
                                with_id=args.neptune,)
            except Exception as e:
                print(e)
                exit()
    


    train_model = pl_train(model = model(),runs = run,ckpt_out= args.checkpoint_out, lr = args.lr, wd = args.wd).to(device)
    trainer = pl.Trainer(devices=1, accelerator="gpu",accumulate_grad_batches=1, max_epochs = args.max_epochs)
    if args.val_data is not None:
        trainer.fit(train_model, train_loader, val_loader)
    else:
        trainer.fit(train_model, train_loader)