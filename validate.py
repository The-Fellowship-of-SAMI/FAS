import torch
import torch.nn.functional as F
import numpy as np
from model.C_CDN import C_CDN,DC_CDN
from model.CDCN import CDCN, SE_CDCN, ATT_CDCN
from model.Finetune import Finetune_model, Finetune_modelv2, WrapperClassifier
from torch.utils.data import DataLoader
from model.utils import CSDataset, CFASD_ZaloDataset, ZaloDataset, NUAADataset, StandardDataset, CFASDDataset
from torchmetrics.functional.classification import binary_confusion_matrix
from torchmetrics.functional.classification import binary_roc
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import pl_model

torch.manual_seed(42)

val_model = pl_model.load_from_checkpoint('checkpoints/checkpoint_att_cdcn_ca_sa_first_cls_best.ckpt',model = ATT_CDCN(theta= 0.7,se = True, sa = True, pos= 'first')).cuda()
# val_model = pl_model.load_from_checkpoint('checkpoints/checkpoint_att_cdcn_best.ckpt',model = ATT_CDCN(theta= 0.7,se = False, sa = False, pos= 'last')).cuda()
# val_model = WrapperClassifier(val_model).cuda()

# val_model = Finetune_model(depth_model= ATT_CDCN(se= True,sa= True), depth_weights= 'checkpoints/checkpoint_att_cdcn_se_sa_cfas.pth',cls_weights= 'checkpoints/checkpoint_cls_sm.pth').cuda()
# val_model = Finetune_model(depth_model= CDCN(theta= 0.5), depth_weights= 'checkpoints/checkpoint_cdcn_cfas_t5.pth',cls_weights= 'checkpoints/checkpoint_cls_sm.pth').cuda()



sampler = None


pred = torch.Tensor().cuda()
target = torch.Tensor().cuda()
# minicsdata = [csdata[i] for i in range(2500,5000)]
# used_dataset = NUAADataset(root_dir = './data/processed/NUAA/test')
used_dataset = CFASDDataset(root_dir='data/processed/test_img', mode= 'val')
# used_dataset = ZaloDataset(root_dir = './data/processed/zalo_data')

# print("Calculating class distribution...")
# cls_dist = np.mean([sample[2] for sample in tqdm(used_dataset)])

# print("Assigning weight to samples...")
# weight_tensor = torch.Tensor([ (1-cls_dist) if used_dataset[i][2] == 1 else cls_dist for i in tqdm(range(used_dataset.__len__())) ])
# sampler = torch.utils.data.WeightedRandomSampler(weight_tensor,int(used_dataset.__len__()*1), replacement=True)
val_model.eval()
# val_model.net.eval()
val_loader = DataLoader(used_dataset, sampler= sampler, batch_size = 4, shuffle= False)

print("Inferencing...")
# with torch.no_grad():
#     for idx, batch in enumerate(tqdm(val_loader)):
#         sample, spoof_label = batch[0].float().cuda(), batch[2].float().cuda()
#         spoof_label = F.one_hot(spoof_label.to(torch.int64),2).float()
#         pred = torch.cat([pred, val_model(sample).squeeze(1)], dim = 0)
#         target = torch.cat([target, spoof_label], dim = 0)

with torch.no_grad():
    for idx, batch in enumerate(tqdm(val_loader)):
        sample, spoof_label = batch[0].float().cuda(), batch[2].float().cuda()
        spoof_label = F.one_hot(spoof_label.to(torch.int64),2).float()
        # print(val_model(sample)[1])
        pred = torch.cat([pred, (val_model(sample)[1].T)[1]], dim = 0)
        
        target = torch.cat([target, (spoof_label.T)[1]], dim = 0)
        # print(pred, target)

# print(target.shape)

roc = binary_roc(pred, target, thresholds = None)
fpr, tpr, threshold = roc[0].cpu(), roc[1].cpu(), roc[2].cpu()



best_threshold = threshold[np.argmin((((1 - tpr)+fpr)/2).numpy())]

# confusion_matrix = binary_confusion_matrix(pred, target, threshold= best_threshold.item())
confusion_matrix = binary_confusion_matrix(pred, target, threshold= .5)

print(confusion_matrix)

tp, fn, fp, tn = confusion_matrix[0,0], confusion_matrix[0,1], confusion_matrix[1,0], confusion_matrix[1,1]

apcer = fn/(tp+fn)
bpcer = fp/(tn+fp)
acer= (apcer+bpcer)/2
print("Best threshold:",best_threshold.item())
print("APCER:",apcer.item())
print("BPCER",bpcer.item())
print("ACER",acer.item())




plt.plot(fpr,fpr, 'r--')
plt.plot(fpr,tpr)
plt.title('ROC curve')
plt.show()