import torch
import torch.onnx
from model.C_CDN import DC_CDN
from model.CDCN import CDCN
from model.Finetune import Finetune_model
from model.CDCNv2 import Conv2d_X
import numpy as np

model = CDCN().cuda()
ft_model = Finetune_model(depth_model= model ,depth_weights= 'checkpoints\checkpoint_cdcn_mix.pth', cls_weights= 'checkpoints\checkpoint_cls.pth')
ft_model.eval()
x = torch.rand((1,3,256,256),requires_grad= True).cuda()

torch_out = ft_model(x)


torch.onnx.export(  ft_model,
                    x,
                    "pickle/cdcn.onnx",
                    export_params= True,
                    opset_version= 12,
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}} )

# import onnx

# onnx_model = onnx.load("pickle/cdcn.onnx")
# onnx.checker.check_model(onnx_model)

# import onnxruntime

# ort_session = onnxruntime.InferenceSession("pickle/cdcn.onnx")

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


