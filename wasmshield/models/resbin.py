from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
import torch
import tqdm
from wasmshield.models.base_handler import BaseHandler
from wasmshield.training.trainer import divide_chunks
import wasmshield.utils
import wasmshield.preprocessing
import numpy as np
from fastai.layers import ConvLayer, NormType

class SelfAttention(nn.Module):

    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, act_cls=None, bias=False)

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
    
class ResBlock(torch.nn.Module):

    def __init__(self, n):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n, n, kernel_size = 3, stride = 1, padding = 1,),
            torch.nn.BatchNorm2d(n),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.conv(x)+x)
    
class ConvRes(torch.nn.Module):
    def __init__(
        self, 
        att=False, 
        reduction='max', 
        num_layers=[1, 1, 1, 1, 1],
        num_channels=[32, 64, 64, 128, 128],
    ):
        
        super().__init__()

        self.num = 0
        layers_channels = list(zip(num_layers, num_channels))
        
        blocks = []
        last_chan = 3+self.num

        for nb_blocks_per_layer, new_chan in layers_channels:

            if last_chan != new_chan:

                blocks.extend(
                    [
                        nn.Conv2d(last_chan, new_chan, kernel_size = 1, stride = 1,),
                        nn.ReLU(inplace=True),
                        torch.nn.BatchNorm2d(new_chan),
                    ]
                )

            blocks.extend([ResBlock(new_chan,) for _ in range(nb_blocks_per_layer)])
            blocks.append(torch.nn.BatchNorm2d(new_chan))
            blocks.append(nn.MaxPool2d(2,2))
            if att is True:
                blocks.append(SelfAttention(new_chan))

            last_chan = new_chan

        self.reduction = (
            torch.nn.AdaptiveAvgPool2d(1) if reduction == 'avg'
            else (
                torch.nn.AdaptiveMaxPool2d(1) if reduction == 'max'
                else torch.nn.Identity()
            )
        )
            
        self.conv1 = torch.nn.Sequential(

            *blocks,
            self.reduction,
            torch.nn.Flatten(),

        )

    def forward(self, x):
        x = self.conv1(x)
        return x 

def build_resbin_18(device='mps'):
    return wasmshield.utils.set_torch_model_to_train(
        ConvRes(
            att=False, 
            reduction='max', 
            num_layers=[3,3,3,3,3],
            num_channels=[32, 64, 64, 128, 128],
        ),
        status=False
    ).to(device)

def build_resbin_18_sa(device='mps'):
    return wasmshield.utils.set_torch_model_to_train(
        ConvRes(
            att=True, 
            reduction='max', 
            num_layers=[3,3,3,3,3],
            num_channels=[32, 64, 64, 128, 128],
        ),
        status=False,
    ).to(device)

def build_resbin_8(device='mps'):
    return wasmshield.utils.set_torch_model_to_train(
        ConvRes(
            att=False, 
            reduction='max', 
            num_layers=[1,1,1,1,1],
            num_channels=[32, 64, 64, 128, 128],
        ),
        status=False,
    ).to(device)

def build_resbin_8_sa(device='mps'):
    return wasmshield.utils.set_torch_model_to_train(
        ConvRes(
            att=True, 
            reduction='max', 
            num_layers=[1,1,1,1,1],
            num_channels=[32, 64, 64, 128, 128],
        ),
        status=False,
    ).to(device)

class ResBinConfig(Enum):
    RESBIN_8 = build_resbin_8
    RESBIN_8_SA = build_resbin_8_sa
    RESBIN_18 = build_resbin_18
    RESBIN_18_SA = build_resbin_18_sa

class ResBinHandler(BaseHandler):

    def __init__(self, model:ConvRes):
        self.model = wasmshield.utils.set_torch_model_to_train(
            model, False
        )


    def preprocess_file(self, file:str, size:int=64):
        return [
            wasmshield.preprocessing.preprocess_image(file, size=size, use_pil=True)
        ]
    
    def preprocess_bytes(self, bytes:bytes, size:int=64):
        return [
            wasmshield.preprocessing.preprocess_image_from_bytes(bytes, size=size, use_pil=True)
        ]
    
    def get_vector_from_preprocessed_file(self, preprocessed_file, device='mps'):
        input = torch.tensor(
            np.array(preprocessed_file), 
            device=device,
            dtype=torch.float32
        ).to(device)
        output = self.model(input).cpu().numpy()
        return output 
    
    def get_vectors_from_files(
        self, 
        files, 
        size:int=64, 
        max_workers=24,
        device='mps'
    ):
        vecs = []
        for chunk in (list(divide_chunks(files,128))):
            input = torch.tensor(
                np.array([
                    self.preprocess_file(f, size=size)[0]
                    for f in chunk
                ]), device=device, dtype=torch.float32
            ).to(device)
            output = self.model(input).cpu().numpy()
            vecs.extend(output)
        return np.array(vecs)
    

