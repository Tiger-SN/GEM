
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn as nn
####Graph
from .models.Net import Net

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            #---CLIP Image encoder stem(x) conv1 shape: torch.Size([16, 32, 112, 112])                   
            #---CLIP Image encoder stem(x) conv2 shape: torch.Size([16, 32, 112, 112])
            #---CLIP Image encoder stem(x) conv3 shape: torch.Size([16, 64, 112, 112])
            #---CLIP Image encoder stem(x) shape: torch.Size([16, 64, 56, 56])
            x = self.relu1(self.bn1(self.conv1(x)))
            #print('CLIP Image encoder stem(x) conv1 shape:',x.shape)
            x = self.relu2(self.bn2(self.conv2(x)))
            #print('CLIP Image encoder stem(x) conv2 shape:',x.shape)
            x = self.relu3(self.bn3(self.conv3(x)))
            #print('CLIP Image encoder stem(x) conv3 shape:',x.shape)
            x = self.avgpool(x)
            return x
            ########Shaonan
            #return x,stem_conv3_x    
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        #print('CLIP Image encoder stem(x) shape:',x.shape)
        """
        ----ModifiedResNet layer1 shape: torch.Size([16, 256, 56, 56])
        ----ModifiedResNet layer2 shape: torch.Size([16, 512, 28, 28])
        ----ModifiedResNet layer3 shape: torch.Size([16, 1024, 14, 14])
        ----ModifiedResNet layer4 shape: torch.Size([16, 2048, 7, 7])
        ----ModifiedResNet attnpool shape: torch.Size([16, 1024])
        """

        #x = self.layer1(x)
        #print('----ModifiedResNet layer1 shape:',x.shape)
        #x = self.layer2(x)
        #print('----ModifiedResNet layer2 shape:',x.shape)
        #x = self.layer3(x)
        #print('----ModifiedResNet layer3 shape:',x.shape)
        #x = self.layer4(x)
        #print('----ModifiedResNet layer4 shape:',x.shape)      
        #x = self.attnpool(x)
        #print('----ModifiedResNet attnpool shape:',x.shape)
        
        ##################
        #####ModifiedResNet  shaonan ORIS
        ##################
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.attnpool(x4)  
        #print('-Resnet x5 shape:',x5.shape)
        
        
        #X_CRIS = (x2,x3,x4) 
        X_CRIS = (x.float(),x2.float(),x3.float())              
        return x5,X_CRIS
        ###Shaonan
        return x5,X_CRIS

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)


        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        #print('VisionTransformer forward x shape: ',x.shape)
        #VisionTransformer forward x shape:  torch.Size([64, 3, 224, 224])
        ####[B 768 7 7]
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        #print('self.conv1(x) shape:',x.shape)
        #VisionTransformer forward x Reshape: torch.Size([64, 768, 49])
        #####768 patch 16 x 16 x 3
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        #print('VisionTransformer forward x Reshape:',x.shape)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)          # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        ####[B,50,768]
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            #print('   if self.proj is not None: ')
            x = x @ self.proj
        #print('---x @ self.proj)',x.shape)
        return x



class prediction_MLP(nn.Module):
    #def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
    #def __init__(self, in_dim=512, hidden_dim=128, out_dim=512): # bottleneck structure
    #def __init__(self, in_dim=1024, hidden_dim=128, out_dim=147456): # bottleneck structure
    def __init__(self, in_dim=1024, hidden_dim=128, out_dim=150528): # Heat map
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.final_layer = nn.Conv2d(3, 64, 3, padding=1)  
        
        #####1X1conc   [ in chanel out chanel ]
        self.conv_121 = nn.Conv2d(3, 3, 3, padding=1)      
        self.out_layer = OutputTransition(64, 10)
        
        

        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x,ori_image):
    #def forward(self, x):
        x = self.layer1(x)
        #print('prediction_MLP self.layer1 x shape:',x.shape)
        ######[B,128,1]
        x = self.layer2(x)
        #print('--self.layer2(x):',x.shape)
        #######View
        x = x.view(x.shape[0],3,224,224)
        #x = x + ori_image  
        x_middle = self.conv_121(x)
        #print('--MLP-pre:',x.shape)
        #########final layer Conv [b 3 224 224] - >[b 64 3 224 224]
        x_conv = self.final_layer(x_middle)
        #####Output class
        x_output  = self.out_layer(x_conv)
        #print('Heatmap OutputTransition shape:',x_output.shape)
        return x_output ,x_middle


#（conv3 + bn + relu + conv3 + bn + resi + relu ）
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
        
#############DownSample
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(in_channels, out_channels, padding=padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
        
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.99)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.99)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out






#############
##Stem Patch
###########
from .gcn_lib import Grapher, act_layer


import math
from torch.autograd import Variable
"""
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
"""








############################
#######CRIS
############################
#from .Shaonan_CRIS import FPN, Projector, TransformerDecoder
#from .layers import FPN, Projector, TransformerDecoder
#from .layers_patch import FPN, Projector, TransformerDecoder
from .layers_patch_heatmap_Graph_W import FPN, Projector, TransformerDecoder
############Regresson

####Graph
from .models.Net import Net
class GEM_model(nn.Module):
    #def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
    #def __init__(self, in_dim=512, hidden_dim=128, out_dim=512): # bottleneck structure
    #def __init__(self, in_dim=1024, hidden_dim=128, out_dim=147456): # bottleneck structure
    #def __init__(self, in_dim=1024, hidden_dim=512, out_dim = 20): # Regerssion map
    #def __init__(self, in_dim=1024, hidden_dim=512, out_dim = 150528): # Regerssion map
    def __init__(self, in_dim=512, hidden_dim=256, out_dim = 37632): # Regerssion  Regin map
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        ###########CRIS
        # Multi-Modal FPN
        fpn_in = [512, 1024, 1024]
        fpn_out= [256, 512, 1024]
        self.neck = FPN(in_channels=fpn_in, out_channels=fpn_out)
        # Decoder3
        # Decoder 1
        num_layers = 3
        num_head = 8
        dim_ffn = 2048
        #dropout = 0.1
        
        ###Test
        #dropout = 0
        dropout = 0.1
        vis_dim = 512
        intermediate = False  
        self.decoder = TransformerDecoder(num_layers=num_layers,
                                          d_model=vis_dim,
                                          nhead=num_head,
                                          dim_ffn=dim_ffn,
                                          dropout=dropout,
                                          return_intermediate= intermediate)
        # Projector
        word_dim =  1024
        vis_dim=  512
        self.proj = Projector(word_dim, vis_dim // 2, 3)




        self.mlp_image = nn.Sequential(
            nn.Linear(512, 512)
            #nn.BatchNorm1d(512),
            #nn.ReLU(inplace=True)
        )
        self.mlp_text = nn.Sequential(
            nn.Linear(512, 512)
            #nn.BatchNorm1d(512),
            #nn.ReLU(inplace=True)
        )
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.final_layer = nn.Conv2d(3, 64, 3, padding=1)
        #####1X1conc   [ in chanel out chanel ]
        
        self.conv_121 = nn.Conv2d(3, 3, 3, padding=1)      
        #self.out_layer = OutputTransition(64, 10)
        
        
        ##############Basic Block ###[inChannle outchannel]
        self.BasicBlock = BasicBlock(3,3)
        
        #self.up_trans = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up_trans = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
        
        

        

        #############Graph Model
        self.MLP_Grah_Second = nn.Sequential(
            ####224 224 1
            nn.Linear(50176, 512),
            nn.ReLU(inplace=True),
            
            #nn.Linear(512, 256),
            #nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            #nn.Linear(256, 256),
            #nn.ReLU(inplace=True),
                        
            nn.Linear(256, 16),
            ##Sigmoid
            nn.Sigmoid()
        )



        #############
        ###Graph
        #############
        self.Gaph_Net = Net()
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    #def forward(self, x,ori_image):
    def forward(self, image_features, state, word,ori_phase, Landmarks_GT):
    
        vis = image_features
        # padding mask used in decoder
        #############
        # ###
        # ###
        #############
        pad_mask = torch.zeros_like(ori_phase).masked_fill_(ori_phase == 0, 1).bool()
        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        #vis = self.backbone.encode_image(img)
        #word, state = self.backbone.encode_text(word)

        fq = self.neck(vis, state)
        #print('Fusion Net shape:',fq.shape)
        b, c, h, w = fq.size()

        fq = self.decoder(fq, word, pad_mask)

        
        
        
        fq = fq.reshape(b, c, h, w)

        pred_map,Landmarks_GT = self.proj(fq, state,Landmarks_GT)       
        
        #######################
        ####Extract Patch
        #######################
        ###GT Patch
        #Landmarks_GT = Landmarks_GT.reshape(Landmarks_GT.shape[0],8,2)
        #print('---Landmarks_GT shape:',Landmarks_GT.shape)
        #print('---Landmarks_GT :',Landmarks_GT)        
        Landmarks_GT = Landmarks_GT*224
        
        Landmarks_GT = Landmarks_GT.to(dtype=torch.int)
        Landmarks_GT = Landmarks_GT.cpu().numpy()
        Landmarks_GT[Landmarks_GT>=219] = 200
        Landmarks_GT[Landmarks_GT<=5] = 8
        Landmarks_GT = Landmarks_GT.reshape(Landmarks_GT.shape[0],8,2)
        patch_size = 6
        block_size = 3

        #patch_size = 8
        #block_size = 4

        GT_patch = torch.zeros(Landmarks_GT.shape[0], 8, patch_size, patch_size)
        

        #print('****',Landmarks_GT)
        for i in range(Landmarks_GT.shape[0]):
            for j in range(8):
                #print('-----')
                x, y = Landmarks_GT[i, j]
                #print('x',x,'y',y)
                #print('---x ',x)
                #print('---y ',y)
                GT_patch[i, j, :, :] = pred_map[i, 0, x-block_size:x+block_size, y-block_size:y+block_size]
            #print('---GT_patch shape:',GT_patch.shape)
        #print('---GT_patch shape:',GT_patch.shape)
        GT_patch = GT_patch.view(GT_patch.shape[0],8,-1).to('cuda')




        #################
        ###Pred
        #print('MLP Regression')
        pred = pred_map.view(pred_map.shape[0],-1)
        ####No BN
        pred = self.MLP_Grah_Second(pred) 
        

        
        
        
        Landmarks_pred = pred*224
        
        Landmarks_pred = Landmarks_pred.to(dtype=torch.int)
        Landmarks_pred = Landmarks_pred.cpu().numpy()
        Landmarks_pred[Landmarks_pred>=219] = 200
        Landmarks_pred[Landmarks_pred<=5] = 8

        Landmarks_pred = Landmarks_pred.reshape(pred.shape[0],8,2)
        
        Pred_patch = torch.zeros(Landmarks_GT.shape[0], 8, patch_size, patch_size)

        for i in range(Pred_patch.shape[0]):
            for j in range(8):
                x, y = Landmarks_pred[i, j]

                Pred_patch[i, j, :, :] = pred_map[i, 0, x-block_size:x+block_size, y-block_size:y+block_size]
            #print('---Pred_patch shape:',Pred_patch.shape)
        #print('---Pred_patch shape:',Pred_patch.shape)
        Pred_patch = Pred_patch.view(Pred_patch.shape[0],8,-1).to('cuda')      
        #print('---Pred_patch shape:',Pred_patch.shape)

        ###########
        ###Graph###
        ###########
        out_heatmap_Graph,srcinlier_s,refinlier_s = self.Gaph_Net(GT_patch,Pred_patch)        



        #return pred,pred_patch
        ##Graph
        #return pred,out_heatmap,out_heatmap_Graph
        #return pred,out_heatmap,out_heatmap_Graph   
        return pred, out_heatmap_Graph,srcinlier_s,refinlier_s,pred_map





##################
##Decoder predict
#################
from clip.config import CFG
from torch import nn
from timm.models.layers import trunc_normal_
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=CFG.device))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(tgt):
    """
    tgt: shape(N, L)
    """
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_padding_mask = (tgt == CFG.pad_idx)

    return tgt_mask, tgt_padding_mask
############
##Decoder predict
###########
class Decoder(nn.Module):
    def __init__(self, vocab_size, encoder_length, dim, num_heads, num_layers):
        super().__init__()
        self.dim = dim
        
        #self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding = nn.Embedding(227, dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, CFG.max_len-1, dim) * .02)
        self.decoder_pos_drop = nn.Dropout(p=0.05)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim, vocab_size)
        
        
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, encoder_length, dim) * .02)
        self.encoder_pos_drop = nn.Dropout(p=0.05)
        
        self.init_weights()
        
    def init_weights(self):
        for name, p in self.named_parameters():
            if 'encoder_pos_embed' in name or 'decoder_pos_embed' in name: 
                print("skipping pos_embed...")
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        trunc_normal_(self.encoder_pos_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)
        
    
    def forward(self, encoder_out, tgt):
        """
        encoder_out: shape(N, L, D)
        tgt: shape(N, L)
        """
        #print('****** Decoder forward***************')
        #print('forward tgt shape :', tgt.shape)
        #print('forward tgt shape :',tgt)
        #forward tgt shape : torch.Size([2, 299])
        tgt_mask, tgt_padding_mask = create_mask(tgt)
        #print(tgt.max())  
        #print(tgt.min())
        tgt_embedding = self.embedding(tgt)


        # tgt_embedding tgt shape : torch.Size([2, 299, 256])
        #print('tgt_embedding tgt shape :',tgt_embedding.shape)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )
        # Decoder encoder_out shape : torch.Size([2, 576, 256])
        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )
        #print('Decoder encoder_out shape :',encoder_out.shape)

        encoder_out = encoder_out.transpose(0, 1)

        tgt_embedding = tgt_embedding.transpose(0, 1)


        preds = self.decoder(memory=encoder_out, 
                             tgt=tgt_embedding,
                             tgt_mask=tgt_mask, 
                             tgt_key_padding_mask=tgt_padding_mask)
        # Decoder preds shape : torch.Size([299, 2, 256])
        #print('Decoder preds shape :',preds.shape)
        
        preds = preds.transpose(0, 1)
        # Decoder preds shape : torch.Size([2, 299, 256])
        #print('Decoder preds shape :', preds.shape)
        # Decoder elf.output(preds) shape : torch.Size([2, 299, 407])
        #print('Decoder elf.output(preds) shape :',self.output(preds).shape)
        return self.output(preds)
    def predict(self, encoder_out, tgt):
        #print('Decoder Predict!!!!')
        length = tgt.size(1)
        padding = torch.ones(tgt.size(0), CFG.max_len-length-1).fill_(CFG.pad_idx).long().to(tgt.device)
        tgt = torch.cat([tgt, padding], dim=1)
        tgt_mask, tgt_padding_mask = create_mask(tgt)
        # is it necessary to multiply it by math.sqrt(d) ?
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )
        
        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )
        
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        #print('tgt_embedding :',tgt_embedding.shape)
        preds = self.decoder(memory=encoder_out, 
                             tgt=tgt_embedding,
                             tgt_mask=tgt_mask, 
                             tgt_key_padding_mask=tgt_padding_mask)
        
        preds = preds.transpose(0, 1)


        return self.output(preds)[:, length-1, :]        
        



class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        #print('Text shape :',text.shape)
        #Text shape : torch.Size([64, 77])
        #self.transformer(x) : torch.Size([77, 64, 512])
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        #print('self.token_embedding(text).type(self.dtype) :',x.shape)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        #print('self.transformer(x) :',x.shape)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x_text = self.ln_final(x).type(self.dtype)
        #self.ln_final(x).type(self.dtype) : torch.Size([2, 77, 512])
        #print('self.ln_final(x).type(self.dtype) :',x_text.shape)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x_text[torch.arange(x_text.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x,x_text
    ###########################
    #######Shaonan Liu
    ##########################
    #def forward(self, image,eye_image, text,tgt):
    #def forward(self, image, text,tgt):
    def forward(self, image, text):
        #print('----------CLIP forward--------------')

        
        
        image_features_clip,image_features = self.encode_image(image)
        text_features,x_text = self.encode_text(text)
 
        
        ######[B,299,?]
        # normalized features
        image_features_clip = image_features_clip / image_features_clip.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)     
        
        ###[B 512]
        #print('----- CLIP Image Features shape :',image_features.shape)
        # cosine similarity as logits
        
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features_clip @ text_features.t()
        logits_per_text = logits_per_image.t()
        

        logits_per_image = logit_scale * image_features_clip @ text_features.t()
        
           
        return  logits_per_image,logits_per_text,image_features, text_features, x_text
     


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        #vision_width = 128
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        #vision_width = 128
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    #print('CLIP model path :',state_dict)
    model.load_state_dict(state_dict)
    print('Load CLIP Pre model Shaonan pretrain.py')
    
    ################
    ##MGCN pertrained model
    #################
    #print('--Load MGCN----')
    #checkpoint = torch.load('MGCN/resnet_50.ckpt')
    #model.load_from_checkpoint(checkpoint_path='MGCN/resnet_50.ckpt')    


    ################
    ##CheXzero pertrained model
    #################
    #CheXzero_path = '/data1/liushaonan/CLIP/Eaygaze_CLIP/Chexzero/RN50.pt'
    #model.load_state_dict(torch.load(CheXzero_path, map_location='cpu'))


    ################
    ##Medical CLIP pertrained model
    #################
    #CheXzero_path = '/data1/liushaonan/CLIP/Eaygaze_CLIP/medical_CLIP/pytorch_model.bin'
    #model.load_state_dict(torch.load(CheXzero_path, map_location='cpu'))
    
    return model.eval()
    #########NO load pre train model
    # model.load_state_dict(state_dict)
    #return model
"""
x_input = torch.rand(2, 20, 3)
target_input = torch.rand(2, 20, 3)

x_input = torch.rand(2, 8, 2)
target_input = torch.rand(2, 8, 2)

model = Net()


GCM_output = model(x_input,target_input)
"""