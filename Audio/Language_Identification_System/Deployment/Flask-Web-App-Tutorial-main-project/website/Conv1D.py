import itertools

from sklearn.metrics import roc_curve,accuracy_score,confusion_matrix,recall_score,precision_score,f1_score, auc, roc_auc_score,classification_report
import torch
import torchaudio
from torch import nn
import numpy as np 
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import torch.nn.functional as F
import os
# PyTorch
import warnings
from torch.utils.data import Subset

import torch
import torch.nn as nn
from torch import cuda, optim
from torch.utils.data import DataLoader, sampler
from torchmetrics import ConfusionMatrix
from torchvision import datasets, models, transforms

warnings.filterwarnings("ignore", category=FutureWarning)

import os

# Timing utility
from timeit import default_timer as timer



# Useful for examining network
from torchsummary import summary

import warnings

from IPython.display import clear_output

warnings.filterwarnings("ignore")

train_on_gpu = cuda.is_available()

import matplotlib.pyplot as plt



class conv1dblock(nn.Module):
    def __init__(self,out_channels,kernel_size,stride,pool_kernal,stride_step,pading=0) :
        super().__init__()
        self.conv=nn.LazyConv1d(out_channels,kernel_size,stride,pading)
        self.maxpool=nn.MaxPool1d(pool_kernal,stride_step,)
    def forward(self,x):
        x=self.conv(x)
        x=self.maxpool(x)
        return F.relu(x)
        
class Network(nn.Module):
    
    def stem(self,stem_channels):
        return nn.Sequential(nn.LazyConv1d(stem_channels, kernel_size=9, stride=3),nn.ReLU())
    def stage(self,out_channels,kernel_size,stride,pool_kernal,stride_step,no_layer):
        block=[]
        for i in range(no_layer):
            if i ==0:
                block.append(conv1dblock(out_channels,kernel_size,stride,pool_kernal,stride_step))
                block.append(conv1dblock(out_channels,kernel_size,stride,pool_kernal,stride_step))
                block.append(conv1dblock(out_channels*2,kernel_size,stride,pool_kernal,stride_step))
            elif i == no_layer-1:
                block.append(conv1dblock(out_channels*(2)**i,kernel_size,stride,pool_kernal,stride_step))
                block.append(conv1dblock(out_channels*(2)**(i+1),kernel_size,stride,int(106/(2**i-1)),stride_step))
            else: 
                block.append(conv1dblock(out_channels*(2)**i,kernel_size,stride,pool_kernal,stride_step))
                block.append(conv1dblock(out_channels*(2)**(i+1),kernel_size,stride,pool_kernal,stride_step))
        return nn.Sequential(*block)
    def __init__(self,out_channels,kernel_size,stride,pool_kernal,stride_step,no_layer, stem_channels,  num_classes=7) :
        super(Network, self).__init__()
        self.net= nn.Sequential(self.stem(stem_channels))
        
        self.net.add_module(f'stage{1}', self.stage(out_channels,kernel_size,stride,pool_kernal,stride_step,no_layer))
        self.net.add_module( f'output',nn.Sequential(nn.Flatten(),nn.LazyLinear(num_classes)))
    def forward(self,x):
        x=self.net(x)
        return x


class RegNetX32(Network):
    def __init__(self,  num_classes=7):
        stem_channels=128
        out_channels=128
        kernel_size=12
        stride=1
        pool_kernal=3
        stride_step =3
        no_layer=2       
        super().__init__(out_channels,kernel_size,stride,pool_kernal,stride_step,no_layer, stem_channels, num_classes)