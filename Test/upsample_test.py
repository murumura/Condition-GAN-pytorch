import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from src.infogan import Upsample

x=torch.randn([1,128, 8, 8])
print(x.shape)
y0 = F.interpolate(
            x, 
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )
print(y0.shape)