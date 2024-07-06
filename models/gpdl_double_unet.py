# Copyright 2022 CircuitNet. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg19, VGG19_Weights

from collections import OrderedDict

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

class conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super(conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
                nn.InstanceNorm2d(dim_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.conv5 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        return x

class DoubleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super(DoubleUNet, self).__init__()
        # Network 1
        self.encoder1 = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.aspp1 = ASPP(512, 256)
        self.se1 = SqueezeExciteBlock(256)

        self.decoder1 = self.build_decoder(256, 128, out_channels)

        # Network 2
        self.encoder2 = self.build_encoder()
        self.aspp2 = ASPP(256, 128)
        self.se2 = SqueezeExciteBlock(128)

        self.decoder2 = self.build_decoder(128, 64, out_channels)

    def build_encoder(self):
        layers = []
        in_channels = 3
        for out_channels in [64, 128, 256]:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(SqueezeExciteBlock(out_channels))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def build_decoder(self, in_channels, mid_channels, out_channels):
        layers = []
        for mid_out_channels in [mid_channels, mid_channels//2, mid_channels//4]:
            layers.append(nn.ConvTranspose2d(in_channels, mid_out_channels, kernel_size=2, stride=2))
            layers.append(nn.Conv2d(mid_out_channels, mid_out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(mid_out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(mid_out_channels, mid_out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(mid_out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(SqueezeExciteBlock(mid_out_channels))
            in_channels = mid_out_channels
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Network 1
        x1 = self.encoder1(x)
        x1 = self.aspp1(x1)
        x1 = self.se1(x1)
        mask1 = self.decoder1(x1)
        
        # Upsample mask1 to match x's dimensions
        mask1_upsampled = F.interpolate(mask1, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # Network 2
        x2 = x * mask1_upsampled
        x2 = self.encoder2(x2)
        x2 = self.aspp2(x2)
        x2 = self.se2(x2)
        mask2 = self.decoder2(x2)

        # Concatenate mask1_upsampled and mask2 along the channel dimension
        final_mask = mask1_upsampled + mask2

        return final_mask

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

"""
# 针对有网络模型，但还没有训练保存 .pth 文件的情况
import netron
import torch.onnx

myNet = DoubleUNet()
x = torch.randn(3, 3, 256, 256)  # 随机生成一个输入
modelData = "./demo.pth"  # 定义模型数据保存的路径
# modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
netron.start(modelData)  # 输出网络结构
"""