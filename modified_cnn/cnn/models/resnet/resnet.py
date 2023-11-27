# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import brevitas.nn as qnn
from brevitas.nn import QuantIdentity
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor
from brevitas.quant.scaled_int import Int32Bias 
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat

def make_quant_conv2d(
        in_channels, out_channels, kernel_size, weight_bit_width, stride=1, padding=0, bias=False, input_quant=None, bias_quant=Int32Bias, weight_quant=Int8WeightPerChannelFloat):
    return qnn.QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        weight_bit_width=weight_bit_width,
        weight_scaling_per_output_channel=True,
        input_quant=input_quant,
        weight_quant=weight_quant,
        bias_quant=bias_quant)

class QuantBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self,in_planes,planes,stride=1,shared_quant_act=None,weight_bit_width=8,act_bit_width=8,bias=False):
        super(QuantBottleneckBlock, self).__init__()
        self.relu0 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv1 = make_quant_conv2d(in_planes,planes,kernel_size=1,stride=1,padding=0,bias=bias,weight_bit_width=weight_bit_width)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv2 = make_quant_conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=bias,weight_bit_width=weight_bit_width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv3 = make_quant_conv2d(planes,self.expansion * planes,kernel_size=1,stride=1,padding=0,bias=bias,weight_bit_width=weight_bit_width)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = nn.Sequential()
        self.relu3 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        #self.dropout = nn.Dropout(0.3)
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                make_quant_conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                    weight_bit_width=weight_bit_width,
                    input_quant=None,
                    weight_quant=None,
                    bias_quant=None),
                nn.BatchNorm2d(self.expansion * planes),
                )
            shared_quant_act = self.downsample[-1]
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        #self.relu3 = shared_quant_act
        self.relu_out = qnn.QuantReLU(return_quant_tensor=True, bit_width=act_bit_width)
        #self.adder = QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        self.adder = QuantIdentity(return_quant_tensor=False)
        self.identity = nn.Identity()

    def forward(self, x):
        #out = self.relu0(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        #out = self.dropout(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #out = self.dropout(out)
        if len(self.downsample):
            x = self.downsample(x)
        """
        out = self.adder(out)
        x = self.adder(x)
        # Check that the addition is made explicitly among QuantTensor structures
        assert isinstance(out, QuantTensor), "Perform add among QuantTensors"
        assert isinstance(x, QuantTensor), "Perform add among QuantTensors"
        """
        out = out + x
        out = self.relu_out(out)
        return out


class QuantResNet(nn.Module):

    def __init__(
            self,
            block_impl,
            num_blocks: List[int],
            first_maxpool=False,
            zero_init_residual=False,
            num_classes=10,
            weight_bit_width=8,
            act_bit_width=8):
        super(QuantResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = make_quant_conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, weight_bit_width=8, bias=True, input_quant=None, bias_quant=None, weight_quant=None)
        self.bn1 = nn.BatchNorm2d(64)
        shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.relu = shared_quant_act
        # MaxPool is typically present for ImageNet but not for CIFAR10
        if first_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()

        self.layer1, shared_quant_act = self._make_layer(
            block_impl, 64, num_blocks[0], 2, shared_quant_act, weight_bit_width, act_bit_width)
        self.layer2, shared_quant_act = self._make_layer(
            block_impl, 128, num_blocks[1], 2, shared_quant_act, weight_bit_width, act_bit_width)
        self.layer3, shared_quant_act = self._make_layer(
            block_impl, 256, num_blocks[2], 2, shared_quant_act, weight_bit_width, act_bit_width)
        self.layer4, _ = self._make_layer(
            block_impl, 512, num_blocks[3], 2, shared_quant_act, weight_bit_width, act_bit_width)

        # Performs truncation to 8b (without rounding), which is supported in FINN
        self.final_pool = qnn.TruncAvgPool2d(kernel_size=4, bit_width=act_bit_width, trunc_quant=TruncTo8bit, float_to_int_impl_type="ROUND")
        # Keep last layer at 8b
        self.linear = qnn.QuantLinear(
            512 * block_impl.expansion, num_classes, weight_bit_width=act_bit_width, bias=True, bias_quant=None, input_quant=None, weight_quant=None)
        #self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QuantBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self,
            block_impl,
            planes,
            num_blocks,
            stride,
            shared_quant_act,
            weight_bit_width,
            act_bit_width):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block = block_impl(
                self.in_planes, planes, stride, shared_quant_act, weight_bit_width, act_bit_width, bias=True)
            layers.append(block)
            shared_quant_act = layers[-1].relu_out
            self.in_planes = planes * block_impl.expansion
        return nn.Sequential(*layers), shared_quant_act

    def forward(self, x: Tensor):
        # There is no input quantizer, we assume the input is already 8b RGB
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.final_pool(out)
        out = out.view(out.size(0), -1)
        #self.dropout = nn.Dropout(0.5)
        out = self.linear(out)
        #self.dropout = nn.Dropout(0.5)
        return out

def get_resnet50():
    return QuantResNet(
        QuantBottleneckBlock, [3, 4, 6, 3],
        num_classes=1000,
        weight_bit_width=4,
        act_bit_width=4,
        first_maxpool=True)
