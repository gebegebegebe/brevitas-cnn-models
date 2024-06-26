# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from collections import namedtuple
from typing import Optional, Tuple, Any

import torch
from torch import Tensor
from torch import nn

import brevitas.nn as qnn
from brevitas.nn import QuantIdentity
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor
from brevitas.quant.scaled_int import Int32Bias 
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat

__all__ = [
    "GoogLeNetOutputs",
    "GoogLeNet",
    "BasicConv2d", "Inception", "InceptionAux",
    "googlenet",
]

bit_precision = 8

# According to the writing of the official library of Torchvision
GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

class GoogLeNet(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
            self,
            num_classes: int = 10,
            aux_logits: bool = True,
            transform_input: bool = False,
            dropout: float = 0.2,
            dropout_aux: float = 0.7,
    ) -> None:
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True, bias_quant=None, first_layer=True)
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv3 = BasicConv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.maxpool2 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes, dropout_aux)
            self.aux2 = InceptionAux(528, num_classes, dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = qnn.TruncAdaptiveAvgPool2d(output_size=(1, 1), bit_width=bit_precision, trunc_quant=TruncTo8bit, float_to_int_impl_type="ROUND")
        self.dropout = nn.Dropout(dropout, True)
        self.fc = nn.Linear(1024, num_classes)
        #self.fc = qnn.QuantLinear(1024, num_classes, bit_width=bit_precision, weight_bit_width=bit_precision, return_quant_tensor=False, bias=True, bias_quant=Int32Bias)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        out = self._forward_impl(x)

        return out

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            print("FOO")
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)

        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool2(out)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool3(out)
        out = self.inception4a(out)
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(out)

        out = self.inception4e(out)
        out = self.maxpool4(out)
        out = self.inception5a(out)
        out = self.inception5b(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        aux3 = self.fc(out)

        return GoogLeNetOutputs(aux3, aux2, aux1)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def make_quant_conv2d(
        in_channels, out_channels, kernel_size, weight_bit_width, stride=1, padding=0, bias=False, input_quant=None, bias_quant=Int32Bias, first_layer=False):
    if first_layer:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)

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
        weight_quant=Int8WeightPerChannelFloat,
        bias_quant=bias_quant)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, requantize=True, bias=False, bias_quant=Int32Bias, first_layer=False, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        kernel_size,_ = kwargs["kernel_size"]
        stride,_ = kwargs["stride"]
        padding,_ = kwargs["padding"]
        self.requantize = requantize
        #self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        if first_layer:
            self.conv = make_quant_conv2d(in_channels, out_channels, kernel_size, 8, stride, padding, bias, None, bias_quant, True)
        else:
            self.conv = make_quant_conv2d(in_channels, out_channels, kernel_size, bit_precision, stride, padding, bias, None, bias_quant)
        self.relu = qnn.QuantReLU(bit_width=bit_precision, return_quant_tensor=True)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        if self.requantize:
            out = self.relu(out)

        return out


class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3red: int,
            ch3x3: int,
            ch5x5red: int,
            ch5x5: int,
            pool_proj: int,
    ) -> None:
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), requantize=False, bias=True)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), requantize=False, bias=True),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), requantize=False, bias=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), requantize=False, bias=True),
        )

        self.out_quant = qnn.QuantIdentity(bit_width=bit_precision, return_quant_tensor=True)

    def forward(self, x: Tensor) -> Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        if not self.training:
            #print(False)
            #print(dir(self.out_quant))
            self.out_quant.eval()

        branch1 = self.out_quant(branch1)
        branch2 = self.out_quant(branch2)
        branch3 = self.out_quant(branch3)
        branch4 = self.out_quant(branch4)
        out = [branch1, branch2, branch3, branch4]

        #out = torch.cat(out, 1)
        #print(type(out))
        out = torch.cat(out, 1)

        return out


class InceptionAux(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            dropout: float = 0.7,
    ) -> None:
        super().__init__()
        #self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.avgpool = qnn.TruncAdaptiveAvgPool2d(output_size=(4, 4), bit_width=bit_precision, trunc_quant=TruncTo8bit, float_to_int_impl_type="ROUND")
        self.conv = BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        #self.relu = nn.ReLU(True)
        self.relu = qnn.QuantReLU(bit_width=bit_precision, return_quant_tensor=False)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        """
        self.relu = qnn.QuantReLU(bit_width=bit_precision, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(2048, 1024, bit_width=bit_precision, weight_bit_width=bit_precision, bias=True, quant_bias=Int32Bias)
        self.fc2 = qnn.QuantLinear(1024, num_classes, bit_width=bit_precision, weight_bit_width=bit_precision, bias=True, quant_bias=Int32Bias)
        """
        self.dropout = nn.Dropout(dropout, True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.avgpool(x)
        out = self.conv(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


def get_inception_v1(**kwargs: Any) -> GoogLeNet:
    model = GoogLeNet(**kwargs)

    return model
