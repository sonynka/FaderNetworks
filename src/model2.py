# Copyright (c) 2017-present, Casti, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class EncoderLayer(nn.Module):
    def __init__(self, dim_in, dim_out, instance_norm=True, kernel=4, stride=2, pad=1):
        super(EncoderLayer, self).__init__()

        enc_layer = []
        enc_layer.append(nn.Conv2d(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=pad))
        norm_fn = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d
        enc_layer.append(norm_fn(dim_out))
        enc_layer.append(nn.LeakyReLU(0.2, inplace=True))

        self.layer = nn.Sequential(*enc_layer)

    def forward(self, x):
        return x + self.layer(x)


class DecoderLayer(nn.Module):
    def __init__(self, dim_in, dim_out, instance_norm=True, tanh=False, deconv_method='convtranspose'):
        super(DecoderLayer, self).__init__()

        dec_layer = []

        # deconvolution
        if deconv_method == 'upsampling':
            dec_layer.append(nn.UpsamplingNearest2d(scale_factor=2))
            dec_layer.append(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1))
        elif deconv_method == 'convtranspose':
            dec_layer.append(nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False))
        elif deconv_method == 'pixelshuffle':
            dec_layer.append(nn.Conv2d(dim_in, dim_out * 4, kernel_size=3, stride=1, padding=1))
            dec_layer.append(nn.PixelShuffle(upscale_factor=2))

        norm_fn = nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d
        dec_layer.append(norm_fn(dim_out, affine=True))

        if tanh:
            dec_layer.append(nn.Tanh())
        else:
            dec_layer.append(nn.ReLU())

        self.layer = nn.Sequential(*dec_layer)

    def forward(self, x):
        return x + self.layer(x)


class Encoder:
    def __init__(self, n_channels=3, conv_dim=16, n_layers=7):

        layers = []
        layers.append(EncoderLayer(n_channels, conv_dim))

        curr_dim = conv_dim
        for i in range(1, n_layers):
            layers.append(EncoderLayer(curr_dim, curr_dim*2))
            curr_dim = curr_dim * 2

        self.layers = nn.ModuleList(layers)


class Decoder:
    def __init__(self, conv_dim=512, n_attr=2, n_layers=7):

        layers = []
        curr_dim = conv_dim

        layers.append(DecoderLayer(curr_dim + 2*n_attr, curr_dim + 2*n_attr))

        for i in range(1, n_layers-1):
            layers.append(DecoderLayer(curr_dim + 2*n_attr, curr_dim/2 + 2*n_attr))
            curr_dim = curr_dim / 2

        layers.append(DecoderLayer(curr_dim + 2*n_attr, curr_dim/2 + 2*n_attr, tanh=True))

        self.layers = nn.ModuleList(layers)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        enc_output = x

        for layer in self.encoder.layers:
            enc_output = layer(enc_output)

        return enc_output

    def decode(self, enc_output, y):
        dec_output = enc_output

        y = y.unsqueeze(2).unsqueeze(3)

        for layer in self.decoder.layers:
            size = dec_output.size(2)
            input = torch.cat([dec_output, y.expand(y.size(0), y.size(1), size, size)], 1)
            dec_output = layer(input)

        return dec_output

    def forward(self, x, y):
        enc_output = self.encode(x)
        dec_output = self.decode(enc_output, y)
        return enc_output, dec_output


class DiscrimLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel, stride, pad, dropout=0.3):
        super(DiscrimLayer, self).__init__()

        enc_layer = []
        enc_layer.append(nn.Conv2d(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=pad))
        enc_layer.append(nn.BatchNorm2d(dim_out))
        enc_layer.append(nn.LeakyReLU(0.2, inplace=True))

        self.layer = nn.Sequential(*enc_layer)

    def forward(self, x):
        return x + self.layer(x)


class LatentDiscriminator(nn.Module):
    def __init__(self, enc_out_dim, conv_dim=512, dropout=0.3, n_attr=2):
        super(LatentDiscriminator, self).__init__()

        conv = []
        conv.append(EncoderLayer(enc_out_dim, conv_dim, instance_norm=False))
        conv.append(nn.Dropout(dropout))
        conv = nn.Sequential(*conv)

        fc = []
        fc.append(nn.Linear(conv_dim, conv_dim))
        fc.append(nn.Linear(conv_dim, n_attr))
        fc.append(nn.Sigmoid())
        fc = nn.Sequential(*fc)

        self.conv = conv
        self.fc = fc

    def forward(self, x):

        disc_output = self.conv(x)
        disc_output = disc_output.view(x.size(0), -1)
        disc_output = self.fc(disc_output)

        return disc_output