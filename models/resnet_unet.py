from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.distributions.uniform import Uniform

# -----------------------------------------------------------
# 权重初始化（沿用原实现）
# -----------------------------------------------------------
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


# -----------------------------------------------------------
# UNet 基础组件（完全沿用原实现）
# -----------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.double_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)

        # 尺寸对齐策略
        if x1.size(3) != x2.size(3):
            x1 = self.up(x1)
        elif x1.size(3) == x2.size(3):
            x1 = self.double_up(x1)
            x2 = self.double_up(x2)

        if x1.size(2) != x2.size(2):
            h1, h2 = x1.size(2), x2.size(2)
            if h1 > h2:
                x1 = x1[:, :, :h2, :]
            else:
                if h1 == 20:
                    pad_needed = h2 - h1
                    x1 = F.pad(x1, (0, 0, 0, pad_needed))
                else:
                    x2 = x2[:, :, :h1, :]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert len(self.ft_chns) == 5

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=self.dropout[3])
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=self.dropout[2])
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=self.dropout[1])
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=self.dropout[0])
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.out_conv(x)


# -----------------------------------------------------------
# ResNet18 + UNet
# -----------------------------------------------------------
ENCODER_DIM_RESNET18 = [64, 64, 128, 256, 512]  # 5 级特征通道数


def encode_for_resnet18(encoder, x):
    """
    手动抽取 ResNet18 的 5 级特征：
    [stem, layer1, layer2, layer3, layer4]
    """
    feats = []
    # stem
    x = encoder.conv1(x)
    x = encoder.bn1(x)
    x = encoder.act1(x)
    feats.append(x)          # 1/2
    x = encoder.maxpool(x)

    x = encoder.layer1(x)
    feats.append(x)          # 1/4

    x = encoder.layer2(x)
    feats.append(x)          # 1/8

    x = encoder.layer3(x)
    feats.append(x)          # 1/16

    x = encoder.layer4(x)
    feats.append(x)          # 1/32

    return feats


class ResNet18_UNet(nn.Module):
    def __init__(self, in_chns=3, class_num=3, heatmap_size=64, pretrained=True):
        super().__init__()
        self.heatmap_size = heatmap_size

        # timm 创建 ResNet18（去掉全局池化与分类头）
        self.encoder = timm.create_model(
            'resnet18',
            pretrained=False,
            in_chans=in_chns,
            num_classes=0,
            global_pool='',
            features_only=False  # 我们自己抽特征
        )

        # 构造解码器
        params = {
            'in_chns': in_chns,
            'feature_chns': ENCODER_DIM_RESNET18,
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu'
        }


        pretrained_path = r'pretrained_models\resnet18_feature_only.pth'
        print(f"正在尝试加载预训练权重: {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=True)
            print(f"✅ 成功加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"❌ 警告: 无法加载预训练权重 {pretrained_path}")
            print(f"错误详情: {str(e)}")
            print("将使用随机初始化的权重")


        self.decoder = Decoder(params)

    def forward(self, x):
        feats = encode_for_resnet18(self.encoder, x)
        out = self.decoder(feats)
        if out.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            out = F.interpolate(out, size=(self.heatmap_size, self.heatmap_size),
                                mode='bilinear', align_corners=True)
        return out


# -----------------------------------------------------------
# 简单测试
# -----------------------------------------------------------
if __name__ == '__main__':
    x = torch.randn(1, 3, 512, 512)
    net = ResNet18_UNet(3, 3, 64, pretrained=True)
    out = net(x)
    print('output shape:', out.shape)   # [1, 3, 64, 64]