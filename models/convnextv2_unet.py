from __future__ import division, print_function
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import sys
sys.path.append(r'F:\DeepLearning_ICG\Hjl_Lx\IUGC2025\IUGC2025_baseline\models')
from convnextv2 import convnextv2_tiny,convnextv2_base


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

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
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.double_up =nn.Upsample(
                scale_factor=4, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):

       
        if self.bilinear:
            x1 = self.conv1x1(x1)
        
        if x1.size(3) != x2.size(3):
          x1 = self.up(x1)
         
        elif x1.size(3) == x2.size(3):
            x1 = self.double_up(x1)
            x2 = self.double_up(x2)

        
        if x1.size(2) != x2.size(2):
            h1, h2 = x1.size(2), x2.size(2)
            if h1 > h2:
            # 裁剪 x1 至 h2 的高度
                 x1 = x1[:, :, :h2, :]
            else:
                if h1 == 20:
                # 计算需要填充的高度
                    pad_needed = h2 - h1
                    # 在高度维度（第三维）的下方填充 pad_needed 个像素
                    x1 = F.pad(x1, (0, 0, 0, pad_needed))
                else:
                # 裁剪 x2 至 h1 的高度
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
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

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
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)  # 5个特征层：stem + 4个stages

        # 调整上采样层的通道数，使用params中的dropout参数
        # 注意：dropout参数从深到浅，对应从up1到up4
        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=self.dropout[3])
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=self.dropout[2])
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=self.dropout[1])
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=self.dropout[0])

        # 最终输出层
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        
    
    def forward(self, feature,):
        x0 = feature[0]  # stem
        x1 = feature[1]  # stage 1
        x2 = feature[2]  # stage 2
        x3 = feature[3]  # stage 3
        x4 = feature[4]  # stage 4

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
       
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


##=========================================my-unet-2d=========================================================


def encode_for_convnext(e, x):
    # 直接使用 forward_features 获取所有特征
    features = e.forward_features(x)
    return features

class convnext_unet(nn.Module):
    def __init__(self, in_chns, class_num, heatmap_size=64, 
    arch='convnextv2_tiny', dropout = [0.05, 0.1, 0.2, 0.3, 0.5]):
        super(convnext_unet, self).__init__()

        self.arch = arch
      
        self.heatmap_size = heatmap_size
        self.encoder_dim = {
            'convnextv2_tiny': [96, 96, 192, 384, 768],  # stem + 4 stages
            'convnextv2_base': [128, 128, 256, 512, 1024]  # stem + 4 stages
        }

        # 根据架构选择编码器
        if self.arch == 'convnextv2_tiny':
            self.encoder = convnextv2_tiny()
            pretrained_path = "pretrained_models/convnextv2_tiny_1k_224_ema.pt"
        elif self.arch == 'convnextv2_base':
            self.encoder = convnextv2_base()
            pretrained_path = "pretrained_models/convnextv2_base_1k_224_ema.pt"
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")

        # 加载预训练权重
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model' in state_dict:
                self.encoder.load_state_dict(state_dict['model'], strict=True)
                print(f"成功加载预训练权重，权重存放路径: {pretrained_path}")
            else:
                self.encoder.load_state_dict(state_dict, strict=True)
                print(f"成功加载预训练权重，权重存放路径: {pretrained_path}")
        except FileNotFoundError:
            print(f"错误: 预训练权重文件不存在")
            print(f"权重存放路径: {pretrained_path}")
            print("请下载对应的预训练权重文件或使用随机初始化权重")
            print("使用随机初始化权重")
        except Exception as e:
            print(f"警告: 加载预训练权重失败: {str(e)}")
            print(f"权重存放路径: {pretrained_path}")
            print("使用随机初始化权重")

        if dropout is None:
            dropout = [0.05, 0.1, 0.2, 0.3, 0.5]
        # self.encoder = convnextv2_tiny()
        params = {'in_chns': in_chns,
                  'feature_chns': self.encoder_dim[self.arch],
                  'dropout': dropout,
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.decoder = Decoder(params)
        print(f'self.arch:{self.arch}')

    def forward(self, x, need_fp=False,):
        # 获取 stem 特征
        stem_feat = self.encoder.downsample_layers[0](x)
        # 获取各个 stage 的特征
        stage_feats = []
        x = stem_feat
        for i in range(4):
            x = self.encoder.stages[i](x)
            stage_feats.append(x)
            if i < 3:  # 除了最后一个stage，都需要下采样
                x = self.encoder.downsample_layers[i+1](x)             
        # 组合所有特征
        features = [stem_feat] + stage_feats
      
        
        output = self.decoder(features)
        if output.size(2) != self.heatmap_size or output.size(3) != self.heatmap_size:
            output = F.interpolate(output, size=(self.heatmap_size, self.heatmap_size), 
                                     mode='bilinear', align_corners=True)
        return output

    
if __name__=='__main__':

     x = torch.rand(1,3,512,512)
    #  net = UNet(3,3)
    #  out = net(x)
    #  print(f'out.shape:{out.shape}')

     net = convnext_unet(3,3)
     # 打印网络结构

     out = net(x)
     print(f'out.shape:{out.shape}')
