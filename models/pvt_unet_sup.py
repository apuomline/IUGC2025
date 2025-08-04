from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import timm
import torch.nn.functional as F

"""
PVT-UNet 模型支持任意层深度监督和特征图融合功能

使用示例:

1. 基本使用（无深度监督）:
   model = pvt_unet(in_chns=3, class_num=3, heatmap_size=64)
   output = model(x)  # 返回单个输出张量

2. 单个深度监督层:
   model = pvt_unet(in_chns=3, class_num=3, heatmap_size=64, 
                   deep_supervision_layers=[2])  # 只在layer3添加深度监督
   output = model(x)  # 直接返回layer3的输出张量

3. 多个深度监督层:
   model = pvt_unet(in_chns=3, class_num=3, heatmap_size=64,
                   deep_supervision_layers=[0, 1, 2])  # 在layer1, layer2, layer3添加深度监督
   outputs = model(x)  # 返回字典: {'main': tensor, 'layer1': tensor, 'layer2': tensor, 'layer3': tensor}

4. 所有深度监督层:
   model = pvt_unet(in_chns=3, class_num=3, heatmap_size=64,
                   deep_supervision_layers=[0, 1, 2, 3])  # 所有层都添加深度监督
   outputs = model(x)  # 返回包含所有层输出的字典

5. 特征图融合模式:
   model = pvt_unet(in_chns=3, class_num=3, heatmap_size=64,
                   deep_supervision_layers=[0, 1, 2], fusion_mode='weighted_sum')
   output = model(x)  # 返回融合后的单个输出张量

深度监督层索引说明:
- 0: layer1 (第一个上采样层输出)
- 1: layer2 (第二个上采样层输出)  
- 2: layer3 (第三个上采样层输出)
- 3: layer4 (第四个上采样层输出)

融合模式说明:
- 'weighted_sum': 加权求和融合
- 'attention': 注意力机制融合
- 'concat': 通道拼接后卷积融合
- 'none': 不融合，返回字典

输出说明:
- 单层深度监督：直接返回对应的特征图张量
- 多层深度监督：返回包含所有层输出的字典
- 启用融合模式：返回融合后的单个张量

所有输出都会被调整到指定的 heatmap_size 尺寸。
"""



def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """Double Convolution Block"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )

    def forward(self, x):
        return self.conv(x)


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


class FeatureFusionModule(nn.Module):
    """特征图融合模块"""
    
    def __init__(self, num_features, fusion_mode='weighted_sum', num_classes=3):
        super(FeatureFusionModule, self).__init__()
        self.num_features = num_features
        self.fusion_mode = fusion_mode
        self.num_classes = num_classes
        
        if fusion_mode == 'weighted_sum':
            # 加权求和融合
            self.weights = nn.Parameter(torch.ones(num_features) / num_features)
            self.softmax = nn.Softmax(dim=0)
            
        elif fusion_mode == 'attention':
            # 注意力机制融合
            attn_mid = max(1, self.num_classes // 4)
            self.attention_weights = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.num_classes, attn_mid, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(attn_mid, self.num_classes, 1),
                    nn.Sigmoid()
                ) for _ in range(num_features)
            ])
            
        elif fusion_mode == 'concat':
            # 通道拼接后卷积融合
            fusion_in_channels = num_classes * num_features
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(fusion_in_channels, num_classes * num_features // 2, 3, padding=1),
                nn.BatchNorm2d(num_classes * num_features // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes * num_features // 2, num_classes, 3, padding=1)
            )
    
    def forward(self, features):
        """
        Args:
            features: list of tensors, each with shape (B, C, H, W)
        Returns:
            fused_feature: tensor with shape (B, C, H, W)
        """
        # 首先将所有特征图调整到相同尺寸（使用第一个特征图的尺寸作为目标）
        target_size = features[0].shape[2:]  # (H, W)
        aligned_features = []
        
        for feature in features:
            if feature.shape[2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=True)
            aligned_features.append(feature)
        
        if self.fusion_mode == 'weighted_sum':
            # 加权求和
            weights = self.softmax(self.weights)
            fused_feature = sum(w * f for w, f in zip(weights, aligned_features))
            
        elif self.fusion_mode == 'attention':
            # 注意力机制融合
            attention_weights = [attn(f) for attn, f in zip(self.attention_weights, aligned_features)]
            fused_feature = sum(w * f for w, f in zip(attention_weights, aligned_features))
            
        elif self.fusion_mode == 'concat':
            # 通道拼接
            concat_feature = torch.cat(aligned_features, dim=1)
            fused_feature = self.fusion_conv(concat_feature)
            
        else:
            # 默认平均融合
            fused_feature = torch.stack(aligned_features).mean(dim=0)
            
        return fused_feature


class Decoder(nn.Module):
    def __init__(self, params, deep_supervision_layers=None, fusion_mode='none'):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        
        # 深度监督配置
        self.deep_supervision_layers = deep_supervision_layers
        if self.deep_supervision_layers is None:
            self.deep_supervision_layers = []  # 默认不进行深度监督
        
        # 融合模式
        self.fusion_mode = fusion_mode

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=self.dropout[3])
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=self.dropout[2])
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=self.dropout[1])
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=self.dropout[0])

        # 主输出层
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        
        # 深度监督输出层 - 为每个解码器层创建输出卷积
        self.deep_supervision_convs = nn.ModuleDict()
        for i in range(4):  # 4个上采样层
            if i in self.deep_supervision_layers:
                layer_name = f'layer{i+1}_out_conv'
                self.deep_supervision_convs[layer_name] = nn.Conv2d(
                    self.ft_chns[3-i], self.n_class, kernel_size=3, padding=1)
        
        # 特征融合模块
        if self.fusion_mode != 'none' and len(self.deep_supervision_layers) > 1:
            self.feature_fusion = FeatureFusionModule(
                num_features=len(self.deep_supervision_layers) + 1,  # +1 for main output
                fusion_mode=self.fusion_mode,
                num_classes=self.n_class
            )

    def forward(self, feature, return_deep_supervision=False):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        # 解码器前向传播
        x_up1 = self.up1(x4, x3)
        x_up2 = self.up2(x_up1, x2)
        x_up3 = self.up3(x_up2, x1)
        x_up4 = self.up4(x_up3, x0)
        
        # 主输出
        main_output = self.out_conv(x_up4)
        
        if return_deep_supervision:
            # 如果只有单层深度监督，直接返回该层的输出
            if len(self.deep_supervision_layers) == 1:
                i = self.deep_supervision_layers[0]
                layer_name = f'layer{i+1}_out_conv'
                if layer_name in self.deep_supervision_convs:
                    if i == 0:  # layer1对应x_up1
                        return self.deep_supervision_convs[layer_name](x_up1)
                    elif i == 1:  # layer2对应x_up2
                        return self.deep_supervision_convs[layer_name](x_up2)
                    elif i == 2:  # layer3对应x_up3
                        return self.deep_supervision_convs[layer_name](x_up3)
                    elif i == 3:  # layer4对应x_up4
                        return self.deep_supervision_convs[layer_name](x_up4)
            
            # 收集深度监督输出（多层情况）
            deep_outputs = {}
            deep_outputs['main'] = main_output
            
            # 为每个指定的深度监督层生成输出
            for i in self.deep_supervision_layers:
                layer_name = f'layer{i+1}_out_conv'
                if layer_name in self.deep_supervision_convs:
                    if i == 0:  # layer1对应x_up1
                        deep_outputs[f'layer{i+1}'] = self.deep_supervision_convs[layer_name](x_up1)
                    elif i == 1:  # layer2对应x_up2
                        deep_outputs[f'layer{i+1}'] = self.deep_supervision_convs[layer_name](x_up2)
                    elif i == 2:  # layer3对应x_up3
                        deep_outputs[f'layer{i+1}'] = self.deep_supervision_convs[layer_name](x_up3)
                    elif i == 3:  # layer4对应x_up4
                        deep_outputs[f'layer{i+1}'] = self.deep_supervision_convs[layer_name](x_up4)
            
            # 如果启用融合模式，则融合所有输出
            if self.fusion_mode != 'none' and len(self.deep_supervision_layers) > 1:
                # 收集所有需要融合的特征
                features_to_fuse = [main_output]
                for i in self.deep_supervision_layers:
                    layer_key = f'layer{i+1}'
                    if layer_key in deep_outputs:
                        features_to_fuse.append(deep_outputs[layer_key])
                
                # 融合特征
                fused_output = self.feature_fusion(features_to_fuse)
                deep_outputs['fused'] = fused_output
            
            return deep_outputs
        
        return main_output


def encode_for_pvt(e, x):

    encode=[]
    x = e.patch_embed(x)
    x_ = x.permute(0,3,1,2).contiguous()
    encode.append(x_)
    x = x_.permute(0,2,3,1).contiguous()
  
    x = e.stages_0(x)

    encode.append(x)

    x = e.stages_1(x)
    
    encode.append(x)

    x= e.stages_2(x)
   
    encode.append(x)

    x= e.stages_3(x)
 
    encode.append(x)

    return encode


class pvt_unet_sup(nn.Module):
    def __init__(self, in_chns, class_num, heatmap_size=64,
                 arch='pvt_v2_b1', deep_supervision_layers=None, 
                 fusion_mode='none', return_deep_supervision=False,
                 dropout=[0.05, 0.1, 0.2, 0.3, 0.5],
):
        super(pvt_unet_sup, self).__init__()
        self.arch = arch
        self.heatmap_size = heatmap_size
        self.return_deep_supervision = return_deep_supervision
        self.deep_supervision_layers = deep_supervision_layers
        self.fusion_mode = fusion_mode
        
        # 如果指定了深度监督层，则启用深度监督
        if self.deep_supervision_layers is not None:
            self.return_deep_supervision = True
            
        self.encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34':[64, 64, 128, 256, 512, ],
            'resnet34d': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small': [96,96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96,96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
            'tf_efficientnet_b0.ns_jft_in1k':[16,24,40,112,320],
            'tf_efficientnet_b1.ns_jft_in1k':[16,24, 40, 112, 320],
            'tf_efficientnet_b2.ns_jft_in1k':[16,24,48,120,352],
            'tf_efficientnet_b3.ns_jft_in1k':[24,32, 48, 136, 384],
            'tf_efficientnet_b4.ns_jft_in1k':[24,32, 56, 160, 448],
            'tf_efficientnet_b5.ns_jft_in1k':[24,40, 64, 176, 512],
            'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
            'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
            'pvt_v2_b1': [64, 64, 128, 320, 512],
            'pvt_v2_b2': [64, 64,128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }

        # decoder_dim = \
        #       [256, 128, 64, 32, 16]
        
        self.encoder = timm.create_model(
            model_name = self.arch , pretrained=False, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )

        # 加载预训练权重
        pretrained_path = 'pretrained_models/pvt_v2_b1_feature_only.pth'
        print(f"正在尝试加载预训练权重: {pretrained_path}")
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=True)
            print(f"✅ 成功加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"❌ 警告: 无法加载预训练权重 {pretrained_path}")
            print(f"错误详情: {str(e)}")
            print("将使用随机初始化的权重")
        if dropout is None :
            dropout= [0.05, 0.1, 0.2, 0.3, 0.5]
        params = {'in_chns': in_chns,
                  'feature_chns': self.encoder_dim[self.arch],
                  'dropout': dropout,
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.decoder = Decoder(params, deep_supervision_layers=self.deep_supervision_layers, fusion_mode=self.fusion_mode)
        print(f'self.arch:{self.arch}')
        if self.deep_supervision_layers:
            print(f'深度监督层: {self.deep_supervision_layers}')
        if self.fusion_mode != 'none':
            print(f'特征融合模式: {self.fusion_mode}')

    def forward(self, x):
        feature = encode_for_pvt(self.encoder, x)
        
        if self.return_deep_supervision:
            outputs = self.decoder(feature, return_deep_supervision=True)
            
            # 如果只有单层深度监督，Decoder已经返回了单个张量
            if len(self.deep_supervision_layers) == 1:
                # 对单层输出进行尺寸调整
                if outputs.size(2) != self.heatmap_size or outputs.size(3) != self.heatmap_size:
                    outputs = F.interpolate(outputs, size=(self.heatmap_size, self.heatmap_size), 
                                          mode='bilinear', align_corners=True)
                return outputs
            
            # 多层深度监督情况
            # 对所有输出进行尺寸调整
            for key in outputs:
                if outputs[key].size(2) != self.heatmap_size or outputs[key].size(3) != self.heatmap_size:
                    outputs[key] = F.interpolate(outputs[key], size=(self.heatmap_size, self.heatmap_size), 
                                               mode='bilinear', align_corners=True)
            
            # 如果启用融合模式且只有一个输出，直接返回融合结果
            if self.fusion_mode != 'none' and 'fused' in outputs:
                return outputs['fused']
            
            return outputs
        
        output = self.decoder(feature)
        if output.size(2) != self.heatmap_size or output.size(3) != self.heatmap_size:
            output = F.interpolate(output, size=(self.heatmap_size, self.heatmap_size), 
                                     mode='bilinear', align_corners=True)
        return output

    
if __name__=='__main__':

    x = torch.rand(1,3,512,512)
    #  net = UNet(3,3)
    #  out = net(x)
    #  print(f'out.shape:{out.shape}')

    # 测试基本功能
    print("=== 测试基本功能 ===")
    net = pvt_unet_sup(3,3,64)
    out = net(x)
    print(f'基本输出 shape:{out.shape}')

    # 测试深度监督功能
    print("\n=== 测试深度监督功能 ===")
    
    # 测试单个深度监督层
    print("1. 测试单个深度监督层 (layer3):")
    net_ds1 = pvt_unet_sup(3, 3, 64, deep_supervision_layers=[2])  # layer3
    out_ds1 = net_ds1(x)
    print(f"输出类型: {type(out_ds1)}")
    for key, value in out_ds1.items():
        print(f"  {key}: {value.shape}")
    
    # 测试多个深度监督层
    print("\n2. 测试多个深度监督层 (layer1, layer3):")
    net_ds2 = pvt_unet_sup(3, 3, 64, deep_supervision_layers=[0, 2])  # layer1, layer3
    out_ds2 = net_ds2(x)
    print(f"输出类型: {type(out_ds2)}")
    for key, value in out_ds2.items():
        print(f"  {key}: {value.shape}")
    
