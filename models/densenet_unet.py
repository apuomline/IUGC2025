import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# 复用pvt_unet.py中的ConvBlock、UpBlock、Decoder结构

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

class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True):
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

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=self.dropout[3])
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=self.dropout[2])
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=self.dropout[1])
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=self.dropout[0])
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
        self.layer3_out_conv = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size=3, padding=1)
    def forward(self, feature, return_layer3=False):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        if return_layer3:
            layer3_output = self.layer3_out_conv(x)
            return layer3_output
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

def encode_for_timm_densenet(e, x):
    feats = e(x)
    if isinstance(feats, dict):
        feats = list(feats.values())
    return feats

class densenet_unet(nn.Module):
    def __init__(self, in_chns, class_num, heatmap_size=64, arch='densenet121', pretrained=True, 
    pretrained_path='densenet121_tv_in1k_feature_only.pth'):
        super(densenet_unet, self).__init__()
        self.heatmap_size = heatmap_size
        self.encoder_dim = {
            'densenet121': [64, 256, 512, 1024, 1024],
            'densenet161': [96, 384, 768, 2112, 2208],
            'densenet169': [64, 256, 512, 1280, 1664],
            'densenet201': [64, 256, 512, 1792, 1920],
            'densenet264d': [96, 384, 1152, 3072, 3520],
        }
        assert arch in self.encoder_dim, f"arch must be one of {list(self.encoder_dim.keys())}"
        feature_chns = self.encoder_dim[arch]
        self.arch = arch
        # 用timm创建主干
        self.encoder = timm.create_model(
            arch,
            in_chans=in_chns,
            features_only=True,
            pretrained=pretrained if pretrained_path is None else False,
            out_indices=(0,1,2,3,4)
        )
        # 加载自定义预训练权重
        if pretrained_path is not None:
            print(f"正在尝试加载预训练权重: {pretrained_path}")
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.encoder.load_state_dict(state_dict, strict=True)
                print(f"✅ 成功加载预训练权重: {pretrained_path}")
            except Exception as e:
                print(f"❌ 警告: 无法加载预训练权重 {pretrained_path}")
                print(f"错误详情: {str(e)}")
                print("将使用随机初始化的权重")
        params = {'in_chns': in_chns,
                  'feature_chns': feature_chns,
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.decoder = Decoder(params)
    def forward(self, x):
        feature = encode_for_timm_densenet(self.encoder, x)
        output = self.decoder(feature)
       
        if output.size(2) != self.heatmap_size or output.size(3) != self.heatmap_size:
            output = F.interpolate(output, size=(self.heatmap_size, self.heatmap_size), mode='bilinear', align_corners=True)
        return output

if __name__ == '__main__':
    # 测试densenet121结构
    model = densenet_unet(in_chns=3, class_num=3, arch='densenet121', pretrained=True
    ,pretrained_path='densenet121_tv_in1k_feature_only.pth')
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print(f'densenet121 out.shape: {out.shape}')

    # # 测试densenet169结构
    # model2 = densenet_unet(in_chns=3, class_num=3, arch='densenet169', pretrained=True)
    # out2 = model2(x)
    # print(f'densenet169 out.shape: {out2.shape}')



