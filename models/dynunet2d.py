import torch
from monai.networks.nets import DynUNet

import torch.nn.functional as F



class MyDynUNet(DynUNet):
    def __init__(self, heatmapsize=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heatmapsize = heatmapsize
    def forward(self, x):
        out = super().forward(x)
        # 若输出不是heatmapsize则插值
        if out.shape[-1] != self.heatmapsize or out.shape[-2] != self.heatmapsize:
            out = F.interpolate(out, size=(self.heatmapsize, self.heatmapsize), mode="bilinear", align_corners=False)
        return out

def create_simple_dynunet_2d(heatmapsize=64):
    """
    创建简单的2D DynUNet模型，用于处理输入图像[1,3,512,512]
    """
    # 模型参数
    spatial_dims = 2  # 2D图像
    in_channels = 3   # RGB图像
    out_channels = 3  # 输出通道（heatmap数量）
    
    # 网络结构参数
    filters = (32, 64, 128, 256, 64, 32)  # 可根据需求调整
    num_layers = len(filters) - 1
    kernel_size = [(3, 3)] * num_layers
    strides = [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2)]  # 512->256->128->64->32->16
    upsample_kernel_size = [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2)]
    
    # 创建自定义DynUNet
    model = MyDynUNet(
        heatmapsize=heatmapsize,
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=filters,
        norm_name=("INSTANCE", {"affine": True}),
        deep_supervision=False,
        res_block=True,
    )
    
    return model


def test_model(heatmapsize=64):
    """
    测试模型前向传播，输出heatmapsize特征图
    """
    # 创建模型
    model = create_simple_dynunet_2d(heatmapsize=heatmapsize)
    
    # 创建测试输入 [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 3, 512, 512)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"最终输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    return model, output


if __name__ == "__main__":
    model, output = test_model(heatmapsize=64)
    print("DynUNet模型创建和测试成功!") 
