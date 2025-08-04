
import torch
import os
from torchvision import transforms


# from pvt_unet import pvt_unet
from densnet_unet import densenet_unet

def euclidean_distance(pred, target):
    """
    计算预测坐标和目标坐标之间的欧氏距离。
    Args:
        pred (torch.Tensor): (B, C, 2) 形状的预测坐标
        target (torch.Tensor): (B, C, 2) 形状的目标坐标
    Returns:
        distance (torch.Tensor): (B, C) 形状的距离
    """
    # print(f'pred.shape:{pred.shape}')
    # print(f'target.shape:{target.shape}')
    pred = pred.view(-1, 3, 2)  # Reshape to (batch_size, 3, 2), each point has x,y coordinates
    target = target.view(-1, 3, 2)
    return torch.sqrt(torch.sum((pred - target)**2, dim=2))

def dark_post_processing(heatmaps):
    """
    DARK后处理：通过泰勒展开对热力图坐标进行亚像素级别的精确化。
    Args:
        heatmaps (torch.Tensor): (B, C, H, W) 形状的预测热力图。
    Returns:
        coords (torch.Tensor): (B, C, 2) 形状的精确坐标。
    """
    B, C, H, W = heatmaps.shape
    
    # 找到每个热力图的最大值点
    heatmaps_reshaped = heatmaps.reshape(B, C, -1)
    _, max_indices = torch.max(heatmaps_reshaped, dim=2)
    
    max_coords_x = (max_indices % W).float()
    max_coords_y = (max_indices // W).float()
    
    # 计算梯度（一阶和二阶导数）
    # 为了避免边界问题，使用中心差分思想，但这里简化为临近差分
    dx = 0.5 * (heatmaps[:, :, :, 2:] - heatmaps[:, :, :, :-2])
    dy = 0.5 * (heatmaps[:, :, 2:, :] - heatmaps[:, :, :-2, :])
    dxx = heatmaps[:, :, :, 2:] - 2 * heatmaps[:, :, :, 1:-1] + heatmaps[:, :, :, :-2]
    dyy = heatmaps[:, :, 2:, :] - 2 * heatmaps[:, :, 1:-1, :] + heatmaps[:, :, :-2, :]
    dxy = 0.25 * (heatmaps[:, :, 2:, 2:] - heatmaps[:, :, 2:, :-2] - heatmaps[:, :, :-2, 2:] + heatmaps[:, :, :-2, :-2])

    # 在最大值点处提取导数值 (需要保证索引安全)
    safe_x = torch.clamp(max_coords_x.long(), 1, W - 2)
    safe_y = torch.clamp(max_coords_y.long(), 1, H - 2)

    # 提取一阶导数 (B, C)
    dx_val = dx[torch.arange(B)[:, None], torch.arange(C), safe_y, safe_x -1]
    dy_val = dy[torch.arange(B)[:, None], torch.arange(C), safe_y -1, safe_x]
    
    # 提取二阶导数 (B, C)
    dxx_val = dxx[torch.arange(B)[:, None], torch.arange(C), safe_y, safe_x-1]
    dyy_val = dyy[torch.arange(B)[:, None], torch.arange(C), safe_y-1, safe_x]
    dxy_val = dxy[torch.arange(B)[:, None], torch.arange(C), safe_y-1, safe_x-1]

    # 求解泰勒展开后的偏移量 (dx, dy)
    # H_inv * grad, H为Hessian矩阵
    determinant = dxx_val * dyy_val - dxy_val**2
    
    # 避免除以零
    inv_determinant = torch.where(
        torch.abs(determinant) > 1e-6,
        1.0 / determinant,
        torch.zeros_like(determinant)
    )

    offset_x = (dxy_val * dy_val - dyy_val * dx_val) * inv_determinant
    offset_y = (dxy_val * dx_val - dxx_val * dy_val) * inv_determinant
    
    # 将偏移量限制在一个像素范围内
    offset_x = torch.clamp(offset_x, -0.5, 0.5)
    offset_y = torch.clamp(offset_y, -0.5, 0.5)

    # 计算最终坐标
    refined_x = max_coords_x + offset_x
    refined_y = max_coords_y + offset_y
    
    # 归一化坐标
    refined_x /= W
    refined_y /= H
    
    return torch.stack([refined_x, refined_y], dim=2)

def extract_coordinates(heatmaps, use_dark=False):
    """从热力图中提取关键点坐标"""
    if use_dark:
        return dark_post_processing(heatmaps)
    B, C, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(B, C, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    y_coords = torch.div(max_indices, W, rounding_mode='floor').float() / H
    x_coords = (max_indices % W).float() / W
    coords = torch.zeros(B, C * 2, device=heatmaps.device)
    for i in range(C):
        coords[:, i*2] = x_coords[:, i]
        coords[:, i*2+1] = y_coords[:, i]
    return coords 

class model:
    def __init__(self):
        '''
        初始化模型，只支持CPU推理，优化数据预处理
        '''
        self.device = torch.device("cpu")
      
       
        self.densenet_unet_model1 = densenet_unet(3,6,64,arch='densenet121').to(self.device)

        self.densenet_unet_model2 = densenet_unet(3,6,64,arch='densenet121').to(self.device)
        self.fusion_weights = [0.5,0.5]  # [pvt_model1, densenet_unet_model, pvt_model2]
        # 优化：transform只初始化一次
        self.tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def load(self, path="./"):
        '''
        加载模型权重，并进行动态量化
        '''
        # 加载 pvt_model1 权重
      

        # 加载 densenet_unet_model 权重
        densenet_unet_path1 = os.path.join(path, "densenet_unet_ mixup_pse_825_se_100_le_220_prob_1_best_val_distance_model.pth")
        print(f"尝试加载 densenet_unet_model1 权重: {densenet_unet_path1}")
        if os.path.exists(densenet_unet_path1):
            try:
                checkpoint = torch.load(densenet_unet_path1, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.densenet_unet_model1.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.densenet_unet_model1.load_state_dict(checkpoint)
                print("densenet_unet_model1 权重加载成功")
            except Exception as e:
                print(f"加载 densenet_unet_model1 权重失败: {e}")


        densenet_unet_path2 = os.path.join(path, "densenet_unet_ mixup_pse_825_se_100_le_260_prob_2_best_val_distance_model.pth")
        print(f"尝试加载 densenet_unet_mode2 权重: {densenet_unet_path2}")
        if os.path.exists(densenet_unet_path2):
            try:
                checkpoint = torch.load(densenet_unet_path2, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.densenet_unet_model2.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.densenet_unet_model2.load_state_dict(checkpoint)
                print("densenet_unet_model2 权重加载成功")
            except Exception as e:
                print(f"加载 densenet_unet_model2 权重失败: {e}")

        else:
            print(f"未找到 densenet_unet_model2 权重文件: {densenet_unet_path2}")

     
        return self

    def predict(self, X):
        '''
        输入一张图片，输出关键点坐标
        '''

    
        self.densenet_unet_model1.eval()
        self.densenet_unet_model2.eval()
        width, height = X.size
        image = self.tf(X).unsqueeze(0).to(self.device)

        with torch.no_grad():
          
            heatmaps1 = self.densenet_unet_model1(image)[:,3:,:,:] ###取前3个通道作为最终的预测结果！
            heatmaps2 = self.densenet_unet_model2(image)[:,3:,:,:]
         
            heatmaps = (
                        self.fusion_weights[0] * heatmaps1 +
                        self.fusion_weights[1] * heatmaps2)

            coords = extract_coordinates(heatmaps, use_dark=True)

        coords = coords.squeeze(0).cpu().numpy()
        coords[::2] *= width
        coords[1::2] *= height

        return coords
    
    def save(self, path="./"):
        '''
        保存模型权重
        '''
        pass

# 输入输出规范同原注释
if __name__=='__main__':
    x = torch.rand(1,3,512,512)
    x_image = transforms.ToPILImage()(x.squeeze(0))
    net = model().load()
    out = net.predict(x_image)
    print(f'out:{out}')

    