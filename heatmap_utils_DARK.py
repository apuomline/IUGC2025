import torch
import torch.nn.functional as F

class HeatmapLoss(torch.nn.Module):
    """
    热力图的均方误差损失 (MSE Loss)
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        return F.mse_loss(pred, gt)

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
    # 与heatmap_net.py一致，输出shape为[batch, num_keypoints*2]，顺序为x1,y1,x2,y2...
    heatmaps_reshaped = heatmaps.reshape(B, C, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    y_coords = torch.div(max_indices, W, rounding_mode='floor').float() / H
    x_coords = (max_indices % W).float() / W
    coords = torch.zeros(B, C * 2, device=heatmaps.device)
    for i in range(C):
        coords[:, i*2] = x_coords[:, i]
        coords[:, i*2+1] = y_coords[:, i]
    return coords 