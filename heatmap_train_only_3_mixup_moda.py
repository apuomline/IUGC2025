import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime
import random
from PIL import Image
import yaml
from heatmap_dataset3 import HeatmapLandmarkDataset
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from heatmap_utils_DARK import HeatmapLoss, extract_coordinates, euclidean_distance
import segmentation_models_pytorch as smp
import models as models
import logging

from models.pvt_unet import pvt_unet
from models.convnextv2_unet import convnext_unet
from models.densenet_unet import densenet_unet


"""
现在问题是如何在训练文件中增加mixup方法？
"""

def mixup_data(images, heatmaps, landmarks, alpha=0.2, device='cuda'):
    """
    通道拼接MixUp：返回混合图像、拼接加权热图、拼接关键点、权重
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = images.size(0)
    if batch_size == 1:
        mixed_images = images
        mixed_heatmaps = torch.cat([heatmaps, heatmaps], dim=1)
        mixed_landmarks = torch.cat([landmarks.unsqueeze(1), landmarks.unsqueeze(1)], dim=1)
        mix_weights = torch.tensor([1.0, 0.0], device=device).view(1, 2)
        return mixed_images, mixed_heatmaps, mixed_landmarks, mix_weights
    index = torch.randperm(batch_size).to(device)
    mixed_images = lam * images + (1 - lam) * images[index, :]
    mixed_heatmaps = torch.cat([
        lam * heatmaps,
        (1 - lam) * heatmaps[index]
    ], dim=1)  # [B, 2K, H, W]
    mixed_landmarks = torch.cat([
        landmarks.unsqueeze(1),
        landmarks[index].unsqueeze(1)
    ], dim=1)  # [B, 2, K*2]
    mix_weights = torch.tensor([lam, 1-lam], device=device).view(1, 2)
    return mixed_images, mixed_heatmaps, mixed_landmarks, mix_weights

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
def update_args_from_config(args, config):
    """使用配置文件中的参数更新args"""
    # 定义参数名称映射
    param_mapping = {
        'data_train_csv': 'train_csv',
        'data_train_dir': 'train_dir', 
        'data_val_csv': 'val_csv',
        'data_val_dir': 'val_dir',
        'heatmap_size': 'heatmap_size',
        'heatmap_sigma': 'sigma',
        'heatmap_num_keypoints': 'num_keypoints',
        'training_batch_size': 'batch_size',
        'training_learning_rate': 'lr',
        'training_weight_decay': 'weight_decay',
        'training_epochs': 'epochs',
        'training_seed': 'seed',
        'scheduler_type': 'scheduler_type',
        'scheduler_step_size': 'scheduler_step_size',
        'scheduler_gamma': 'scheduler_gamma',
        'scheduler_patience': 'scheduler_patience',
        'scheduler_min_lr': 'scheduler_min_lr',
        'scheduler_warmup_epochs': 'warmup_epochs',
        'scheduler_exp_lr_decay': 'exp_lr_decay',
        'save_dir': 'save_dir',
        'save_interval': 'save_interval',
        'save_model_suffix': 'model_suffix',
        'save_timestamp': 'timestamp',
        'model_name': 'model_name',
        'model_arch': 'arch',
        'model_return_layer3': 'return_layer3',
        'pretrained': 'pretrained',
        'mixup_alpha': 'mixup_alpha'  # 添加MixUp参数映射
    }
    
    def update_nested_config(config_dict, prefix=''):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # 递归处理嵌套字典
                update_nested_config(value, prefix + key + '_')
            else:
                # 处理叶子节点
                param_name = prefix + key if prefix else key
                # 查找映射的参数名称
                mapped_param = param_mapping.get(param_name, param_name)
                
                if hasattr(args, mapped_param):
                    # 获取参数的类型
                    param_type = type(getattr(args, mapped_param))
                    # 转换值的类型
                    if param_type == bool:
                        # 对于布尔值，保持原样
                        setattr(args, mapped_param, value)
                    elif param_type == int:
                        # 对于整数，转换为int
                        setattr(args, mapped_param, int(value))
                    elif param_type == float:
                        # 对于浮点数，转换为float
                        setattr(args, mapped_param, float(value))
                    else:
                        # 对于其他类型（如字符串），保持原样
                        setattr(args, mapped_param, value)
                else:
                    # 调试信息：打印未找到的参数
                    print(f"Warning: Parameter '{mapped_param}' not found in args")
    
    update_nested_config(config)
    return args


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda', writer=None, save_dir=None, args=None):
    best_val_loss = float('inf')
    best_val_distance = float('inf')
    num_keypoints = args.num_keypoints
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_distance = 0.0
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for i, (images, heatmaps, landmarks) in enumerate(train_loop):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            landmarks = landmarks.to(device)
            if hasattr(args, 'mixup_alpha') and args.mixup_alpha > 0:
                mixed_images, mixed_heatmaps, mixed_landmarks, mix_weights = mixup_data(
                    images, heatmaps, landmarks, alpha=args.mixup_alpha, device=device
                )
                outputs = model(mixed_images)   # [B, 2K, H, W]
                loss = criterion(outputs, mixed_heatmaps)
                with torch.no_grad():
                    K = num_keypoints
                    outputs1 = outputs[:, :K, :, :]
                    outputs2 = outputs[:, K:, :, :]
                    pred_coords1 = extract_coordinates(outputs1, use_dark=False)
                    pred_coords2 = extract_coordinates(outputs2, use_dark=False)
                    distance1 = torch.mean(euclidean_distance(pred_coords1, mixed_landmarks[:, 0, :]))
                    distance2 = torch.mean(euclidean_distance(pred_coords2, mixed_landmarks[:, 1, :]))
                    distance = mix_weights[0, 0] * distance1 + mix_weights[0, 1] * distance2
            else:
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                with torch.no_grad():
                    pred_coords = extract_coordinates(outputs, use_dark=False)
                    distance = torch.mean(euclidean_distance(pred_coords, landmarks))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            batch_distance = distance.item()
            train_loss += batch_loss
            train_distance += batch_distance
            train_loop.set_postfix(loss=f"{batch_loss:.4f}", distance=f"{batch_distance:.4f}")
            iteration = (epoch - 1) * len(train_loader) + i
            writer.add_scalar('Loss/train_batch', batch_loss, iteration)
            writer.add_scalar('Distance/train_batch', batch_distance, iteration)
            writer.add_scalar('Learning_Rate/train_batch', optimizer.param_groups[0]['lr'], iteration)
            if hasattr(args, 'mixup_alpha') and args.mixup_alpha > 0:
                writer.add_scalar('MixUp/lambda', mix_weights[0, 0].item(), iteration)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_distance = train_distance / len(train_loader)
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_distance = 0.0
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for i, (images, heatmaps, landmarks) in enumerate(val_loop):
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                landmarks = landmarks.to(device)
                outputs = model(images)  # [B, 2K, H, W] or [B, K, H, W]
                K = num_keypoints
                if outputs.shape[1] == 2 * K:
                    outputs = outputs[:, :K, :, :]  # 只取前K通道
                loss = criterion(outputs, heatmaps)
                pred_coords = extract_coordinates(outputs, use_dark=False)
                distance = torch.mean(euclidean_distance(pred_coords, landmarks))
                batch_loss = loss.item()
                batch_distance = distance.item()
                val_loss += batch_loss
                val_distance += batch_distance
                val_loop.set_postfix(loss=f"{batch_loss:.4f}", distance=f"{batch_distance:.4f}")
                iteration = (epoch - 1) * len(val_loader) + i
                writer.add_scalar('Loss/val_batch', batch_loss, iteration)
                writer.add_scalar('Distance/val_batch', batch_distance, iteration)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_distance = val_distance / len(val_loader)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Distance/train_epoch', avg_train_distance, epoch)
        writer.add_scalar('Distance/val_epoch', avg_val_distance, epoch)
        writer.add_scalar('Learning_Rate/epoch', optimizer.param_groups[0]['lr'], epoch)
        print('=' * 50)
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Distance: {avg_train_distance:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Distance: {avg_val_distance:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        if hasattr(args, 'mixup_alpha') and args.mixup_alpha > 0:
            print(f'MixUp Alpha: {args.mixup_alpha}')
        print('=' * 50)
        if save_dir:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(save_dir, 'checkpoints', f'{args.model_name}_{args.model_suffix}_best_val_loss_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_distance': avg_val_distance,
                    'train_loss': avg_train_loss,
                    'train_distance': avg_train_distance
                }, save_path)
                print(f"Saved best validation loss model (loss: {best_val_loss:.4f}) to {save_path}")
            if avg_val_distance < best_val_distance:
                best_val_distance = avg_val_distance
                save_path = os.path.join(save_dir, 'checkpoints', f'{args.model_name}_{args.model_suffix}_best_val_distance_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_distance': best_val_distance,
                    'train_loss': avg_train_loss,
                    'train_distance': avg_train_distance
                }, save_path)
                print(f"Saved best validation distance model (distance: {best_val_distance:.4f}) to {save_path}")
    return best_val_loss, best_val_distance

def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create save directory with model name and suffix
    args.save_dir = os.path.join(args.save_dir, f"{args.model_name}_{args.model_suffix}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    
    # Save current configuration parameters
    with open(os.path.join(args.save_dir, 'training_config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create TensorBoard logger
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    # Basic data preprocessing
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # Create datasets
    train_dataset = HeatmapLandmarkDataset(
        csv_file=args.train_csv,
        img_dir=args.train_dir,
        transform=train_transform,
        train=True,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma
    )
    val_dataset = HeatmapLandmarkDataset(
        csv_file=args.val_csv,
        img_dir=args.val_dir,
        transform=val_transform,
        train=False,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma
    )
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # 关键点数
    num_keypoints = args.num_keypoints
    out_channels = num_keypoints * 2 if args.mixup_alpha > 0 else num_keypoints
    # Create model
    if 'pvt' in args.model_name.lower():
        model = pvt_unet(3, out_channels, heatmap_size=args.heatmap_size, arch=args.arch)
        print(f"Using PVT UNet model with architecture: {args.arch}")
    elif 'convnextv2' in args.model_name.lower():
        model = convnext_unet(3, out_channels, heatmap_size=args.heatmap_size, arch=args.arch)
        print(f"Using ConvNeXt UNet model with architecture: {args.arch}")
    elif 'densenet' in args.model_name.lower():
        model = densenet_unet(3, out_channels, heatmap_size=args.heatmap_size, arch=args.arch,
            pretrained=True,pretrained_path='pretrained_models/densenet121_tv_in1k_feature_only.pth')
        print(f"Using densenet_unet model with architecture: {args.arch}")
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    model = model.to(device)
    # Print model structure
    # print(model)
    # Define loss function
    criterion = HeatmapLoss()
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Define learning rate scheduler
    if args.scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
    elif args.scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.scheduler_gamma,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
            verbose=True
        )
    elif args.scheduler_type == 'MultiStepLR':
        # 计算学习率衰减的步长
        total_steps = args.epochs
        warmup_steps = args.warmup_epochs
        decay_steps = total_steps - warmup_steps
        
        # 计算每个epoch的学习率
        lr_schedule = []
        current_lr = args.lr
        min_lr = args.min_lr
        decay_factor = args.exp_lr_decay
        
        for step in range(decay_steps):
            lr = max(min_lr, current_lr * (decay_factor ** step))
            lr_schedule.append(lr)
        
        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[warmup_steps + i for i in range(len(lr_schedule))],
            gamma=decay_factor
        )
        print(f"Using MultiStepLR scheduler with warmup epochs: {warmup_steps}")
        print(f"Initial learning rate: {args.lr}, Minimum learning rate: {args.min_lr}")
        print(f"Decay factor: {args.exp_lr_decay}")

    elif args.scheduler_type == 'CosineAnnealingLR':
        # 支持余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=getattr(args, 'min_lr', 1e-6)
        )
        print(f"Using CosineAnnealingLR scheduler with T_max={args.epochs}, eta_min={getattr(args, 'min_lr', 1e-7)}")
   
    else:
        raise ValueError(f"Unsupported scheduler type: {args.scheduler_type}")
    
    # Load pretrained weights if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Starting from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")
            start_epoch = 1
    else:
        start_epoch = 1
    
    # Training loop
    print("Starting training...")
    
    # 打印MixUp配置信息
    if hasattr(args, 'mixup_alpha') and args.mixup_alpha > 0:
        print(f"MixUp data augmentation enabled with alpha={args.mixup_alpha}")
        print("MixUp will be applied during training to improve model robustness")
    else:
        print("MixUp data augmentation disabled")
    
    best_val_loss, best_val_distance = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        writer=writer,
        save_dir=args.save_dir,
        args=args
    )
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation distance: {best_val_distance:.4f}")
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ultrasound Image Landmark Heatmap Regression Training Script")
    
    """
    用于调试convnextv2模型对应的训练超参数
    """
    # 添加配置文件参数
    parser.add_argument('--config', type=str, default='config/densenet121_unet.yaml',
                        help='Path to configuration file')
    
    # 数据集参数
    parser.add_argument('--train_csv', type=str, default='inputs\label.csv',
                        help='Path to training CSV file')
    parser.add_argument('--train_dir', type=str, default=r'inputs\Labeled cases',
                        help='Directory with training images')
    parser.add_argument('--val_csv', type=str, default=r'inputs\val_label.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--val_dir', type=str, default=r'inputs\val_images',
                        help='Directory with validation images')
    
    # 热力图参数
    parser.add_argument('--heatmap_size', type=int, default=128,
                        help='Size of the heatmap')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Standard deviation for Gaussian heatmap')
    parser.add_argument('--num_keypoints', type=int, default=3,
                        help='Number of keypoints')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # 学习率调度器参数
    parser.add_argument('--scheduler_type', type=str, default='StepLR',
                        choices=['StepLR', 'ReduceLROnPlateau', 'MultiStepLR', 'CosineAnnealingLR'],
                        help='Type of learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                        help='Gamma factor for learning rate decay')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for ReduceLROnPlateau scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs for MultiStepLR scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Minimum learning rate for MultiStepLR scheduler')
    parser.add_argument('--exp_lr_decay', type=float, default=0.9,
                        help='Decay factor for MultiStepLR scheduler')
    
    # MixUp参数
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Alpha parameter for MixUp data augmentation')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='results_heatmap_train_only',
                        help='Directory to save results')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Interval to save model (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--timestamp', action='store_true',
                        help='Add timestamp to save directory to avoid overwriting previous results')
    parser.add_argument('--model_suffix', type=str, default='0',
                        help='Model name suffix to distinguish different training configurations')
    parser.add_argument('--model_name', type=str, default='smp_mitb',
                        help='Model name')
    parser.add_argument('--arch', type=str, default='pvt_v2_b1',
                        choices=['pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b4', 'convnextv2_tiny', 'convnextv2_base'],
                        help='Architecture for PVT or ConvNeXt models')


    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 使用配置文件更新参数
    args = update_args_from_config(args, config)
    
    # 运行主函数
    main(args)  