import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import csv
from torch.utils.data import Dataset, DataLoader
from heatmap_utils_DARK import extract_coordinates
from inference_models.pvt_unet_sup import pvt_unet_sup
from inference_models.pvt_unet import pvt_unet
from inference_models.convnextv2_unet import convnext_unet

class UnlabeledImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        self.img_list = []
        for root, _, files in os.walk(img_dir):
            for f in files:
                if os.path.splitext(f)[-1].lower() in self.img_exts:
                    rel_dir = os.path.relpath(root, img_dir)
                    rel_path = os.path.join(rel_dir, f) if rel_dir != '.' else f
                    self.img_list.append(rel_path)
        self.img_list.sort()
        self.transform = transform
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        rel_img_path = self.img_list[idx]
        img_path = os.path.join(self.img_dir, rel_img_path)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, rel_img_path

def load_model_weights(model, weight_path, device):
    """加载模型权重"""
    if weight_path and os.path.isfile(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        # 兼容两种权重保存格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model weights from {weight_path}")
    else:
        raise FileNotFoundError(f"Weight file not found at {weight_path}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 根据模型名称初始化模型
    if args.model_name == 'pvt_convnext_unet':
        print("Using pvt_convnext_unet fusion model.")
        pvt_model1 = pvt_unet_sup(3, 3, heatmap_size=args.heatmap_size,arch='pvt_v2_b1',
        deep_supervision_layers=[2,3],fusion_mode='concat').to(device)

        pvt_model2 = pvt_unet(3,3,64,arch='pvt_v2_b1').to(device)

        convnext_model = convnext_unet(3, 3, heatmap_size=args.heatmap_size).to(device)
        
        load_model_weights(pvt_model1, args.pvt_model_weight1, device)
        load_model_weights(pvt_model2, args.pvt_model_weight2, device)
        load_model_weights(convnext_model, args.convnext_model_weight, device)
        
        pvt_model1.eval()
        pvt_model2.eval()
        convnext_model.eval()
        model = None  # 融合模型无单一实体
  
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    # 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    # 数据加载
    dataset = UnlabeledImageDataset(args.img_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    results = []
    with torch.no_grad():
        for imgs, img_names in dataloader:
            imgs = imgs.to(device)
            
            # 模型推理
            if args.model_name == 'pvt_convnext_unet':
                heatmaps_pvt1 = pvt_model1(imgs)
                heatmaps_pvt2 = pvt_model2(imgs)
                heatmaps_convnext = convnext_model(imgs)
                outputs = (args.fusion_weight1 * heatmaps_pvt1 + args.fusion_weight2 * heatmaps_pvt2    
                            + args.fusion_weight3 * heatmaps_convnext)
            else:
                outputs = model(imgs)

            # 坐标提取（含DARK后处理）
            pred_coords = extract_coordinates(outputs, use_dark=args.use_dark).cpu().numpy()
            # 计算每个关键点的最大置信度
            heatmaps = outputs.cpu().numpy()  # (B, 3, H, W)
            confidences = heatmaps.max(axis=(2, 3))  # (B, 3)
            img_h, img_w = 512, 512  # 与 transforms.Resize((512, 512)) 保持一致
            threshold = args.conf_threshold  # 置信度阈值

            for i in range(imgs.size(0)):
                row = [img_names[i]]
                for j in range(3):
                    if confidences[i, j] >= threshold:
                        if args.use_dark:
                            x = float(pred_coords[i, j, 0]) * img_w
                            y = float(pred_coords[i, j, 1]) * img_h
                        else:
                            x = float(pred_coords[i, j*2]) * img_w
                            y = float(pred_coords[i, j*2+1]) * img_h
                        row.extend([x, y])
                    else:
                        row.extend([None, None])  # 置信度低于阈值，输出None
                results.append(row)
                print(f"{img_names[i]}: {row[1:]}")

    # 结果写入CSV
    header = ['Filename', 'PS1', 'PS2', 'FH1']
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results:
            filename = row[0]
            keypoints = []
            for j in range(3):
                x, y = row[1 + j * 2], row[2 + j * 2]
                if x is not None and y is not None:
                    keypoints.append(f"({int(round(x))}, {int(round(y))})")
                else:
                    keypoints.append("None")
            writer.writerow([filename] + keypoints)
    print(f"已将伪标签写入 {args.out_csv}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='未标注图像批量推理生成伪标签（支持融合模型与DARK）')
    
    # 路径参数
    parser.add_argument('--img_dir', type=str, default='pse_unmatched_37', help='未标注图片文件夹')
    parser.add_argument('--out_csv', type=str, default='pseudo_split_outputs/pse_unmatched_37_threshold_50.csv', help='输出csv文件路径')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='pvt_convnext_unet', choices=['pvt_unet', 'convnextv2_tiny', 'pvt_convnext_unet'], help='模型名称')
    parser.add_argument('--heatmap_size', type=int, default=64, help='热力图尺寸')
    parser.add_argument('--batch_size', type=int, default=8, help='推理batch size')
    

    parser.add_argument('--pvt_model_weight1', default='trained_model_pths/pvt_unet_1_re_layer3_4_flod1_moda_best_val_distance_model.pth', help='pvt_convnext_unet融合模型中pvt的权重路径')
    parser.add_argument('--pvt_model_weight2', default='trained_model_pths/pvt_unet_1_new_re4_flod1_best_val_distance_model.pth', help='pvt_convnext_unet融合模型中pvt的权重路径')
    parser.add_argument('--convnext_model_weight', default='trained_model_pths/convnextv2_tiny_unet_6_fold1_best_val_distance_model.pth', help='pvt_convnext_unet融合模型中convnext的权重路径')
    parser.add_argument('--fusion_weight1', type=float, default=0.3, help='pvt_convnext_unet融合模型中pvt1的权重')
    parser.add_argument('--fusion_weight2', type=float, default=0.4, help='pvt_convnext_unet融合模型中pvt2的权重')
    parser.add_argument('--fusion_weight3', type=float, default=0.3, help='pvt_convnext_unet融合模型中convnext的权重')
    parser.add_argument('--conf_threshold', type=float, default=0.50, help='关键点置信度阈值')
    parser.add_argument('--use_dark', default=True, help='是否使用DARK后处理')

    args = parser.parse_args()
    main(args)  