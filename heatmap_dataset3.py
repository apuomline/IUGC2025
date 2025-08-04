import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pda
import ast
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage
import random
import pandas as pd
class HeatmapLandmarkDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True, heatmap_size=64, sigma=2.0, cfg=None):
        """
        Ultrasound Image Landmark Heatmap Dataset
        
        Args:
            csv_file (str): Path to the CSV file with landmark coordinates
            img_dir (str): Directory with images
            transform (callable, optional): Transform to be applied on images
            train (bool): Whether this is training set
            heatmap_size (int): Size of the heatmap
            sigma (float): Standard deviation for Gaussian heatmap
            cfg (Config): Configuration object containing augmentation parameters
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.train = train
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.cfg = cfg
        print(f'===================use more data augmentation==================')
        # 基础图像变换
        self.transform = iaa.Sequential([
            iaa.Affine(
                rotate=(-5, 5),  # ROTATION: 5
                scale=(1 - 0.125, 1 + 0.125),  # SCALE: 0.125
                translate_px={"x": [-30, 30], "y": [-20, 20]},  # TRANSLATION_X/Y
                shear=(-0, 0)  # SHEAR: 0
            ),
            iaa.Multiply(mul=(1 - 0.6, 1 + 0.6)),  # MULTIPLY: 0.6
            iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 1.5))),  # BLUR_RATE: 0.1
            iaa.GammaContrast((0.3, 2.0)),  # CONTRAST_GAMMA_MIN/MAX
            iaa.ElasticTransformation(
                alpha=(0, 400),  # ELASTIC_TRANSFORM_ALPHA: 400
                sigma=30  # ELASTIC_TRANSFORM_SIGMA: 30
            ),
        ], random_order=False)

        # 特殊增强
        self.invert_transform = iaa.Invert(0.1)  # INVERT_RATE: 0.1
        self.cutout = iaa.Cutout(
            nb_iterations=(0, 1),  # CUTOUT_ITERATIONS: 1
            size=(0.04, 0.3),  # CUTOUT_SIZE_MIN/MAX
            squared=False
        )
        self.coarse_dropout = iaa.CoarseDropout(0.02, size_percent=0.08)
        self.gaussian_noise = iaa.AdditiveGaussianNoise(scale=(0, 0))  # GAUSSIAN_NOISE: 0

        # 图像预处理
        if transform is None:
            self.basic_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
        else:
            self.basic_transform = transform

    def __len__(self):
        return len(self.data)
    
    def generate_heatmap(self, center_x, center_y, height, width):
        """Generate a Gaussian heatmap for a single keypoint"""
        x = np.arange(0, width, 1, np.float32)
        y = np.arange(0, height, 1, np.float32)
        y = y[:, np.newaxis]
        
        # Calculate heatmap center
        x0 = center_x
        y0 = center_y
        
        # Generate Gaussian distribution heatmap
        heatmap = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
        return heatmap

    def __getitem__(self, index):
        # Get image filename
        row = self.data.iloc[index]
        filename = row['Filename']
        img_path = os.path.join(self.img_dir, filename)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Extract landmark coordinates
        ps1 = ast.literal_eval(row["PS1"])
        ps2 = ast.literal_eval(row["PS2"])
        aop = ast.literal_eval(row["FH1"])
        
        # 创建关键点对象
        keypoints = [ps1, ps2, aop]
        kps = KeypointsOnImage.from_xy_array(keypoints, shape=image.shape)
        
        if self.train:
            # 应用数据增强
            # 1. 基础变换
            image_aug, kps_aug = self.transform(image=image, keypoints=kps)
            
            # 2. 特殊增强
            if random.random() < 0.1:  # INVERT_RATE: 0.1
                image_aug = self.invert_transform(image=image_aug)
            
            if random.random() < 0.3:  # USE_SKEWED_SCALE_RATE: 0.3
                image_aug = self.cutout(image=image_aug)
            
            # 更新关键点
            keypoints = kps_aug.to_xy_array()
        else:
            keypoints = kps.to_xy_array()
        
        # 应用基础变换
        image = Image.fromarray(image_aug if self.train else image)
        image = self.basic_transform(image)
        
        # Original image size (assuming 512x512)
        img_width, img_height = 512, 512
        
        # Create heatmap labels - one channel per keypoint, 3 channels total
        heatmaps = np.zeros((3, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        # Scale coordinates to heatmap size
        scale_x = self.heatmap_size / img_width
        scale_y = self.heatmap_size / img_height
        
        # Generate heatmap for each keypoint
        for i, kp in enumerate(keypoints):
            # Convert coordinates to heatmap size
            x = int(kp[0] * scale_x)
            y = int(kp[1] * scale_y)
            
            # Ensure coordinates are within heatmap range
            x = max(0, min(x, self.heatmap_size - 1))
            y = max(0, min(y, self.heatmap_size - 1))
            
            # Generate heatmap
            heatmaps[i] = self.generate_heatmap(x, y, self.heatmap_size, self.heatmap_size)
        
        # Convert to tensor
        heatmaps = torch.from_numpy(heatmaps)
        
        # Save original coordinates (for evaluation metrics)
        landmarks = [
            keypoints[0][0] / img_width, keypoints[0][1] / img_height,
            keypoints[1][0] / img_width, keypoints[1][1] / img_height,
            keypoints[2][0] / img_width, keypoints[2][1] / img_height
        ]
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        
        return image, heatmaps, landmarks 
