import numpy as np
from os.path import isfile
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import torch
import os
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from pvt_unet import PVT_v2_UNet


# Coordinate extraction function
def extract_coordinates(heatmaps, original_img_size=512):
    """
    Extract keypoint coordinates from heatmaps
    
    Args:
        heatmaps (torch.Tensor): Predicted heatmaps [batch_size, num_keypoints, height, width]
        original_img_size (int): Original image size
    
    Returns:
        torch.Tensor: Extracted coordinates [batch_size, num_keypoints*2]
    """
    batch_size, num_keypoints, height, width = heatmaps.shape
    
    # Find the maximum response position in each heatmap
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_keypoints, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    
    # Convert to 2D coordinates
    y_coords = torch.div(max_indices, width, rounding_mode='floor').float() / height
    x_coords = (max_indices % width).float() / width
    
    # Combine coordinates
    coords = torch.zeros(batch_size, num_keypoints * 2, device=heatmaps.device)
    for i in range(num_keypoints):
        coords[:, i*2] = x_coords[:, i]
        coords[:, i*2+1] = y_coords[:, i]
    
    return coords

class model:
    def __init__(self):
        '''
        Initialize the model
        '''
        # self.model = HeatmapUNet(num_keypoints=3, heatmap_size=64).cpu()
        self.model = PVT_v2_UNet(3,3,64).cpu()
        ###不需要加载预训练权重
    def load(self, path="./"):
        '''
        Load model weights
        '''
        # Try multiple possible model filenames
        possible_model_paths = [
            os.path.join(path, "model_weight.pth"),
            os.path.join(path, "pvt_b1_unet_6_flod1_best_val_distance_model.pth"),
            os.path.join(path, "heatmap_model.pth")
        ]
        
        for model_path in possible_model_paths:
            if os.path.exists(model_path):
                print(f"Loading model: {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location="cpu")
                    # Check if it's a checkpoint containing multiple components
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        # Use the model state dict instead of the entire checkpoint
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                        print(f"Successfully loaded model_state_dict from checkpoint")
                    else:
                        # Try to load directly, assuming it's a simple model state dictionary
                        self.model.load_state_dict(checkpoint)
                    return self
                except Exception as e:
                    print(f"Failed to load model file {model_path}: {e}")
                    continue
        
        # If no model file is found, try loading the default file
        default_model_path = os.path.join(path, "unet_heatmap.pth")
        print(f"No model file found, trying to load from default path: {default_model_path}")
        try:
            checkpoint = torch.load(default_model_path, map_location="cpu")
            # Check if it's a checkpoint containing multiple components
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Use the model state dict instead of the entire checkpoint
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Successfully loaded model_state_dict from checkpoint")
            else:
                # Try to load directly, assuming it's a simple model state dictionary
                self.model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Failed to load model file: {e}")
            print("Please ensure the model file exists and is named 'unet_heatmap.pth', 'model.pth', or 'heatmap_model.pth'")
        
        return self
    
    def predict(self, X):
        '''
        Prediction function, input an image, output coordinates
        
        Args:
            X: PIL.Image object, the input image
 
        
        Returns:
            coords: numpy array of shape (6,), predicted keypoint coordinates
                    Format: [x1, y1, x2, y2, x3, y3] where:
                    - (x1, y1) is the coordinate of the first keypoint
                    - (x2, y2) is the coordinate of the second keypoint
                    - (x3, y3) is the coordinate of the third keypoint
                    The coordinates should be in the pixel space of the original input image.
        '''
        self.model.eval()

        width, height = X.size
        # Apply the same transformations as during training
        tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        image = tf(X).unsqueeze(0)  # Add batch dimension (1, 3, H, W)

        with torch.no_grad():
            # Original image prediction
            heatmaps = self.model(image)

            # Flipped image prediction with test-time augmentation (TTA)
            flipped_image = TF.hflip(image)
            flipped_heatmaps = self.model(flipped_image)
            flipped_heatmaps = TF.hflip(flipped_heatmaps)

            # Assuming keypoints 0 and 1 are a left/right pair, and keypoint 2 is central.
            # The order of keypoints must be swapped to match the unflipped image.
            flip_indices = torch.tensor([1, 0, 2], dtype=torch.long, device=flipped_heatmaps.device)
            flipped_heatmaps = flipped_heatmaps[:, flip_indices, :, :]
            
            # Average heatmaps for a more robust prediction
            heatmaps = (heatmaps + flipped_heatmaps) / 2.0
            
            # Extract coordinates from heatmaps
            coords = extract_coordinates(heatmaps)

        # Convert to numpy array
        coords = coords.squeeze(0).cpu().numpy()

        # Convert normalized coordinates back to original image size
        coords[::2] *= width   # x coordinates
        coords[1::2] *= height  # y coordinates

        return coords
    
    def save(self, path="./"):
        '''
        Save model weights
        '''
        pass

# IMPORTANT: Input and Output Specification
# ----------------------------------------
# Input (X):   
#   - A PIL.Image object (from PIL.Image.open)
#   - Represents an RGB image 
# 
# Output (coords):
#   - A numpy array of shape (6,) containing 3 keypoint coordinates
#   - Format: [x1, y1, x2, y2, x3, y3]
#   - Coordinates must be in the pixel space of the original input image
#   - The coordinates should represent the exact locations of the detected keypoints


if __name__=='__main__':
    x = torch.rand(1,3,512,512)
    x_image = transforms.ToPILImage()(x.squeeze(0))
    net = model().load()
    out = net.predict(x_image)
    print(f'out:{out}')

    