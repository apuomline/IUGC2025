import torch
import os
from torchvision import transforms
from pvt_unet import pvt_unet
from convnextv2_unet import convnext_unet

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
    def __init__(self,inchannel=3,outchannel=3,heatmap_size=64):
        '''
        Initialize the model
        '''
        self.pvt_model = pvt_unet(3, 3, 64).cpu()
        self.convnext_model = convnext_unet(3,3,64).cpu()
        self.fusion_weights = [0.5, 0.5]  # [pvt_weight, convnext_weight]
        ###不需要加载预训练权重
    def load(self, path="./"):
        '''
        Load model weights
        '''
        # Load PVT model
        pvt_model_path = os.path.join(path, "pvt_b1_unet_6_flod1_best_val_distance_model.pth")
        print(f"Attempting to load PVT model from: {pvt_model_path}")
        if os.path.exists(pvt_model_path):
            try:
                checkpoint = torch.load(pvt_model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.pvt_model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"Successfully loaded pvt_model state_dict from checkpoint.")
                else:
                    self.pvt_model.load_state_dict(checkpoint)
                    print(f"Successfully loaded pvt_model state_dict directly.")
            except Exception as e:
                print(f"Failed to load PVT model file {pvt_model_path}: {e}")
        else:
            print(f"PVT model file not found at {pvt_model_path}. The model will use initial weights.")

        # Load ConvNeXt model - assuming a standard name, e.g., 'convnext_unet.pth'
        convnext_model_path = os.path.join(path, "convnextv2_tiny_unet_6_fold1_best_val_distance_model.pth")
        print(f"Attempting to load ConvNeXt model from: {convnext_model_path}")
        if os.path.exists(convnext_model_path):
            try:
                checkpoint = torch.load(convnext_model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.convnext_model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"Successfully loaded convnext_model state_dict from checkpoint.")
                else:
                    self.convnext_model.load_state_dict(checkpoint)
                    print(f"Successfully loaded convnext_model state_dict directly.")
            except Exception as e:
                print(f"Failed to load ConvNeXt model file {convnext_model_path}: {e}")
        else:
            print(f"ConvNeXt model file not found at {convnext_model_path}. The model will use initial weights.")
        
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
        self.pvt_model.eval()
        self.convnext_model.eval()

        width, height = X.size
        # Apply the same transformations as during training
        tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        image = tf(X).unsqueeze(0)  # Add batch dimension (1, 3, H, W)

        with torch.no_grad():
            # Get predictions from both models
            heatmaps_pvt = self.pvt_model(image)
            print(f'heatmaps_pvt.shape:{heatmaps_pvt.shape}')
            heatmaps_convnext = self.convnext_model(image)
            print(f'heatmaps_convnext.shape:{heatmaps_convnext.shape}')
        

            # --- Weighted Fusion ---
            heatmaps = (self.fusion_weights[0] * heatmaps_pvt + self.fusion_weights[1] * heatmaps_convnext)
            
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

    