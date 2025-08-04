import json
import math
import traceback
from collections import defaultdict

import imgaug.augmenters
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import random
import imgaug.augmenters as iaa
from einops import rearrange
from imgaug import KeypointsOnImage

from tqdm import tqdm
import torch

# 数据集参数
DATASET_PARAMS = {
    'IMG_SIZE': (512, 512),
    'NUMBER_KEY_POINTS': 3,
    'USE_ALL_RCNN_IMAGES': False
}

# mu = 0.45367337891144466
# sigma = 0.28753600293497544

# Updated RCNN style imgs
# mu = 0.4469038915295547
# sigma = 0.27797534502741916


# mu = 0.5
# sigma = 0.5


# mu = 0.45
# sigma = 0.3


def simulate_x_ray_artefacts(image):
    # randomly add noise or multiply by 0-0.1 to a random slice of the image if
    input_dtype = image.dtype
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32)

    # randomly decide height or width
    axis = random.choice([0, 1])

    # decide size of slice (25,50,75 pixels)
    slice_size = random.choice([25, 50, 75, 100, 125])
    # choose start_idx
    start_idx = random.randint(0, image.shape[axis] - slice_size)
    end_idx = start_idx + slice_size
    if axis == 0:  # height
        slice_mask = np.s_[start_idx:end_idx, :]
    else:  # width
        slice_mask = np.s_[:, start_idx:end_idx]
    # Generate noise or scale factor
    if random.choice([True, False]):  # Apply noise
        np.add(image[slice_mask], np.random.normal(0, 15, image[slice_mask].shape), out=image[slice_mask])
    else:  # Apply scaling
        np.multiply(image[slice_mask], np.random.uniform(0.5, 1.5), out=image[slice_mask])

    image.clip(min=0, max=255, out=image)
    return image.astype(input_dtype)



def normalise(x, method="none"):
    # 0-1 (none), -1-1 (mu=0.5, sigma=0.5), dataset mean and std
    if method == "none":
        return x
    elif method == "mu=0.5,sig=0.5":
        return (x - 0.5) / 0.5
    elif method == "dataset":
        mu = 0.4469038915295547
        sigma = 0.27797534502741916
        return (x - mu) / sigma
    else:
        return x


def renormalise(x, img=True, method="none"):
    # 0-1 (none), -1-1 (mu=0.5, sigma=0.5), dataset mean and std
    if method == "mu=0.5,sig=0.5":
        norm = (x * 0.5) + 0.5
    elif method == "dataset":
        mu = 0.4469038915295547
        sigma = 0.27797534502741916
        norm = (x * sigma) + mu
    else:
        norm = x
    if img and norm.mean() < 3:
        return norm * 255
    return norm


def pad_box_to_multiple_of_32(box: imgaug.augmentables.bbs.BoundingBox, down_sampled_width=512):
    # height/width = aspect ratio
    x1, x2, y1, y2 = math.floor(box.x1), math.ceil(box.x2), math.floor(box.y1), math.ceil(box.y2)
    height = y2 - y1
    width = x2 - x1
    new_width = width + (32 - width % 32)

    aspect_ratio = height / width
    new_downsampled_height = down_sampled_width * aspect_ratio
    padded_downsampled_height = new_downsampled_height + (32 - new_downsampled_height % 32) + 1
    new_aspect_ratio = padded_downsampled_height / down_sampled_width

    new_height = new_aspect_ratio * width
    new_height = new_height + (32 - new_height % 32)
    new_y1 = y1 - (new_height - height) / 2
    new_y2 = y2 + (new_height - height) / 2
    new_x1 = x1 - (new_width - width) / 2
    new_x2 = x2 + (new_width - width) / 2

    return imgaug.augmentables.bbs.BoundingBox(x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2)


def make_full_box_at_least_other_boxes(boxes, labels, image_name=""):
    if image_name == "396":
        print("before", list(zip(boxes, labels)))
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if label != 1:
            continue
        x1, y1, x2, y2 = box
        for other_box, other_label in zip(boxes, labels):
            # if other_label != 1:
            #     continue
            if torch.equal(box, other_box):
                continue
            other_x1, other_y1, other_x2, other_y2 = other_box
            x1 = min(x1, other_x1)
            y1 = min(y1, other_y1)
            x2 = max(x2, other_x2)
            y2 = max(y2, other_y2)
        boxes[i] = torch.Tensor([x1, y1, x2, y2])
    if image_name == "396":
        print("after", list(zip(boxes, labels)))

    return boxes


def rcnn_save_image(batch, boxes, labels, resize_dir):
    image_name = batch["name"][0]

    boxes = make_full_box_at_least_other_boxes(boxes, labels, image_name)
    cached_image_name_base = f"{resize_dir}/{image_name}"

    annotations_names = []
    if batch["landmarks_all_annotators"].shape[0] == 1:
        annotations_names.append(f"{cached_image_name_base}_annotations")
    else:
        for annotation_dir in [f"annotation_{i}" for i in range(batch["landmarks_all_annotators"].shape[0])]:
            cached_annotation_name = f"{cached_image_name_base}_{annotation_dir.split('/')[-1]}"
            annotations_names.append(cached_annotation_name)

    cached_meta_name = f"{cached_image_name_base}_meta.json"

    image_batch = rearrange(batch["x"], 'b h w c -> b c h w').to(torch.float32).numpy()
    image_original = batch["image_original"].numpy()[0]
    landmarks_original = batch["landmarks_original"].numpy().reshape(-1, DATASET_PARAMS['NUMBER_KEY_POINTS'], 2).astype(
        np.float64)

    landmarks_per_image_mask = [None, [0, 3, 16, 18, 23, 24, 26, 27, 28],
                                [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                                [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 29, 30, 31, 32,
                                 35,
                                 36, 37],
                                [1, 2, 33, 34, 51, 52]]
    resizer_to_padded_original = imgaug.augmenters.Resize(
        {"height": image_original.shape[0], "width": image_original.shape[1]})
    crop_to_removed_padding = imgaug.augmenters.CropToFixedSize(height=batch["img_pre_padded_resolution"][0, 0],
                                                                width=batch["img_pre_padded_resolution"][0, 1],
                                                                position="right-bottom")

    resize_back_down_1_resize = iaa.Resize({"height": DATASET_PARAMS['IMG_SIZE'][0], "width": "keep-aspect-ratio"})
    resize_back_down_2_crop_final = iaa.CropToMultiplesOf(32, 32, position="right-bottom")
    shifts = {}
    scale_factors = {}
    box = None
    try:
        # extract boxed region from image
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if not DATASET_PARAMS['USE_ALL_RCNN_IMAGES'] and label != 1:
                continue
            label = int(label.item())
            image_original_copy = image_original.copy()
            x1, y1, x2, y2 = box
            y1 -= 8
            y2 += 8

            # convert box to kp objects
            box = [imgaug.augmentables.bbs.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, )]

            image_padded, box = crop_to_removed_padding(image=image_batch[0, 0], bounding_boxes=box)
            image_resized, box = resizer_to_padded_original(image=image_padded, bounding_boxes=box)

            box = pad_box_to_multiple_of_32(box[0], down_sampled_width=DATASET_PARAMS['IMG_SIZE'][1])

            # take the crop of the box
            x1 = max(0, math.floor(box.x1))
            y1 = max(0, math.floor(box.y1))
            x2 = min(math.ceil(box.x2), image_original_copy.shape[1])
            y2 = min(math.ceil(box.y2), image_original_copy.shape[0])
            cropped_image = image_original_copy[y1:y2, x1:x2]

            # calculate shifts
            label_landmarks = landmarks_original.copy()
            if label - 1 > 0:
                label_landmarks = label_landmarks[:, landmarks_per_image_mask[label - 1]]
            label_landmarks = label_landmarks.reshape(-1, 2)
            label_landmarks[:, 0] -= x1
            label_landmarks[:, 1] -= y1

            label_landmarks = KeypointsOnImage.from_xy_array(label_landmarks, shape=cropped_image.shape)

            shifts[label] = [x1, y1]
            box.x1, box.y1, box.x2, box.y2 = box.x1 - x1, box.y1 - y1, box.x2 - x1, box.y2 - y1
            original_middle_of_landmark = [(box.x2 + box.x1) / 2, (box.y1 + box.y2) / 2]

            # downsize to config width
            cropped_image, box_downsized = resize_back_down_1_resize(image=cropped_image, bounding_boxes=box)
            # pad back to multiple of 32 from bottom right
            cropped_image, box_downsized = resize_back_down_2_crop_final(image=cropped_image,
                                                                         bounding_boxes=box_downsized)

            # calculate scale factor information
            downsized_middle_of_box_landmark = np.array(
                [box_downsized.x1 + box_downsized.x2, box_downsized.y1 + box_downsized.y2]) / 2

            scale_factors[label] = original_middle_of_landmark / downsized_middle_of_box_landmark
            scale_factors[label] = scale_factors[label].tolist()

            # save image
            cropped_image_name = f"{cached_image_name_base}"
            if label > 1:
                cropped_image_name += f"_{label}"
            cv2.imwrite(f"{cropped_image_name}.bmp", cropped_image)

            # process landmarks
            label_landmarks = resize_back_down_1_resize(keypoints=label_landmarks)
            label_landmarks = resize_back_down_2_crop_final(keypoints=label_landmarks)

            if landmarks_per_image_mask[label - 1] is None:
                continue

            # save landmarks
            for i, annotation_name in enumerate(annotations_names):
                if i == 0:
                    landmarks = label_landmarks.to_xy_array()
                else:
                    landmarks = batch["landmarks_all_annotators"].numpy()[i].reshape(-1, DATASET_PARAMS['NUMBER_KEY_POINTS'], 2)
                    landmarks = landmarks[:, landmarks_per_image_mask[label - 1]]
                    landmarks = landmarks.reshape(-1, 2)
                    landmarks[:, 0] -= x1
                    landmarks[:, 1] -= y1
                    landmarks = KeypointsOnImage.from_xy_array(landmarks, shape=cropped_image.shape)
                    landmarks = resize_back_down_1_resize(keypoints=landmarks)
                    landmarks = resize_back_down_2_crop_final(keypoints=landmarks)
                    landmarks = landmarks.to_xy_array()

                np.save(f"{annotation_name}.npy", landmarks)

            # save meta
            meta = {
                "shifts": shifts,
                "scale_factors": scale_factors,
                "landmarks_per_image_mask": landmarks_per_image_mask,
                "label": label,
            }
            with open(cached_meta_name, "w") as f:
                json.dump(meta, f)

    except Exception as e:
        print(f"Error processing image {image_name}: {str(e)}")
        traceback.print_exc()
        return None


def colour_augmentation(image):
    # https://github.com/runnanchen/Anatomic-Landmark-Detection/blob/main/data_enhancement.py
    # random_factor = np.random.randint(0, 50) / 10.
    # color_image = iaa.pillike.enhance_color(image, random_factor)

    # random_factor = np.random.randint(3, 10) / 10.  # was 1-2.1
    # random_factor = 0.05
    # brightness_image = iaa.pillike.enhance_brightness(image, random_factor)
    # return brightness_image
    # return image
    # random_factor = np.random.randint(5, 15) / 10.  # 随机因1子
    # contrast_image = iaa.pillike.enhance_contrast(image, random_factor)
    # return contrast_image
    random_factor = np.random.randint(5, 20) / 10.
    return iaa.pillike.enhance_sharpness(image, random_factor)




def get_coordinates_from_heatmap_boolean_mean(heatmap: torch.tensor, threshold=0.5):
    scale_factor = torch.amax(heatmap, dim=(2, 3), keepdim=True)
    scaled_output = heatmap / scale_factor
    mask = scaled_output >= threshold
    indices = torch.nonzero(mask).float()
    # take the mean of the indices across channel dim
    output_coordinates = torch.zeros((heatmap.shape[0], heatmap.shape[1], 2), device=heatmap.device)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if torch.sum(mask[i, j]) > 0:
                current_indices = indices[(indices[:, 0] == i) & (indices[:, 1] == j)]
                output_coordinates[i, j] = torch.mean(current_indices[:, 2:].float(), dim=0)
            else:
                output_coordinates[i, j] = torch.tensor([-1, -1], device=heatmap.device)
    return output_coordinates


def get_coordinates_from_heatmap(heatmap: torch.tensor, k=1, threshold=0.5):
    # if k > 10:
    #     return get_coordinates_from_heatmap_boolean_mean(heatmap, 0.5)
    # heatmap is in the format [B, C, H, W]
    # coordinates are in the format [B, C, 2]
    # get the value and flattened index of the heatmap over each channel
    b, c, _, _ = heatmap.shape
    max_values, max_indices = torch.topk(heatmap.view(heatmap.shape[0], heatmap.shape[1], -1), dim=2, k=k)

    # add a spatial dimension
    max_values = max_values.unsqueeze(-1)
    max_indices = max_indices.unsqueeze(-1)
    # convert the flattened index to 2D coordinates
    coordinates = torch.cat([max_indices % heatmap.shape[3], max_indices // heatmap.shape[3]], dim=-1)
    if torch.isclose(torch.sum(max_values, dtype=max_values.dtype),
                     torch.zeros((1,), dtype=max_values.dtype, device=max_values.device)):
        coords = torch.zeros((b, c, 2), device=coordinates.device) - 1
        return coords.float()

    # Softmax the values across the k channel
    # max_values = torch.nn.functional.softmax(max_values, dim=2)
    max_values = max_values / torch.sum(max_values, dim=2, keepdim=True)
    # average the coordinates of the top k values by their value
    coordinates = torch.sum(coordinates * max_values, dim=2)

    return coordinates.float()


def invert_preprocessing(landmarks, invert_aug, image=None):
    landmarks[:, 0] = landmarks[:, 0] * invert_aug["pre-resize shape"][1] / image.shape[1]
    landmarks[:, 1] = landmarks[:, 1] * invert_aug["pre-resize shape"][0] / image.shape[0]
    landmarks[:, 0] += invert_aug["shift"][0]
    landmarks[:, 1] += invert_aug["shift"][1]
    return landmarks


def test_inverse_preprocessing():
    labels = pd.read_csv(
        "../datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/labels.csv")
    for image_file in os.listdir(
            "../datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/images"):
        # for image_file in ["396.bmp"]:
        image = cv2.imread(
            f"datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/images/{image_file}",
            cv2.IMREAD_GRAYSCALE)
        landmarks = labels.loc[labels["image file"] == image_file.split("/")[-1]].iloc[0, 2:].values.astype(
            'float').reshape(-1, 2)
        preprocessed_img, preprocessed_landmarks, ops = handle_miccai_preprocessing(image, landmarks.copy())
        inverted_landmarks = invert_preprocessing(preprocessed_landmarks.copy(), ops, preprocessed_img)
        # plot_landmarks_on_image(image, inverted_landmarks, name=f"{image_file} inverted", show=True)
        plot_landmarks_on_image(preprocessed_img, preprocessed_landmarks, name=f"{image_file} preprocessed", show=True)

        assert np.allclose(landmarks, inverted_landmarks, atol=1)


def test_preprocess_just_landmarks():
    cfg = Config()
    cfg.DATASET.IMG_SIZE = [800, 736]
    cfg.DATASET.CHALLENGE_PREPROCESSING = True
    preprocessing = LandmarkPreprocessing(cfg)
    labels = pd.read_csv(
        "../datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/labels.csv")
    # for image_file in os.listdir(
    #         "datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/images"):

    for image_file in ["443.bmp", "521.bmp", "001.bmp", "519.bmp"]:
        image = cv2.imread(
            f"../datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/images/{image_file}",
            cv2.IMREAD_GRAYSCALE)
        landmarks = labels.loc[labels["image file"] == image_file.split("/")[-1]].iloc[0, 2:].values.astype(
            'float').reshape(-1, 2)

        preprocessed_img, preprocessed_landmarks, ops = preprocessing(image, landmarks.copy())
        alternative_landmarks = preprocessing.preprocess_just_landmarks(landmarks.copy(), ops)
        plot_landmarks_on_image(image, landmarks, name=f"original res {image_file}", show=True)
        plot_landmarks_on_image(preprocessed_img, alternative_landmarks,
                                name=f"{image_file} preprocessed - just landmark", show=True)

        assert np.allclose(preprocessed_landmarks, alternative_landmarks, atol=1)


class LandmarkPreprocessing:
    def __init__(self):
        self.img_size = DATASET_PARAMS['IMG_SIZE']
        self.number_key_points = DATASET_PARAMS['NUMBER_KEY_POINTS']
        
    def __call__(self, image, landmarks=None):
        if landmarks is not None:
            landmarks = landmarks.reshape(-1, self.number_key_points, 2)
            landmarks = KeypointsOnImage.from_xy_array(landmarks, shape=image.shape)
            image, landmarks = self.preprocess(image, landmarks)
            landmarks = landmarks.to_xy_array()
            landmarks = landmarks.reshape(-1, self.number_key_points * 2)
            return image, landmarks
        else:
            image = self.preprocess(image)
            return image

    def preprocess(self, image, landmarks=None):
        # 基础预处理
        image = image.astype(np.float32)
        image = image / 255.0
        
        # 调整大小
        if landmarks is not None:
            image, landmarks = iaa.Resize({"height": self.img_size[0], "width": self.img_size[1]})(image=image, keypoints=landmarks)
        else:
            image = iaa.Resize({"height": self.img_size[0], "width": self.img_size[1]})(image=image)
            
        return image, landmarks if landmarks is not None else image

    def preprocess_just_landmarks(self, landmarks, invert_aug):
        landmarks = landmarks.reshape(-1, self.number_key_points, 2)
        landmarks = KeypointsOnImage.from_xy_array(landmarks, shape=(self.img_size[0], self.img_size[1]))
        landmarks = self.preprocess(None, landmarks)[1]
        landmarks = landmarks.to_xy_array()
        landmarks = landmarks.reshape(-1, self.number_key_points * 2)
        return landmarks

    def invert(self, landmarks, invert_aug):
        landmarks = landmarks.reshape(-1, self.number_key_points, 2)
        landmarks = KeypointsOnImage.from_xy_array(landmarks, shape=(self.img_size[0], self.img_size[1]))
        landmarks = invert_aug(keypoints=landmarks)
        landmarks = landmarks.to_xy_array()
        landmarks = landmarks.reshape(-1, self.number_key_points * 2)
        return landmarks

    def __getstate__(self):
        return {
            'img_size': self.img_size,
            'number_key_points': self.number_key_points
        }

    def __setstate__(self, state):
        self.img_size = state['img_size']
        self.number_key_points = state['number_key_points']


def handle_miccai_preprocessing(image, landmarks=None, final_shape=(704, 704)):
    if landmarks is None:
        landmarks = np.array([[0, 0]])
    shift_operations = [0, 0]

    # ----------- cut from outside constinuous areas -----------
    columns = get_first_noncontinuous_column(image, threshold=20)
    rows = get_first_noncontinuous_row(image, threshold=15)
    image = image[rows[0]:rows[1], columns[0]:columns[1]]
    # update keypoints
    landmarks[:, 0] -= columns[0]
    landmarks[:, 1] -= rows[0]

    shift_operations[0] += rows[0]
    shift_operations[1] += columns[0]

    # ----- pad to be centered around averaged landmark 3 -----
    kps = KeypointsOnImage.from_xy_array(
        landmarks, shape=image.shape
    )
    if image.shape[0] > image.shape[1] + 200:
        center = 0.34 * image.shape[1], 0.49 * image.shape[0]
    elif image.shape[0] + 200 < image.shape[1]:
        center = 0.40 * image.shape[1], 0.51 * image.shape[0]
    else:
        center = 0.39 * image.shape[1], 0.40 * image.shape[0]
    add_padding = iaa.PadToFixedSize(int(image.shape[1] - center[0]) * 2, int(image.shape[0] - center[1]) * 2,
                                     position="left-top")
    image_padded, kps = add_padding(image=image, keypoints=kps)
    shift_operations[0] += image.shape[0] - image_padded.shape[0]
    shift_operations[1] += image.shape[1] - image_padded.shape[1]

    # ----- crop back into image, keeping most of height and majority width -----
    crop = iaa.CropToFixedSize(int(image_padded.shape[1] * 3 / 4), int(image_padded.shape[0] * 4.7 / 5),
                               position="left-top")
    image_cropped, kps = crop(image=image_padded, keypoints=kps)
    shift_operations[0] += image_padded.shape[0] - image_cropped.shape[0]
    shift_operations[1] += image_padded.shape[1] - image_cropped.shape[1]

    # -------------- pad to head shape aspect ratio --------------
    # plot_landmarks_on_image(image_cropped, kps.to_xy_array(), name="pre-resize- pre pad", show=True)
    IMG_SIZE = [800, 640]
    pad_to_aspect_ratio = iaa.PadToAspectRatio(IMG_SIZE[1] / IMG_SIZE[0], position="left-top")

    padded_to_aspect_ratio, kps = pad_to_aspect_ratio(image=image_cropped, keypoints=kps)
    shift_operations[0] += image_cropped.shape[0] - padded_to_aspect_ratio.shape[0]
    shift_operations[1] += image_cropped.shape[1] - padded_to_aspect_ratio.shape[1]

    # final_image = padded_to_aspect_ratio

    # -------------- crop off top of head which is not used --------------

    # plot_landmarks_on_image(padded_to_aspect_ratio, kps.to_xy_array(), name="pre-resize - post pad, pre crop",
    #                         show=True)
    crop_to_aspect_ratio = iaa.CropToAspectRatio(final_shape[1] / final_shape[0], position="center-top")

    cropped_to_aspect_ratio, kps = crop_to_aspect_ratio(image=padded_to_aspect_ratio, keypoints=kps)
    shift_operations[0] += padded_to_aspect_ratio.shape[0] - cropped_to_aspect_ratio.shape[0]
    shift_operations[1] += padded_to_aspect_ratio.shape[1] - cropped_to_aspect_ratio.shape[1]
    final_image = cropped_to_aspect_ratio
    # --------------------------- resize ---------------------------
    resize = iaa.Resize(
        {"width": final_shape[1], "height": final_shape[0]}
    )
    resized_image, kps = resize(image=final_image, keypoints=kps)
    return resized_image, kps.to_xy_array(), {"shift": shift_operations[::-1],
                                              "pre-resize shape": final_image.shape}


def get_first_noncontinuous_column(image: np.array, threshold=0.1):
    output = []

    for i in range(image.shape[1]):
        if np.std(image[:int(image.shape[0] * 0.75), i]) > threshold:
            # print("col_left", i, np.mean(image[:int(image.shape[0] * 0.75), i]),
            #       np.std(image[:int(image.shape[0] * 0.75), i]))
            output.append(i)
            break
    # col_right = 0
    # for i in range(image.shape[1] - 1, -1, -1):
    #
    #     std = np.std(image[:int(image.shape[0] * 0.8), i])
    #     if std > threshold:
    #         print("col_right", i, np.mean(image[:int(image.shape[0] * 0.8), i]), std)
    #         col_right = i
    #         break
    # image = image[:, output[0]:col_right]
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(image[:, :-25], (5, 5), 3), 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 27, 4)
    edges = cv2.Canny(thresh, 50, 100, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 3, 0.6, 80, minLineLength=50, maxLineGap=10)

    rightmost = -1
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # if it is vertical
        if abs(x1 - x2) < 20:
            # if it is not 30 pixels from the maximum right

            # if it is the current rightmost
            if x2 < image.shape[1] - 20 and x2 > rightmost:
                if np.mean(image[:int(image.shape[0] * 0.75), x2 + 20:]) > 245:
                    continue
                height = (y1, y2)
                rightmost = x2
    output.append(rightmost + 150)
    # plt.scatter([rightmost], [height[0]], c='r', s=3)
    # plt.scatter([rightmost], [height[1]], c='r', s=3)
    # plt.imshow(image)
    # plt.show()

    return output


def get_first_noncontinuous_row(image: np.array, threshold=0.1):
    output = []

    for i in range(image.shape[0]):
        if np.std(image[i, :]) > threshold:
            # print("row_top", i, np.mean(image[i, :]), np.std(image[i, :]))
            output.append(i)
            break

    for i in range(image.shape[0] - 1, -1, -1):
        if np.std(image[i, :]) > threshold:
            # print("row_bottom", i, np.mean(image[i, :]), np.std(image[i, :]))
            output.append(i + 50)
            break
    return output



def plot_landmarks_on_image(image, image_label, name="", save=False, show=False):
    # clear the plot
    plt.clf()
    plt.imshow(image, cmap='gray')
    plt.scatter(image_label[:, 0], image_label[:, 1], c='r', s=2)
    for i, (x, y) in enumerate(image_label):
        plt.text(x, y, str(i), color='red', fontsize=8)
    plt.title(name)
    if save:
        plt.savefig(f"datasets/dataset_cache/Ceph-Xray/2024-MICCAI-TEST/{name}.png")
    if show:
        plt.show()



if __name__ == "__main__":
    # main(False)
    # test_noncontinuous()
    # plot_landmarks_on_image("494.bmp")
    # test_inverse_preprocessing()
    test_preprocess_just_landmarks()
