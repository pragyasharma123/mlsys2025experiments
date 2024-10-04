# -*- coding: utf-8 -*-
"""Modified Inference Stats for Raspberry Pi 5 (Frames per second and Latency)"""
'''
import os
import pandas as pd
import random
from PIL import Image
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import time

# THESE ARE THE CLASSNAMES FOR THE 18 DIFFERENT HAND GESTURES
class_names = [
   'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace',
   'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up',
   'two_up_inverted', 'no_gesture']

# Configuration and global settings
device = torch.device("cpu")
num_classes = len(class_names)  # Update as necessary
num_keypoints = 16  # Update as necessary
model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)

class DynamicLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, activation_func):
        super(DynamicLinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        if activation_func == "ReLU":
            self.activation = nn.ReLU()
        elif activation_func == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation_func == "ELU":
            self.activation = nn.ELU()

    def forward(self, x):
        x = self.linear(x)
        if x.size(0) > 1:  # Apply batch normalization only if batch size > 1
            x = self.bn(x)
        return self.dropout(self.activation(x))

class DynamicPoseEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, num_keypoints, max_people=13):
        super().__init__()
        self.max_people = max_people
        self.layers = nn.ModuleList()
        input_size = combined_feature_size
        self.layers.append(DynamicLinearBlock(input_size, 512, 0.4, "ReLU"))
        self.layers.append(DynamicLinearBlock(512, 256, 0.2, "ReLU"))
        self.output_layer = nn.Linear(256, num_keypoints * max_people * 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_people, -1, 2)
        return x

class GestureRecognitionHead(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(GestureRecognitionHead, self).__init__()
        self.layers = nn.Sequential(
            DynamicLinearBlock(embedding_size, 512, 0.1, "ReLU"),
            DynamicLinearBlock(512, 512, 0.1, "ReLU")
        )
        self.output_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)

class CombinedModel(nn.Module):
    def __init__(self, num_classes, num_keypoints, max_people=13):
        super(CombinedModel, self).__init__()
        self.max_people = max_people
        self.gesture_head = GestureRecognitionHead(embedding_size=768, num_classes=num_classes)
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.pose_estimation_head = DynamicPoseEstimationHead(combined_feature_size=768 + 512, num_keypoints=num_keypoints, max_people=max_people)

    def forward(self, x):
        vit_outputs = self.backbone(pixel_values=x)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]
        gesture_output = self.gesture_head(vit_features)

        # CNN features processing
        cnn_features = self.feature_extractor(x)
        processed_cnn_features = self.cnn_feature_processor(cnn_features)
        combined_features = torch.cat((processed_cnn_features, vit_features.detach()), dim=1)

        keypoints = self.pose_estimation_head(combined_features)
        keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1] range

        return keypoints, gesture_output


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Prepare the model
    model = CombinedModel(num_classes=num_classes, num_keypoints=16)
    model.to(device)

    # Prepare dummy input tensor for inference
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    # Measure inference performance
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Start time measurement for inference
        start_time = time.time()

        # Inference step
        for _ in range(100):  # Run inference multiple times to get a more stable measurement
            _, gesture_outputs = model(input_tensor)  # Perform inference

        # End time measurement for inference
        end_time = time.time()

        # Calculate time taken for inference
        time_taken = end_time - start_time
        inferences_per_second = (batch_size * 100) / time_taken  # Total inferences / time taken
        frames_per_second = inferences_per_second  # Since each inference corresponds to a frame

        # Print inferences per second
        print(f"Inference Performance: {inferences_per_second:.2f} inferences/second")
        print(f"Frames processed per second: {frames_per_second:.2f}")

        # Calculate latency per sample
        latency_per_sample = time_taken / (batch_size * 100)
        print(f"Latency per sample: {latency_per_sample:.6f} seconds")

if __name__ == '__main__':
    main()

import json
import os
from PIL import Image
from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader, Subset
from transformers import ViTModel, ViTConfig
from torch import nn
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch
from transformers import ViTModel, ViTConfig
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

class PoseEstimationDataset(Dataset):
    def __init__(self, image_dir, json_path=None, transform=None, target_size=(224, 224)):
        self.image_dir = image_dir
        self.transform = transform or Compose([Resize(target_size), ToTensor()])

        if json_path is not None:
            with open(json_path, 'r') as file:
                self.data = json.load(file)
        else:
            self.data = None
            self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        else:
            return len(self.image_filenames)

    def __getitem__(self, idx):
        if self.data is not None:
            item = self.data[idx]
            image_path = os.path.join(self.image_dir, item['image_filename'])
        else:
            image_path = os.path.join(self.image_dir, self.image_filenames[idx])

        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB
        orig_width, orig_height = image.size

        if self.transform:
            image = self.transform(image)

        if self.data is not None:
            keypoints = []
            denormalized_keypoints = []
            for joint_data in item['ground_truth'].values():
                for joint in joint_data:
                    x, y = joint[:2]  # Only take x and y, ignoring visibility
                    denormalized_keypoints.append([x, y])
                    if not (x == 0 and y == 0):  # Filter out (0.0, 0.0) keypoints for normalized
                        keypoints.append([x / orig_width, y / orig_height])

            keypoints_tensor = torch.tensor(keypoints).float()
            denormalized_keypoints_tensor = torch.tensor(denormalized_keypoints).float()

            # Check for 'head' and 'upper_neck' keypoints, handling empty lists
            head_keypoints = item['ground_truth'].get('head', [[0, 0, 0]])
            upper_neck_keypoints = item['ground_truth'].get('upper_neck', [[0, 0, 0]])

            head = head_keypoints[0][:2] if head_keypoints else [0, 0]
            upper_neck = upper_neck_keypoints[0][:2] if upper_neck_keypoints else [0, 0]

            head_normalized = [head[0] / orig_width, head[1] / orig_height]
            upper_neck_normalized = [upper_neck[0] / orig_width, upper_neck[1] / orig_height]

            return image, keypoints_tensor, denormalized_keypoints_tensor, head_normalized, upper_neck_normalized, item['image_filename'], orig_width, orig_height

        else:
            # For the test dataset without annotations
            return image, None, None, None, None, os.path.basename(image_path), orig_width, orig_height



# pose estimation utility functions
def filter_keypoints_by_variance(keypoints, variance_threshold=0.01):
    """
    Filter keypoints based on variance across the batch.
    Keypoints with low variance are likely to be less accurate.

    :param keypoints: Predicted keypoints, tensor of shape (batch_size, max_people, num_keypoints, 2).
    :param variance_threshold: Variance threshold for filtering.
    :return: Filtered keypoints tensor.
    """
    # Calculate variance across the batch dimension
    variances = torch.var(keypoints, dim=0)  # Shape: (max_people, num_keypoints, 2)

    # Identify keypoints with variance below the threshold
    low_variance_mask = variances < variance_threshold

    # Filter out low-variance keypoints by setting them to zero
    # Note: This step merely invalidates low-variance keypoints without removing them.
    # You may need to adjust this logic based on how you want to handle filtered keypoints.
    filtered_keypoints = keypoints.clone()
    filtered_keypoints[:, low_variance_mask] = 0  # Set low-variance keypoints to zero

    return filtered_keypoints


def calculate_accuracy(valid_predictions, valid_gt, threshold=0.05):
    """
    Calculate the accuracy of valid predictions against the ground truth keypoints.

    :param valid_predictions: Tensor of matched predicted keypoints, shape (N, 2) where N is the number of matched keypoints.
    :param valid_gt: Tensor of ground truth keypoints corresponding to the matched predictions, shape (N, 2).
    :param threshold: Distance threshold to consider a prediction as correct.
    :return: Accuracy as a percentage of correctly predicted keypoints.
    """
    if valid_predictions.numel() == 0 or valid_gt.numel() == 0:
        return 0.0  # Return 0 accuracy if there are no keypoints to compare

    # Calculate the Euclidean distance between each pair of valid predicted and ground truth keypoints
    distances = torch.norm(valid_predictions - valid_gt, dim=1)

    # Determine which predictions are within the threshold distance of the ground truth keypoints
    correct_predictions = distances < threshold

    # Calculate accuracy as the percentage of predictions that are correct
    accuracy = torch.mean(correct_predictions.float()) * 100  # Convert fraction to percentage

    return accuracy.item()

def calculate_valid_accuracy(pred_keypoints, gt_keypoints, threshold=0.05):
    """
    Calculate accuracy based on the distance between predicted and ground truth keypoints,
    considering only those keypoints that are matched within a specified threshold.
    """
    total_correct = 0
    total_valid = 0

    for pred, gt in zip(pred_keypoints, gt_keypoints):
        # Assuming pred and gt are already on the correct device and properly scaled
        distances = calculate_distances(pred, gt)
        matched = match_keypoints(distances, threshold)

        total_correct += len(matched)
        total_valid += gt.size(0)  # Assuming gt is a 2D tensor of shape [N, 2]

    if total_valid > 0:
        accuracy = (total_correct / total_valid) * 100
    else:
        accuracy = 0.0

    return accuracy



def threshold_filter_keypoints(keypoints, lower_bound=0.05, upper_bound=0.95):
    """
    Filter keypoints based on a simple thresholding mechanism.

    :param keypoints: The keypoints predicted by the model, shaped as (batch_size, max_people, num_keypoints, 2).
    :param lower_bound: Lower bound for valid keypoint values.
    :param upper_bound: Upper bound for valid keypoint values.
    :return: Thresholded keypoints tensor.
    """
    # Create a mask for keypoints that fall within the specified bounds
    valid_mask = (keypoints > lower_bound) & (keypoints < upper_bound)

    # Apply the mask to both dimensions of the keypoints (x and y)
    valid_keypoints = keypoints * valid_mask.all(dim=-1, keepdim=True)

    return valid_keypoints

def calculate_distances(pred_keypoints, gt_keypoints):
    """
    Calculate distances between predicted keypoints and ground truth keypoints.

    :param pred_keypoints: Predicted keypoints as a tensor of shape (num_predictions, 2).
    :param gt_keypoints: Ground truth keypoints as a tensor of shape (num_gt_keypoints, 2).
    :return: A tensor of distances of shape (num_predictions, num_gt_keypoints).
    """
    pred_keypoints = pred_keypoints.to(device)
    gt_keypoints = gt_keypoints.to(device)
    num_predictions = pred_keypoints.shape[0]
    num_gt = gt_keypoints.shape[0]
    distances = torch.zeros((num_predictions, num_gt))

    for i in range(num_predictions):
        for j in range(num_gt):
            distances[i, j] = torch.norm(pred_keypoints[i] - gt_keypoints[j])

    return distances

def match_keypoints(distances, threshold=0.05):
    """
    Match predicted keypoints to ground truth keypoints based on minimum distance.

    :param distances: A tensor of distances between predictions and ground truth keypoints.
    :param threshold: Distance threshold for valid matches.
    :return: Indices of predicted keypoints that match ground truth keypoints.
    """
    matched_indices = []

    for i in range(distances.shape[1]):  # Iterate over ground truth keypoints
        min_dist, idx = torch.min(distances[:, i], dim=0)
        if min_dist < threshold:
            matched_indices.append(idx.item())

    return matched_indices

def denormalize_keypoints(keypoints, orig_width, orig_height):
    """
    Denormalize keypoints from [0, 1] range back to original image dimensions.

    :param keypoints: Tensor of keypoints in normalized form, shape (N, 2) where N is the number of keypoints.
    :param orig_width: Original width of the image.
    :param orig_height: Original height of the image.
    :return: Denormalized keypoints tensor.
    """
    denormalized_keypoints = keypoints.clone()
    denormalized_keypoints[:, 0] *= orig_width  # Scale x coordinates
    denormalized_keypoints[:, 1] *= orig_height  # Scale y coordinates
    return denormalized_keypoints

def pad_tensors_to_match(a, b):
    """
    Pad the shorter import json
import os
from PIL import Image
from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader, Subset
from transformers import ViTModel, ViTConfig
from torch import nn
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch
from transformers import ViTModel, ViTConfig
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

class PoseEstimationDataset(Dataset):
    def __init__(self, image_dir, json_path=None, transform=None, target_size=(224, 224)):
        self.image_dir = image_dir
        self.transform = transform or Compose([Resize(target_size), ToTensor()])

        if json_path is not None:
            with open(json_path, 'r') as file:
                self.data = json.load(file)
        else:
            self.data = None
            self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        else:
            return len(self.image_filenames)

    def __getitem__(self, idx):
        if self.data is not None:
            item = self.data[idx]
            image_path = os.path.join(self.image_dir, item['image_filename'])
        else:
            image_path = os.path.join(self.image_dir, self.image_filenames[idx])

        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB
        orig_width, orig_height = image.size

        if self.transform:
            image = self.transform(image)

        if self.data is not None:
            keypoints = []
            denormalized_keypoints = []
            for joint_data in item['ground_truth'].values():
                for joint in joint_data:
                    x, y = joint[:2]  # Only take x and y, ignoring visibility
                    denormalized_keypoints.append([x, y])
                    if not (x == 0 and y == 0):  # Filter out (0.0, 0.0) keypoints for normalized
                        keypoints.append([x / orig_width, y / orig_height])

            keypoints_tensor = torch.tensor(keypoints).float()
            denormalized_keypoints_tensor = torch.tensor(denormalized_keypoints).float()

            # Check for 'head' and 'upper_neck' keypoints, handling empty lists
            head_keypoints = item['ground_truth'].get('head', [[0, 0, 0]])
            upper_neck_keypoints = item['ground_truth'].get('upper_neck', [[0, 0, 0]])

            head = head_keypoints[0][:2] if head_keypoints else [0, 0]
            upper_neck = upper_neck_keypoints[0][:2] if upper_neck_keypoints else [0, 0]

            head_normalized = [head[0] / orig_width, head[1] / orig_height]
            upper_neck_normalized = [upper_neck[0] / orig_width, upper_neck[1] / orig_height]

            return image, keypoints_tensor, denormalized_keypoints_tensor, head_normalized, upper_neck_normalized, item['image_filename'], orig_width, orig_height

        else:
            # For the test dataset without annotations
            return image, None, None, None, None, os.path.basename(image_path), orig_width, orig_height



# pose estimation utility functions
def filter_keypoints_by_variance(keypoints, variance_threshold=0.01):
    """
    Filter keypoints based on variance across the batch.
    Keypoints with low variance are likely to be less accurate.

    :param keypoints: Predicted keypoints, tensor of shape (batch_size, max_people, num_keypoints, 2).
    :param variance_threshold: Variance threshold for filtering.
    :return: Filtered keypoints tensor.
    """
    # Calculate variance across the batch dimension
    variances = torch.var(keypoints, dim=0)  # Shape: (max_people, num_keypoints, 2)

    # Identify keypoints with variance below the threshold
    low_variance_mask = variances < variance_threshold

    # Filter out low-variance keypoints by setting them to zero
    # Note: This step merely invalidates low-variance keypoints without removing them.
    # You may need to adjust this logic based on how you want to handle filtered keypoints.
    filtered_keypoints = keypoints.clone()
    filtered_keypoints[:, low_variance_mask] = 0  # Set low-variance keypoints to zero

    return filtered_keypoints


def calculate_accuracy(valid_predictions, valid_gt, threshold=0.05):
    """
    Calculate the accuracy of valid predictions against the ground truth keypoints.

    :param valid_predictions: Tensor of matched predicted keypoints, shape (N, 2) where N is the number of matched keypoints.
    :param valid_gt: Tensor of ground truth keypoints corresponding to the matched predictions, shape (N, 2).
    :param threshold: Distance threshold to consider a prediction as correct.
    :return: Accuracy as a percentage of correctly predicted keypoints.
    """
    if valid_predictions.numel() == 0 or valid_gt.numel() == 0:
        return 0.0  # Return 0 accuracy if there are no keypoints to compare

    # Calculate the Euclidean distance between each pair of valid predicted and ground truth keypoints
    distances = torch.norm(valid_predictions - valid_gt, dim=1)

    # Determine which predictions are within the threshold distance of the ground truth keypoints
    correct_predictions = distances < threshold

    # Calculate accuracy as the percentage of predictions that are correct
    accuracy = torch.mean(correct_predictions.float()) * 100  # Convert fraction to percentage

    return accuracy.item()

def calculate_valid_accuracy(pred_keypoints, gt_keypoints, threshold=0.05):
    """
    Calculate accuracy based on the distance between predicted and ground truth keypoints,
    considering only those keypoints that are matched within a specified threshold.
    """
    total_correct = 0
    total_valid = 0

    for pred, gt in zip(pred_keypoints, gt_keypoints):
        # Assuming pred and gt are already on the correct device and properly scaled
        distances = calculate_distances(pred, gt)
        matched = match_keypoints(distances, threshold)

        total_correct += len(matched)
        total_valid += gt.size(0)  # Assuming gt is a 2D tensor of shape [N, 2]

    if total_valid > 0:
        accuracy = (total_correct / total_valid) * 100
    else:
        accuracy = 0.0

    return accuracy



def threshold_filter_keypoints(keypoints, lower_bound=0.05, upper_bound=0.95):
    """
    Filter keypoints based on a simple thresholding mechanism.

    :param keypoints: The keypoints predicted by the model, shaped as (batch_size, max_people, num_keypoints, 2).
    :param lower_bound: Lower bound for valid keypoint values.
    :param upper_bound: Upper bound for valid keypoint values.
    :return: Thresholded keypoints tensor.
    """
    # Create a mask for keypoints that fall within the specified bounds
    valid_mask = (keypoints > lower_bound) & (keypoints < upper_bound)

    # Apply the mask to both dimensions of the keypoints (x and y)
    valid_keypoints = keypoints * valid_mask.all(dim=-1, keepdim=True)

    return valid_keypoints

def calculate_distances(pred_keypoints, gt_keypoints):
    """
    Calculate distances between predicted keypoints and ground truth keypoints.

    :param pred_keypoints: Predicted keypoints as a tensor of shape (num_predictions, 2).
    :param gt_keypoints: Ground truth keypoints as a tensor of shape (num_gt_keypoints, 2).
    :return: A tensor of distances of shape (num_predictions, num_gt_keypoints).
    """
    pred_keypoints = pred_keypoints.to(device)
    gt_keypoints = gt_keypoints.to(device)
    num_predictions = pred_keypoints.shape[0]
    num_gt = gt_keypoints.shape[0]
    distances = torch.zeros((num_predictions, num_gt))

    for i in range(num_predictions):
        for j in range(num_gt):
            distances[i, j] = torch.norm(pred_keypoints[i] - gt_keypoints[j])

    return distances

def match_keypoints(distances, threshold=0.05):
    """
    Match predicted keypoints to ground truth keypoints based on minimum distance.

    :param distances: A tensor of distances between predictions and ground truth keypoints.
    :param threshold: Distance threshold for valid matches.
    :return: Indices of predicted keypoints that match ground truth keypoints.
    """
    matched_indices = []

    for i in range(distances.shape[1]):  # Iterate over ground truth keypoints
        min_dist, idx = torch.min(distances[:, i], dim=0)
        if min_dist < threshold:
            matched_indices.append(idx.item())

    return matched_indices

def denormalize_keypoints(keypoints, orig_width, orig_height):
    """
    Denormalize keypoints from [0, 1] range back to original image dimensions.

    :param keypoints: Tensor of keypoints in normalized form, shape (N, 2) where N is the number of keypoints.
    :param orig_width: Original width of the image.
    :param orig_height: Original height of the image.
    :return: Denormalized keypoints tensor.
    """
    denormalized_keypoints = keypoints.clone()
    denormalized_keypoints[:, 0] *= orig_width  # Scale x coordinates
    denormalized_keypoints[:, 1] *= orig_height  # Scale y coordinates
    return denormalized_keypoints

def pad_tensors_to_match(a, b):
    """
    Pad the shorter tensor among 'a' and 'b' with zeros to match the length of the longer tensor.
    Returns padded tensors and a mask indicating the original elements.

    Args:
    a (Tensor): First tensor.
    b (Tensor): Second tensor.

    Returns:
    Tensor, Tensor, Tensor: Padded version of 'a', padded version of 'b', and a mask.
    """
    max_len = max(a.size(0), b.size(0))

    # Create masks for original keypoints (1 for real, 0 for padded)
    mask_a = torch.ones(a.size(0), dtype=torch.float32, device=a.device)
    mask_b = torch.ones(b.size(0), dtype=torch.float32, device=b.device)

    # Pad tensors to match the maximum length
    padded_a = torch.cat([a, torch.zeros(max_len - a.size(0), *a.shape[1:], device=a.device)], dim=0)
    padded_b = torch.cat([b, torch.zeros(max_len - b.size(0), *b.shape[1:], device=b.device)], dim=0)

    # Pad masks to match the maximum length
    padded_mask_a = torch.cat([mask_a, torch.zeros(max_len - mask_a.size(0), device=a.device)], dim=0)
    padded_mask_b = torch.cat([mask_b, torch.zeros(max_len - mask_b.size(0), device=b.device)], dim=0)

    # Combine masks (logical AND) since we want to consider keypoints that are present in both tensors
    combined_mask = padded_mask_a * padded_mask_b

    return padded_a, padded_b, combined_mask



def masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss between 'pred' and 'target', applying 'mask' to ignore padded values.

    Args:
    pred (Tensor): Predicted keypoints.
    target (Tensor): Ground truth keypoints.
    mask (Tensor): Mask tensor indicating valid keypoints.

    Returns:
    Tensor: Masked MSE loss.
    """
    # Ensure the mask is boolean for advanced indexing
    mask = mask.bool()

    # Flatten the tensors and mask to simplify indexing
    pred_flat = pred.view(-1, pred.size(-1))
    target_flat = target.view(-1, target.size(-1))
    mask_flat = mask.view(-1)

    # Apply mask
    valid_pred = pred_flat[mask_flat]
    valid_target = target_flat[mask_flat]

    # Compute MSE loss on valid keypoints only
    loss = F.mse_loss(valid_pred, valid_target)

    return loss

def calculate_pckh(valid_predictions, gt_keypoints_subset, threshold):
    distances_to_gt = torch.norm(valid_predictions - gt_keypoints_subset, dim=1)
    correct_predictions = (distances_to_gt < threshold).float()
    return correct_predictions.mean().item()
    


# Device selection (CUDA GPU if available, otherwise CPU)
device = torch.device("cpu")
print(device)

# THESE ARE THE CLASSNAMES FOR THE 18 DIFFERENT HAND GESTURES
class_names = [
   'call',
   'dislike',
   'fist',
   'four',
   'like',
   'mute',
   'ok',
   'one',
   'palm',
   'peace',
   'peace_inverted',
   'rock',
   'stop',
   'stop_inverted',
   'three',
   'three2',
   'two_up',
   'two_up_inverted',
   'no_gesture']

num_keypoints = 16
num_classes = len(class_names)
model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)

class DynamicLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, activation_func):
        super(DynamicLinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        if activation_func == "ReLU":
            self.activation = nn.ReLU()
        elif activation_func == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation_func == "ELU":
            self.activation = nn.ELU()

    def forward(self, x):
        x = self.linear(x)
        if x.size(0) > 1:  # Apply batch normalization only if batch size > 1
            x = self.bn(x)
        return self.dropout(self.activation(x))

class GestureRecognitionHead(nn.Module):
    def __init__(self, embedding_size, num_classes, layer_sizes, dropout_rates, activations):
        super(GestureRecognitionHead, self).__init__()
        layers = []
        input_size = embedding_size

        for i, (size, dropout_rate, activation) in enumerate(zip(layer_sizes, dropout_rates, activations)):
            layers.append(DynamicLinearBlock(input_size, size, dropout_rate, activation))
            input_size = size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)

class DynamicPoseEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, num_keypoints, max_people=13,
                 layer_sizes=[512, 256], dropout_rates=[0.4, 0.2], activations=["ReLU", "ReLU"]):
        super().__init__()
        self.max_people = max_people
        self.layers = nn.ModuleList()
        input_size = combined_feature_size
        for size, dropout, activation in zip(layer_sizes, dropout_rates, activations):
            self.layers.append(DynamicLinearBlock(input_size, size, dropout, activation))
            input_size = size
        self.output_layer = nn.Linear(input_size, num_keypoints * max_people * 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_people, -1, 2)
        return x


class CombinedModel(nn.Module):
    def __init__(self, num_classes, num_keypoints, max_people=13, gesture_layer_sizes=[512, 512], gesture_dropout_rates=[0.1, 0.1], gesture_activations=["ReLU", "ReLU"], pose_layer_sizes=[512, 256], pose_dropout_rates=[0.5, 0.25], pose_activations=["ReLU", "LeakyReLU"]):
        super(CombinedModel, self).__init__()
        self.max_people = max_people
        self.gesture_head = GestureRecognitionHead(embedding_size=768, num_classes=num_classes, layer_sizes=gesture_layer_sizes, dropout_rates=gesture_dropout_rates, activations=gesture_activations)

        for param in self.gesture_head.parameters():
            param.requires_grad = False  # Freeze the gesture recognition head

        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.pose_estimation_head = DynamicPoseEstimationHead(combined_feature_size=768 + 512,
                                                              num_keypoints=num_keypoints,
                                                              max_people=max_people,
                                                              layer_sizes=[512, 256],
                                                              dropout_rates=[0.4, 0.2],
                                                              activations=["ReLU", "ReLU"])

    def forward(self, x):
        with torch.no_grad():
            vit_outputs = self.backbone(pixel_values=x)
            vit_features = vit_outputs.last_hidden_state[:, 0, :]
            gesture_output = self.gesture_head(vit_features)

        cnn_features = self.feature_extractor(x)
        processed_cnn_features = self.cnn_feature_processor(cnn_features)
        combined_features = torch.cat((processed_cnn_features, vit_features.detach()), dim=1)

        keypoints = self.pose_estimation_head(combined_features)
        keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1] range
        return keypoints, gesture_output


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    keypoints_tensors = [item[1] for item in batch]  # List of tensors
    denormalized_keypoints_tensors = [item[2] for item in batch]  # List of tensors
    head_points = [item[3] for item in batch]  # Collect head points
    upper_neck_points = [item[4] for item in batch]  # Collect upper neck points
    image_filenames = [item[5] for item in batch]  # Adjusted to item[5]
    orig_widths = torch.tensor([item[6] for item in batch])  # Adjusted to item[6]
    orig_heights = torch.tensor([item[7] for item in batch])  # Adjusted to item[7]

    # Since images can be stacked into a single tensor directly,
    # we leave them as is. For variable-sized tensors like keypoints,
    # we keep them as lists of tensors.

    return images, keypoints_tensors, denormalized_keypoints_tensors, head_points, upper_neck_points, image_filenames, orig_widths, orig_heights

def custom_collate_fn_test(batch):
    images = torch.stack([item[0] for item in batch])
    image_filenames = [item[5] for item in batch]  # Adjusted to item[5]
    orig_widths = torch.tensor([item[6] for item in batch])  # Adjusted to item[6]
    orig_heights = torch.tensor([item[7] for item in batch])  # Adjusted to item[7]

    return images, image_filenames, orig_widths, orig_heights


def visualize_keypoints(image, keypoints, ground_truth, orig_width, orig_height):
    # Move image to CPU and detach from the computation graph
    image = image.detach().cpu()

    # Unnormalize the image if it was normalized
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean  # Revert normalization

    # Convert image tensor to numpy format and clip any values out of range [0,1]
    image = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    image = np.clip(image, 0, 1)

    # Resize the image back to original dimensions
    image = cv2.resize(image, (orig_width, orig_height))

    # Scale keypoints back to the original image dimensions
    keypoints = keypoints.detach().cpu().numpy()
    keypoints[:, 0] *= orig_width
    keypoints[:, 1] *= orig_height

    ground_truth = ground_truth.detach().cpu().numpy()
    ground_truth[:, 0] *= orig_width
    ground_truth[:, 1] *= orig_height

    # Plot the image and keypoints
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=50, c='red', marker='o', label='Predicted')
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], s=50, c='green', marker='x', label='Ground Truth')

    plt.legend()
    plt.axis('off')
    plt.savefig('keypoints_comparison.png')
    plt.show()

def visualize_keypoints_test(image, keypoints, orig_width, orig_height):
    # Move image to CPU and detach from the computation graph
    image = image.detach().cpu()

    # Unnormalize the image if it was normalized
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean  # Revert normalization

    # Convert image tensor to numpy format and clip any values out of range [0,1]
    image = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    image = np.clip(image, 0, 1)

    # Resize the image back to original dimensions
    image = cv2.resize(image, (orig_width, orig_height))

    # Scale keypoints back to the original image dimensions
    keypoints = keypoints.detach().cpu().numpy()
    keypoints[:, 0] *= orig_width
    keypoints[:, 1] *= orig_height

    # Plot the image and keypoints
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=50, c='red', marker='o', label='Predicted')

    plt.legend()
    plt.axis('off')
    plt.show()

    # Optionally, save the image with keypoints
    plt.savefig('test_keypoints_comparison.png')


    # Define any transforms you want to apply to your images
p_transforms = Compose([
        Resize((224, 224)),  # Resize the image
        ToTensor(),  # Convert the image to a PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Prepare the model
    model = CombinedModel(num_classes=num_classes, num_keypoints=16)
    model.to(device)

    # Prepare dummy input tensor for inference
    batch_size = 1  # You can adjust batch size based on your needs
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    # Measure inference performance
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Start time measurement for inference
        start_time = time.time()

        # Inference step (running 100 iterations for stability)
        for _ in range(100):
            gesture_outputs = model(input_tensor)

        # End time measurement for inference
        end_time = time.time()

        # Calculate time taken for inference
        time_taken = end_time - start_time
        total_samples = batch_size * 100  # Total number of samples processed
        frames_per_second = total_samples / time_taken  # Multiply batch size by number of iterations

        # Calculate latency per sample (time taken per sample)
        latency_per_sample = time_taken / total_samples

        # Print frames per second and latency
        print(f"Inference Performance: {frames_per_second:.2f} frames/second")
        print(f"Latency per sample: {latency_per_sample:.6f} seconds")
if __name__ == '__main__':
    main()
    
tensor among 'a' and 'b' with zeros to match the length of the longer tensor.
    Returns padded tensors and a mask indicating the original elements.

    Args:
    a (Tensor): First tensor.
    b (Tensor): Second tensor.

    Returns:
    Tensor, Tensor, Tensor: Padded version of 'a', padded version of 'b', and a mask.
    """
    max_len = max(a.size(0), b.size(0))

    # Create masks for original keypoints (1 for real, 0 for padded)
    mask_a = torch.ones(a.size(0), dtype=torch.float32, device=a.device)
    mask_b = torch.ones(b.size(0), dtype=torch.float32, device=b.device)

    # Pad tensors to match the maximum length
    padded_a = torch.cat([a, torch.zeros(max_len - a.size(0), *a.shape[1:], device=a.device)], dim=0)
    padded_b = torch.cat([b, torch.zeros(max_len - b.size(0), *b.shape[1:], device=b.device)], dim=0)

    # Pad masks to match the maximum length
    padded_mask_a = torch.cat([mask_a, torch.zeros(max_len - mask_a.size(0), device=a.device)], dim=0)
    padded_mask_b = torch.cat([mask_b, torch.zeros(max_len - mask_b.size(0), device=b.device)], dim=0)

    # Combine masks (logical AND) since we want to consider keypoints that are present in both tensors
    combined_mask = padded_mask_a * padded_mask_b

    return padded_a, padded_b, combined_mask



def masked_mse_loss(pred, target, mask):
    """
    Compute MSE loss between 'pred' and 'target', applying 'mask' to ignore padded values.

    Args:
    pred (Tensor): Predicted keypoints.
    target (Tensor): Ground truth keypoints.
    mask (Tensor): Mask tensor indicating valid keypoints.

    Returns:
    Tensor: Masked MSE loss.
    """
    # Ensure the mask is boolean for advanced indexing
    mask = mask.bool()

    # Flatten the tensors and mask to simplify indexing
    pred_flat = pred.view(-1, pred.size(-1))
    target_flat = target.view(-1, target.size(-1))
    mask_flat = mask.view(-1)

    # Apply mask
    valid_pred = pred_flat[mask_flat]
    valid_target = target_flat[mask_flat]

    # Compute MSE loss on valid keypoints only
    loss = F.mse_loss(valid_pred, valid_target)

    return loss

def calculate_pckh(valid_predictions, gt_keypoints_subset, threshold):
    distances_to_gt = torch.norm(valid_predictions - gt_keypoints_subset, dim=1)
    correct_predictions = (distances_to_gt < threshold).float()
    return correct_predictions.mean().item()
    


# Device selection (CUDA GPU if available, otherwise CPU)
device = torch.device("cpu")
print(device)

# THESE ARE THE CLASSNAMES FOR THE 18 DIFFERENT HAND GESTURES
class_names = [
   'call',
   'dislike',
   'fist',
   'four',
   'like',
   'mute',
   'ok',
   'one',
   'palm',
   'peace',
   'peace_inverted',
   'rock',
   'stop',
   'stop_inverted',
   'three',
   'three2',
   'two_up',
   'two_up_inverted',
   'no_gesture']

num_keypoints = 16
num_classes = len(class_names)
model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)

class DynamicLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, activation_func):
        super(DynamicLinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        if activation_func == "ReLU":
            self.activation = nn.ReLU()
        elif activation_func == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation_func == "ELU":
            self.activation = nn.ELU()

    def forward(self, x):
        x = self.linear(x)
        if x.size(0) > 1:  # Apply batch normalization only if batch size > 1
            x = self.bn(x)
        return self.dropout(self.activation(x))

class GestureRecognitionHead(nn.Module):
    def __init__(self, embedding_size, num_classes, layer_sizes, dropout_rates, activations):
        super(GestureRecognitionHead, self).__init__()
        layers = []
        input_size = embedding_size

        for i, (size, dropout_rate, activation) in enumerate(zip(layer_sizes, dropout_rates, activations)):
            layers.append(DynamicLinearBlock(input_size, size, dropout_rate, activation))
            input_size = size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)

class DynamicPoseEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, num_keypoints, max_people=13,
                 layer_sizes=[512, 256], dropout_rates=[0.4, 0.2], activations=["ReLU", "ReLU"]):
        super().__init__()
        self.max_people = max_people
        self.layers = nn.ModuleList()
        input_size = combined_feature_size
        for size, dropout, activation in zip(layer_sizes, dropout_rates, activations):
            self.layers.append(DynamicLinearBlock(input_size, size, dropout, activation))
            input_size = size
        self.output_layer = nn.Linear(input_size, num_keypoints * max_people * 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_people, -1, 2)
        return x


class CombinedModel(nn.Module):
    def __init__(self, num_classes, num_keypoints, max_people=13, gesture_layer_sizes=[512, 512], gesture_dropout_rates=[0.1, 0.1], gesture_activations=["ReLU", "ReLU"], pose_layer_sizes=[512, 256], pose_dropout_rates=[0.5, 0.25], pose_activations=["ReLU", "LeakyReLU"]):
        super(CombinedModel, self).__init__()
        self.max_people = max_people
        self.gesture_head = GestureRecognitionHead(embedding_size=768, num_classes=num_classes, layer_sizes=gesture_layer_sizes, dropout_rates=gesture_dropout_rates, activations=gesture_activations)

        for param in self.gesture_head.parameters():
            param.requires_grad = False  # Freeze the gesture recognition head

        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.pose_estimation_head = DynamicPoseEstimationHead(combined_feature_size=768 + 512,
                                                              num_keypoints=num_keypoints,
                                                              max_people=max_people,
                                                              layer_sizes=[512, 256],
                                                              dropout_rates=[0.4, 0.2],
                                                              activations=["ReLU", "ReLU"])

    def forward(self, x):
        with torch.no_grad():
            vit_outputs = self.backbone(pixel_values=x)
            vit_features = vit_outputs.last_hidden_state[:, 0, :]
            gesture_output = self.gesture_head(vit_features)

        cnn_features = self.feature_extractor(x)
        processed_cnn_features = self.cnn_feature_processor(cnn_features)
        combined_features = torch.cat((processed_cnn_features, vit_features.detach()), dim=1)

        keypoints = self.pose_estimation_head(combined_features)
        keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1] range
        return keypoints, gesture_output


def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    keypoints_tensors = [item[1] for item in batch]  # List of tensors
    denormalized_keypoints_tensors = [item[2] for item in batch]  # List of tensors
    head_points = [item[3] for item in batch]  # Collect head points
    upper_neck_points = [item[4] for item in batch]  # Collect upper neck points
    image_filenames = [item[5] for item in batch]  # Adjusted to item[5]
    orig_widths = torch.tensor([item[6] for item in batch])  # Adjusted to item[6]
    orig_heights = torch.tensor([item[7] for item in batch])  # Adjusted to item[7]

    # Since images can be stacked into a single tensor directly,
    # we leave them as is. For variable-sized tensors like keypoints,
    # we keep them as lists of tensors.

    return images, keypoints_tensors, denormalized_keypoints_tensors, head_points, upper_neck_points, image_filenames, orig_widths, orig_heights

def custom_collate_fn_test(batch):
    images = torch.stack([item[0] for item in batch])
    image_filenames = [item[5] for item in batch]  # Adjusted to item[5]
    orig_widths = torch.tensor([item[6] for item in batch])  # Adjusted to item[6]
    orig_heights = torch.tensor([item[7] for item in batch])  # Adjusted to item[7]

    return images, image_filenames, orig_widths, orig_heights


def visualize_keypoints(image, keypoints, ground_truth, orig_width, orig_height):
    # Move image to CPU and detach from the computation graph
    image = image.detach().cpu()

    # Unnormalize the image if it was normalized
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean  # Revert normalization

    # Convert image tensor to numpy format and clip any values out of range [0,1]
    image = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    image = np.clip(image, 0, 1)

    # Resize the image back to original dimensions
    image = cv2.resize(image, (orig_width, orig_height))

    # Scale keypoints back to the original image dimensions
    keypoints = keypoints.detach().cpu().numpy()
    keypoints[:, 0] *= orig_width
    keypoints[:, 1] *= orig_height

    ground_truth = ground_truth.detach().cpu().numpy()
    ground_truth[:, 0] *= orig_width
    ground_truth[:, 1] *= orig_height

    # Plot the image and keypoints
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=50, c='red', marker='o', label='Predicted')
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], s=50, c='green', marker='x', label='Ground Truth')

    plt.legend()
    plt.axis('off')
    plt.savefig('keypoints_comparison.png')
    plt.show()

def visualize_keypoints_test(image, keypoints, orig_width, orig_height):
    # Move image to CPU and detach from the computation graph
    image = image.detach().cpu()

    # Unnormalize the image if it was normalized
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean  # Revert normalization

    # Convert image tensor to numpy format and clip any values out of range [0,1]
    image = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    image = np.clip(image, 0, 1)

    # Resize the image back to original dimensions
    image = cv2.resize(image, (orig_width, orig_height))

    # Scale keypoints back to the original image dimensions
    keypoints = keypoints.detach().cpu().numpy()
    keypoints[:, 0] *= orig_width
    keypoints[:, 1] *= orig_height

    # Plot the image and keypoints
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=50, c='red', marker='o', label='Predicted')

    plt.legend()
    plt.axis('off')
    plt.show()

    # Optionally, save the image with keypoints
    plt.savefig('test_keypoints_comparison.png')


    # Define any transforms you want to apply to your images
p_transforms = Compose([
        Resize((224, 224)),  # Resize the image
        ToTensor(),  # Convert the image to a PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Prepare the model
    model = CombinedModel(num_classes=num_classes, num_keypoints=16)
    model.to(device)

    # Prepare dummy input tensor for inference
    batch_size = 1  # You can adjust batch size based on your needs
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    # Measure inference performance
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Start time measurement for inference
        start_time = time.time()

        # Inference step (running 100 iterations for stability)
        for _ in range(100):
            gesture_outputs = model(input_tensor)

        # End time measurement for inference
        end_time = time.time()

        # Calculate time taken for inference
        time_taken = end_time - start_time
        total_samples = batch_size * 100  # Total number of samples processed
        frames_per_second = total_samples / time_taken  # Multiply batch size by number of iterations

        # Calculate latency per sample (time taken per sample)
        latency_per_sample = time_taken / total_samples

        # Print frames per second and latency
        print(f"Inference Performance: {frames_per_second:.2f} frames/second")
        print(f"Latency per sample: {latency_per_sample:.6f} seconds")
if __name__ == '__main__':
    main()
    
    
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

class DepthEstimationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            root_dir (str): Root directory containing the dataset.
            transform (callable, optional): Optional transform to be applied
                on an image.
            target_size (tuple): Target size for resizing images and depth maps.
        """
        self.root_dir = root_dir
        self.transform = transform or Compose([
            Resize(target_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.depth_transform = Compose([
            Resize(target_size),
            ToTensor()
        ])
        self.image_paths = []
        self.depth_paths = []

        # Populate the list of image and depth paths
        for scene_dir in sorted(os.listdir(root_dir)):
            scene_path = os.path.join(root_dir, scene_dir)
            if os.path.isdir(scene_path):
                for file_name in sorted(os.listdir(scene_path)):
                    if file_name.endswith('.jpg'):
                        image_path = os.path.join(scene_path, file_name)
                        depth_path = image_path.replace('.jpg', '.png')
                        self.image_paths.append(image_path)
                        self.depth_paths.append(depth_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]

        # Load image and depth map
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path)

        # Apply transformationsif self.transform:
        image = self.transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)

        return image, depth

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import random
from torch.utils.data import DataLoader, Dataset
import time
import subprocess
import os
import numpy as np
from PIL import Image



# THESE ARE THE CLASSNAMES FOR THE 19 DIFFERENT HAND GESTURES
class_names = [
   'call',
   'dislike',
   'fist',
   'four',
   'like',
   'mute',
   'ok',
   'one',
   'palm',
   'peace',
   'peace_inverted',
   'rock',
   'stop',
   'stop_inverted',
   'three',
   'three2',
   'two_up',
   'two_up_inverted',
   'no_gesture']

FORMATS = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")
# Configuration and global settings
device = torch.device("cpu")
num_classes = len(class_names)  # Update as necessary
num_keypoints = 16  # Update as necessary
model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)

class DynamicLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, activation_func):
        super(DynamicLinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU() if activation_func == "ReLU" else nn.LeakyReLU() if activation_func == "LeakyReLU" else nn.ELU()

    def forward(self, x):
        x = self.linear(x)
        if x.size(0) > 1:
            x = self.bn(x)
        return self.dropout(self.activation(x))

class GestureRecognitionHead(nn.Module):
    def __init__(self, embedding_size, num_classes, layer_sizes, dropout_rates, activations):
        super(GestureRecognitionHead, self).__init__()
        layers = []
        input_size = embedding_size

        for i, (size, dropout_rate, activation) in enumerate(zip(layer_sizes, dropout_rates, activations)):
            layers.append(DynamicLinearBlock(input_size, size, dropout_rate, activation))
            input_size = size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)

class DynamicPoseEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, num_keypoints, max_people=13,
                 layer_sizes=[512, 256], dropout_rates=[0.4, 0.2], activations=["ReLU", "ReLU"]):
        super().__init__()
        self.max_people = max_people
        self.layers = nn.ModuleList()
        input_size = combined_feature_size
        for size, dropout, activation in zip(layer_sizes, dropout_rates, activations):
            self.layers.append(DynamicLinearBlock(input_size, size, dropout, activation))
            input_size = size
        self.output_layer = nn.Linear(input_size, num_keypoints * max_people * 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_people, -1, 2)
        return x

class DepthEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, output_size=(224, 224), layer_sizes=[512, 256], dropout_rates=[0.4, 0.2], activations=["ReLU", "ReLU"]):
        super(DepthEstimationHead, self).__init__()
        self.output_size = output_size
        layers = []
        input_size = combined_feature_size

        for size, dropout_rate, activation in zip(layer_sizes, dropout_rates, activations):
            layers.append(DynamicLinearBlock(input_size, size, dropout_rate, activation))
            input_size = size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, output_size[0] * output_size[1])  # This ensures output is 224*224

    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, *self.output_size)  # Reshaping output to (batch_size, 1, 224, 224)
        return x

class CombinedModel(nn.Module):
    def __init__(self, num_classes, num_keypoints, max_people=13, output_size=(224, 224),
                 gesture_layer_sizes=[512, 512], gesture_dropout_rates=[0.1, 0.1],
                 gesture_activations=["ReLU", "ReLU"], pose_layer_sizes=[512, 256],
                 pose_dropout_rates=[0.5, 0.25], pose_activations=["ReLU", "LeakyReLU"],
                 depth_layer_sizes=[512, 256], depth_dropout_rates=[0.4, 0.2],
                 depth_activations=["ReLU", "ReLU"]):
        super(CombinedModel, self).__init__()
        self.max_people = max_people

         # Gesture Recognition Head
        self.gesture_head = GestureRecognitionHead(embedding_size=768, num_classes=num_classes, layer_sizes=gesture_layer_sizes, dropout_rates=gesture_dropout_rates, activations=gesture_activations)
        for param in self.gesture_head.parameters():
            param.requires_grad = False # Freeze the gesture recognition head

        # ViT Backbone
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # CNN Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.cnn_feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Pose Estimation Head
        self.pose_estimation_head = DynamicPoseEstimationHead(combined_feature_size=768 + 512,
                                                              num_keypoints=num_keypoints,
                                                              max_people=max_people,
                                                              layer_sizes=pose_layer_sizes,
                                                              dropout_rates=pose_dropout_rates,
                                                              activations=pose_activations)
        for param in self.pose_estimation_head.parameters():
            param.requires_grad = False # Freeze pose head


        # Depth Estimation Head
        self.depth_estimation_head = DepthEstimationHead(combined_feature_size=768 + 512,
                                                         output_size=output_size,
                                                         layer_sizes=depth_layer_sizes,
                                                         dropout_rates=depth_dropout_rates,
                                                         activations=depth_activations)

    def forward(self, x):
        # ViT Backbone
        with torch.no_grad():
            vit_outputs = self.backbone(pixel_values=x)
            vit_features = vit_outputs.last_hidden_state[:, 0, :]
            gesture_output = self.gesture_head(vit_features)


        # CNN Feature Extractor
        cnn_features = self.feature_extractor(x)
        processed_cnn_features = self.cnn_feature_processor(cnn_features)

        # Combined Features
        combined_features = torch.cat((processed_cnn_features, vit_features.detach()), dim=1)

        # Pose Estimation Output
        keypoints = self.pose_estimation_head(combined_features)
        keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1] range

        # Depth Estimation Output
        depth_map = self.depth_estimation_head(combined_features)

        return keypoints, gesture_output, depth_map

def compute_mae(depth_output, depth_maps):
    mae = torch.mean(torch.abs(depth_output - depth_maps))
    return mae

def compute_absolute_relative_error(depth_output, depth_maps):
    epsilon = 1e-6
    relative_error = torch.abs(depth_output - depth_maps) / (depth_maps + epsilon)
    are = torch.mean(relative_error)
    return are

def compute_rmse(depth_output, depth_maps):
    mse = torch.mean((depth_output - depth_maps) ** 2)
    rmse = torch.sqrt(mse)
    return rmse


def main():
    torch.manual_seed(42)
    random.seed(42)

    # Instantiate the model
    model = CombinedModel(num_classes=num_classes, num_keypoints=num_keypoints)
    model.to(device)

    # Prepare dummy input tensor for inference
    batch_size = 1
    num_iterations = 100  # Number of iterations for stability
    total_samples = batch_size * num_iterations  # Total number of samples processed
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    # Measure inference performance
    model.eval()
    with torch.no_grad():
        # Start time measurement for inference
        start_time = time.time()

        # Perform inference multiple times for stability
        for _ in range(num_iterations):
            depth_output = model(input_tensor)

        # End time measurement for inference
        end_time = time.time()

        # Calculate time taken for inference
        time_taken = end_time - start_time
        frames_per_second = total_samples / time_taken

        # Calculate latency per sample (time taken per sample)
        latency_per_sample = time_taken / total_samples

        # Print frames per second
        print(f"Inference Performance: {frames_per_second:.2f} frames/second")
        print(f"Latency per sample: {latency_per_sample:.6f} seconds")


if __name__ == '__main__':
    main()




import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import random
import time

# THESE ARE THE CLASSNAMES FOR THE 19 DIFFERENT HAND GESTURES
class_names = [
   'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace',
   'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up',
   'two_up_inverted', 'no_gesture']

# Configuration and global settings
device = torch.device("cpu")  # Set to CPU since Raspberry Pi doesn't have GPU
num_classes = len(class_names)
num_keypoints = 16
model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)

class DynamicLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, activation_func):
        super(DynamicLinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        if activation_func == "ReLU":
            self.activation = nn.ReLU()
        elif activation_func == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation_func == "ELU":
            self.activation = nn.ELU()

    def forward(self, x):
        x = self.linear(x)
        if x.size(0) > 1:
            x = self.bn(x)
        return self.dropout(self.activation(x))

class DynamicPoseEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, num_keypoints, max_people=13,
                 layer_sizes=[512, 256], dropout_rates=[0.4, 0.2], activations=["ReLU", "ReLU"]):
        super().__init__()
        self.max_people = max_people
        self.layers = nn.ModuleList()
        input_size = combined_feature_size
        for size, dropout, activation in zip(layer_sizes, dropout_rates, activations):
            self.layers.append(DynamicLinearBlock(input_size, size, dropout, activation))
            input_size = size
        self.output_layer = nn.Linear(input_size, num_keypoints * max_people * 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_people, -1, 2)
        return x

class GestureRecognitionHead(nn.Module):
    def __init__(self, embedding_size, num_classes, layer_sizes, dropout_rates, activations):
        super(GestureRecognitionHead, self).__init__()
        layers = []
        input_size = embedding_size

        for i, (size, dropout_rate, activation) in enumerate(zip(layer_sizes, dropout_rates, activations)):
            layers.append(DynamicLinearBlock(input_size, size, dropout_rate, activation))
            input_size = size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)

class CombinedModel(nn.Module):
    def __init__(self, num_classes, num_keypoints, max_people=13, gesture_layer_sizes=[512, 512], gesture_dropout_rates=[0.1, 0.1], gesture_activations=["ReLU", "ReLU"], pose_layer_sizes=[512, 256], pose_dropout_rates=[0.5, 0.25], pose_activations=["ReLU", "LeakyReLU"]):
        super(CombinedModel, self).__init__()
        self.max_people = max_people
        self.gesture_head = GestureRecognitionHead(embedding_size=768, num_classes=num_classes, layer_sizes=gesture_layer_sizes, dropout_rates=gesture_dropout_rates, activations=gesture_activations)

        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.cnn_feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.pose_estimation_head = DynamicPoseEstimationHead(combined_feature_size=768 + 512,
                                                              num_keypoints=num_keypoints,
                                                              max_people=max_people,
                                                              layer_sizes=[512, 256],
                                                              dropout_rates=[0.4, 0.2],
                                                              activations=["ReLU", "ReLU"])

        # Freeze the Pose Estimation Head
        for param in self.pose_estimation_head.parameters():
            param.requires_grad = False

        # Explicitly freeze the feature_extractor and cnn_feature_processor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        for param in self.cnn_feature_processor.parameters():
            param.requires_grad = False

    def forward(self, x):
        vit_outputs = self.backbone(pixel_values=x)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]
        gesture_output = self.gesture_head(vit_features)

        with torch.no_grad():
            # CNN features processing
            cnn_features = self.feature_extractor(x)
            processed_cnn_features = self.cnn_feature_processor(cnn_features)
            combined_features = torch.cat((processed_cnn_features, vit_features.detach()), dim=1)

            keypoints = self.pose_estimation_head(combined_features)
            keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1] range

            return keypoints, gesture_output

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Prepare the model
    model = CombinedModel(num_classes=num_classes, num_keypoints=16)
    model.to(device)

    # Prepare dummy input tensor for inference
    batch_size = 2
    num_iterations = 100
    total_samples = batch_size * num_iterations
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    # Measure inference performance
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Start time measurement for inference
        start_time = time.time()

        # Combined Inference step (running 100 iterations for both pose and gesture recognition)
        for _ in range(num_iterations):
            keypoints, gesture_outputs = model(input_tensor)  # Perform both pose and gesture inferences

        # End time measurement for inference
        end_time = time.time()

        # Calculate time taken for inference
        time_taken = end_time - start_time
        frames_per_second = total_samples / time_taken  # Total samples processed over time taken

        # Calculate latency per sample (time taken per sample)
        latency_per_sample = time_taken / total_samples

        # Print frames per second
        print(f"Inference Performance: {frames_per_second:.2f} frames/second")
        print(f"Latency per sample: {latency_per_sample:.6f} seconds")

if __name__ == '__main__':
    main()




import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import random
import time

# THESE ARE THE CLASSNAMES FOR THE 19 DIFFERENT HAND GESTURES
class_names = [
   'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace',
   'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up',
   'two_up_inverted', 'no_gesture']

# Configuration and global settings
device = torch.device("cpu")  # Set to CPU since Raspberry Pi doesn't have GPU
num_classes = len(class_names)
num_keypoints = 16
model_name = 'google/vit-base-patch16-224'
config = ViTConfig.from_pretrained(model_name)
vit_backbone = ViTModel.from_pretrained(model_name, config=config)

class DynamicLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate, activation_func):
        super(DynamicLinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU() if activation_func == "ReLU" else nn.LeakyReLU() if activation_func == "LeakyReLU" else nn.ELU()

    def forward(self, x):
        x = self.linear(x)
        if x.size(0) > 1:
            x = self.bn(x)
        return self.dropout(self.activation(x))

class GestureRecognitionHead(nn.Module):
    def __init__(self, embedding_size, num_classes, layer_sizes, dropout_rates, activations):
        super(GestureRecognitionHead, self).__init__()
        layers = []
        input_size = embedding_size

        for i, (size, dropout_rate, activation) in enumerate(zip(layer_sizes, dropout_rates, activations)):
            layers.append(DynamicLinearBlock(input_size, size, dropout_rate, activation))
            input_size = size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)

class DynamicPoseEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, num_keypoints, max_people=13,
                 layer_sizes=[512, 256], dropout_rates=[0.4, 0.2], activations=["ReLU", "ReLU"]):
        super().__init__()
        self.max_people = max_people
        self.layers = nn.ModuleList()
        input_size = combined_feature_size
        for size, dropout, activation in zip(layer_sizes, dropout_rates, activations):
            self.layers.append(DynamicLinearBlock(input_size, size, dropout, activation))
            input_size = size
        self.output_layer = nn.Linear(input_size, num_keypoints * max_people * 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_people, -1, 2)
        return x

class DepthEstimationHead(nn.Module):
    def __init__(self, combined_feature_size, output_size=(224, 224), layer_sizes=[512, 256], dropout_rates=[0.4, 0.2], activations=["ReLU", "ReLU"]):
        super(DepthEstimationHead, self).__init__()
        self.output_size = output_size
        layers = []
        input_size = combined_feature_size

        for size, dropout_rate, activation in zip(layer_sizes, dropout_rates, activations):
            layers.append(DynamicLinearBlock(input_size, size, dropout_rate, activation))
            input_size = size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, output_size[0] * output_size[1])  # This ensures output is 224x224

    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, *self.output_size)  # Reshaping output to (batch_size, 1, 224, 224)
        return x

class CombinedModel(nn.Module):
    def __init__(self, num_classes, num_keypoints, max_people=13, output_size=(224, 224),
                 gesture_layer_sizes=[512, 512], gesture_dropout_rates=[0.1, 0.1],
                 gesture_activations=["ReLU", "ReLU"], pose_layer_sizes=[512, 256],
                 pose_dropout_rates=[0.5, 0.25], pose_activations=["ReLU", "LeakyReLU"],
                 depth_layer_sizes=[512, 256], depth_dropout_rates=[0.4, 0.2],
                 depth_activations=["ReLU", "ReLU"]):
        super(CombinedModel, self).__init__()
        self.max_people = max_people

        # Gesture Recognition Head
        self.gesture_head = GestureRecognitionHead(embedding_size=768, num_classes=num_classes, layer_sizes=gesture_layer_sizes, dropout_rates=gesture_dropout_rates, activations=gesture_activations)

        # ViT Backbone
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # CNN Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.cnn_feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Pose Estimation Head
        self.pose_estimation_head = DynamicPoseEstimationHead(combined_feature_size=768 + 512,
                                                              num_keypoints=num_keypoints,
                                                              max_people=max_people,
                                                              layer_sizes=pose_layer_sizes,
                                                              dropout_rates=pose_dropout_rates,
                                                              activations=pose_activations)

        # Depth Estimation Head
        self.depth_estimation_head = DepthEstimationHead(combined_feature_size=768 + 512,
                                                         output_size=output_size,
                                                         layer_sizes=depth_layer_sizes,
                                                         dropout_rates=depth_dropout_rates,
                                                         activations=depth_activations)

    def forward(self, x):
        # ViT Backbone
        vit_outputs = self.backbone(pixel_values=x)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]
        gesture_output = self.gesture_head(vit_features)

        # CNN Feature Extractor
        cnn_features = self.feature_extractor(x)
        processed_cnn_features = self.cnn_feature_processor(cnn_features)

        # Combined Features
        combined_features = torch.cat((processed_cnn_features, vit_features), dim=1)

        # Pose Estimation Output
        keypoints = self.pose_estimation_head(combined_features)
        keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1] range

        # Depth Estimation Output
        depth_map = self.depth_estimation_head(combined_features)

        return keypoints, gesture_output, depth_map


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Prepare the model
    model = CombinedModel(num_classes=num_classes, num_keypoints=num_keypoints)
    model.to(device)

    # Prepare dummy input tensor for inference
    batch_size = 3
    num_iterations = 100
    total_samples = batch_size * num_iterations
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    # Measure inference performance
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        # Start time measurement for inference
        start_time = time.time()

        # Perform inference multiple times for stability
        for _ in range(num_iterations):
            keypoints, gesture_output, depth_output = model(input_tensor)

        # End time measurement for inference
        end_time = time.time()

        # Calculate time taken for inference
        time_taken = end_time - start_time
        frames_per_second = total_samples / time_taken

        # Calculate latency per sample (time taken per sample)
        latency_per_sample = time_taken / total_samples

        # Print frames per second
        print(f"Inference Performance: {frames_per_second:.2f} frames/second")
        print(f"Latency per sample: {latency_per_sample:.6f} seconds")


if __name__ == '__main__':
    main()


'''













































