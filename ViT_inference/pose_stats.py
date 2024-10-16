import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import random
from torch.utils.data import DataLoader, Dataset
import time
import subprocess
import os
#import cv2
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.gesture_head = GestureRecognitionHead(
            embedding_size=768,
            num_classes=num_classes,
            layer_sizes=gesture_layer_sizes,
            dropout_rates=gesture_dropout_rates,
            activations=gesture_activations
        )
        for param in self.gesture_head.parameters():
            param.requires_grad = False  # Freeze the gesture recognition head

        # ViT Backbone
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # Freeze ViT Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

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
        self.pose_estimation_head = DynamicPoseEstimationHead(
            combined_feature_size=768 + 512,
            num_keypoints=num_keypoints,
            max_people=max_people,
            layer_sizes=pose_layer_sizes,
            dropout_rates=pose_dropout_rates,
            activations=pose_activations
        )

        # Depth Estimation Head
        self.depth_estimation_head = DepthEstimationHead(
            combined_feature_size=768 + 512,
            output_size=output_size,
            layer_sizes=depth_layer_sizes,
            dropout_rates=depth_dropout_rates,
            activations=depth_activations
        )

        for param in self.depth_estimation_head.parameters():
            param.requires_grad = False  # Freeze the pose estimation head

    def forward(self, x, task='all'):
        # ViT Backbone
        vit_outputs = self.backbone(pixel_values=x)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]

        outputs = {}

        if task in ['gesture', 'all']:
            # Gesture Recognition Output using only ViT features
            gesture_output = self.gesture_head(vit_features)
            outputs['gesture'] = gesture_output

        if task in ['pose', 'depth', 'all']:
            # CNN Feature Extractor
            cnn_features = self.feature_extractor(x)
            processed_cnn_features = self.cnn_feature_processor(cnn_features)

            # Combined Features
            combined_features = torch.cat((processed_cnn_features, vit_features), dim=1)

            if task in ['pose', 'all']:
                # Pose Estimation Output using combined features
                keypoints = self.pose_estimation_head(combined_features)
                keypoints = torch.sigmoid(keypoints)  # Normalize keypoints to [0, 1] range
                outputs['pose'] = keypoints

            if task in ['depth', 'all']:
                # Depth Estimation Output using combined features
                depth_map = self.depth_estimation_head(combined_features)
                outputs['depth'] = depth_map

        if len(outputs) == 1:
            return list(outputs.values())[0]  # Return the single output directly
        else:
            return outputs


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
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():

        # Start time measurement for inference
        start_time = time.time()

        # Perform inference multiple times for stability
        for _ in range(num_iterations):
            # Compute only the depth estimation task
            keypoints = model(input_tensor, task='pose')


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
