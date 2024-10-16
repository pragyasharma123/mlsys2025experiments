import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import random
import time
import subprocess

# Configuration and global settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 19  # Number of gesture classes
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

        for size, dropout_rate, activation in zip(layer_sizes, dropout_rates, activations):
            layers.append(DynamicLinearBlock(input_size, size, dropout_rate, activation))
            input_size = size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)

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
        self.output_layer = nn.Linear(input_size, output_size[0] * output_size[1])

    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, *self.output_size)  # Reshaping output to (batch_size, 1, 224, 224)
        return x

class CombinedModel(nn.Module):
    def __init__(self, num_classes, output_size=(224, 224),
                 gesture_layer_sizes=[512, 512], gesture_dropout_rates=[0.1, 0.1], gesture_activations=["ReLU", "ReLU"],
                 depth_layer_sizes=[512, 256], depth_dropout_rates=[0.4, 0.2], depth_activations=["ReLU", "ReLU"]):
        super(CombinedModel, self).__init__()
        self.gesture_head = GestureRecognitionHead(embedding_size=768, num_classes=num_classes,
                                                   layer_sizes=gesture_layer_sizes, dropout_rates=gesture_dropout_rates,
                                                   activations=gesture_activations)

        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
        for param in self.backbone.parameters():
            param.requires_grad = False

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

        self.depth_estimation_head = DepthEstimationHead(
            combined_feature_size=768 + 512, output_size=output_size,
            layer_sizes=depth_layer_sizes, dropout_rates=depth_dropout_rates, activations=depth_activations
        )

    def forward(self, x, task=['gesture', 'depth']):
        # ViT Backbone
        vit_outputs = self.backbone(pixel_values=x)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]

        outputs = {}

        # Gesture Recognition Output
        if 'gesture' in task:
            gesture_output = self.gesture_head(vit_features)
            outputs['gesture'] = gesture_output

        # Depth Estimation Output
        if 'depth' in task:
            cnn_features = self.feature_extractor(x)
            processed_cnn_features = self.cnn_feature_processor(cnn_features)
            combined_features = torch.cat((processed_cnn_features, vit_features), dim=1)
            depth_map = self.depth_estimation_head(combined_features)
            outputs['depth'] = depth_map

        return outputs


def main():
    torch.manual_seed(42)
    random.seed(42)

    # Instantiate the model
    model = CombinedModel(num_classes=num_classes)
    model.to(device)

    # Prepare dummy input tensor for inference
    batch_size = 2
    num_iterations = 10
    total_samples = batch_size * num_iterations
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    # Measure inference performance
    model.eval()

    with torch.no_grad():

        start_time = time.time()

        for _ in range(num_iterations):
            outputs = model(input_tensor, task=['gesture', 'depth'])
            gesture_outputs = outputs['gesture']
            depth_outputs = outputs['depth']


        end_time = time.time()

        time_taken = end_time - start_time
        frames_per_second = total_samples / time_taken
        latency_per_sample = time_taken / total_samples

        print(f"Inference Performance: {frames_per_second:.2f} frames/second")
        print(f"Latency per sample: {latency_per_sample:.6f} seconds")



if __name__ == '__main__':
    main()

