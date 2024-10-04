import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import torchsummary
from torchvision.models import resnet50
from torch.utils.data import DataLoader, TensorDataset

class unipose(nn.Module):
    def __init__(self, num_classes=15, stride=8):
        super(unipose, self).__init__()
        self.stride = stride
        self.num_classes = num_classes

        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        self.backbone = resnet50(pretrained=True)

        # Modify the backbone to return intermediate features
        self.backbone_layers = nn.Sequential(*list(self.backbone.children())[:-2])

        self.wasp = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, input):
        x = self.backbone_layers(input)  # Extract features from the backbone
        x = self.wasp(x)
        x = self.decoder(x)
        if self.stride != 8:
            x = F.interpolate(x, size=(input.size()[2:]), mode='bilinear', align_corners=True)
        return x

def benchmark_model(model, device, data_loader):
    model.eval()  # Set model to evaluation mode for inference
    total_samples_processed = 0
    total_time = 0

    with torch.no_grad():  # Disable gradient computation for inference
        for inputs, _ in data_loader:
            start_time = time.time()

            # Forward pass only
            outputs = model(inputs.to(device))

            end_time = time.time()
            batch_time = end_time - start_time  # Measure time for this batch
            total_time += batch_time
            total_samples_processed += len(inputs)

    throughput_per_sample = total_samples_processed / total_time
    print(f"Throughput per sample: {throughput_per_sample:.2f} samples per second")

    return throughput_per_sample, total_time, total_samples_processed

def main():
    batch_size = 1  # Real-time inference with batch size 1
    num_epochs = 1  # For inference, this will be 1
    output_json = 'benchmark_results_inference.json'

    device = torch.device("cpu")  # Set to CPU since Raspberry Pi lacks a GPU
    model = unipose(num_classes=15).to(device)

    # Create dummy data for inference
    inputs = torch.randn(1, 3, 256, 256)  # Single sample for demonstration
    input_size = (3, 256, 256)
    torchsummary.summary(model, input_size)
    targets = torch.randint(0, 15, (1, 15, 32, 32))  # Single target for demonstration

    data_loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)

    # Benchmark model for inference
    throughput_per_sample, elapsed_time, total_samples_processed = benchmark_model(model, device, data_loader)

    # Calculate model size in MB
    model_size_mb = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)

    # Calculate latency per sample
    latency_per_sample = elapsed_time / total_samples_processed

    # Print results
    print(f"Throughput per sample: {throughput_per_sample:.2f} samples per second")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Latency per sample: {latency_per_sample:.6f} seconds")

    output_data_dict = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "throughput_per_sample": throughput_per_sample,
        "model_size_mb": model_size_mb,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "latency_per_sample": latency_per_sample,
    }

    with open(output_json, "w") as f:
        json.dump(output_data_dict, f, indent=4)
    print(f"Results saved to {output_json}")

if __name__ == '__main__':
    main()
