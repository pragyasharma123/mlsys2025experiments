import torch
import sys
import time
import json
import importlib.util
import os
from collections import namedtuple

# Path to the bts.py file
bts_path = '/home/pragyasharma/bts/pytorch/bts.py'

# Check if file exists
if not os.path.exists(bts_path):
    raise FileNotFoundError(f"Cannot find bts.py at {bts_path}")

# Load the bts.py module
spec = importlib.util.spec_from_file_location('bts', bts_path)
bts = importlib.util.module_from_spec(spec)
sys.modules['bts'] = bts
spec.loader.exec_module(bts)

# Now, import BtsModel from the dynamically loaded module
BtsModel = bts.BtsModel

def load_model(checkpoint_path, params):
    model = BtsModel(params)  # Initialize the BTS model
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load model checkpoint on CPU

    # Handle 'module.' prefix in state_dict keys (from DataParallel or DistributedDataParallel)
    state_dict = checkpoint['model']
    if any(key.startswith('module.') for key in state_dict.keys()):
        # Remove 'module.' prefix from keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        state_dict = new_state_dict

    # Load the corrected state_dict into the model
    model.load_state_dict(state_dict)
    model.eval()
    return model

def benchmark_model(model, device, input_tensor, focal_tensor):
    model.eval()  # Set model to evaluation mode for inference
    total_samples_processed = 0
    total_time = 0

    num_repeats = 5
    with torch.no_grad():  # Disable gradient computation for inference
        for _ in range(num_repeats):
            start_time = time.time()

            outputs = model(input_tensor, focal_tensor)  # Forward pass

            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            total_samples_processed += input_tensor.size(0)

    throughput_per_sample = total_samples_processed / total_time
    print(f"Throughput per sample: {throughput_per_sample:.2f} samples per second")

    return throughput_per_sample, total_time, total_samples_processed

def main():
    # Define params using namedtuple to match the expected structure
    Params = namedtuple('Params', ['encoder', 'bts_size', 'max_depth', 'dataset'])
    params = Params(
        encoder='densenet161_bts',  # or another encoder you are using
        bts_size=512,
        max_depth=80,
        dataset='nyu'
    )

    checkpoint_path = "/home/pragyasharma/bts/bts_nyu_v2_pytorch_densenet161/model"

    device = torch.device("cpu")  # Ensure running on CPU for Raspberry Pi
    model = load_model(checkpoint_path, params).to(device)

    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Calculate model size in MB
    model_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per parameter (float32)
    print(f"Model size: {model_size_mb:.2f} MB")

    # Create dummy input for testing inference
    input_tensor = torch.randn(1, 3, 480, 640).to(device)  # Match the expected input size of the model
    focal_tensor = torch.tensor([715.0873], device=device)  # Dummy focal length tensor

    # Run benchmark and monitor performance
    throughput_per_sample, total_time, total_samples_processed = benchmark_model(model, device, input_tensor, focal_tensor)

    # Calculate latency per sample
    latency_per_sample = total_time / total_samples_processed

    # Print results
    print(f"Throughput per sample: {throughput_per_sample:.2f} samples per second")
    print(f"Latency per sample: {latency_per_sample:.6f} seconds")

    # Save results to JSON
    output_data_dict = {
        "throughput_per_sample": throughput_per_sample,
        "latency_per_sample": latency_per_sample,
        "total_parameters": total_params,
        "model_size_mb": model_size_mb
    }

    

if __name__ == '__main__':
    main()

