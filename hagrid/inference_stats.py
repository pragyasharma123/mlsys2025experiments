import logging
import time
import torch
import numpy as np
import json
from omegaconf import OmegaConf, DictConfig
from albumentations.pytorch import ToTensorV2
import albumentations as A
from models import classifiers_list, detectors_list

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

class Demo:
    @staticmethod
    def get_transform_for_inf(transform_config):
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

def build_model(config: DictConfig):
    model_name = config.model.name
    num_classes = len(config.dataset.targets)
    model_config = {"num_classes": num_classes, "pretrained": config.model.pretrained}
    
    if model_name in detectors_list:
        model_config["num_classes"] += 1
        model_config.update(
            {
                "pretrained_backbone": config.model.pretrained_backbone,
                "img_size": config.dataset.img_size,
                "img_mean": config.dataset.img_mean,
                "img_std": config.dataset.img_std,
            }
        )
        model = detectors_list[model_name](**model_config)
        model.type = "detector"
    elif model_name in classifiers_list:
        model = classifiers_list[model_name](**model_config)
        model.criterion = getattr(torch.nn, config.criterion)()
        model.type = "classifier"
    else:
        raise Exception(f"Unknown model {model_name}")

    return model

def benchmark_model(model, device, input_tensor, num_repeats):
    model.eval()  # Set model to evaluation mode for inference
    total_time = 0

    with torch.no_grad():  # Disable gradient computation for inference
        for _ in range(num_repeats):
            start_time = time.time()

            output = model(input_tensor)  # Forward pass

            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time

    # Calculate metrics
    samples_per_second = num_repeats / total_time
    avg_inference_time = total_time / num_repeats  # Latency per sample

    print(f"Avg inference time (latency per sample): {avg_inference_time:.6f} seconds")
    print(f"Samples per second: {samples_per_second:.2f} samples per second")

    return samples_per_second, avg_inference_time

def main():
    path_to_config = "/home/pragyasharma/hagrid-profiling/configs/ResNext50.yaml"  # Path to the config file
    conf = OmegaConf.load(path_to_config)
    device = torch.device("cpu")  # Set to CPU for Raspberry Pi

    # Build the model
    model = build_model(conf).to(device)

    if conf.model.checkpoint:
        snapshot = torch.load(conf.model.checkpoint, map_location=device)
        model.load_state_dict(snapshot["MODEL_STATE"])

    model = model.float()

    # Set model to evaluation mode
    model.eval()

    # Set the transform
    transform = Demo.get_transform_for_inf(conf.test_transforms)

    # Create a dummy input for inference
    dummy_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    processed_input = transform(image=dummy_input)['image'].unsqueeze(0).to(device)

    batch_size = 1  # Real-time inference with batch size 1
    input_tensor = processed_input.repeat(batch_size, 1, 1, 1).to(device)

    num_repeats = 100

    # Benchmark model for inference
    samples_per_second, avg_inference_time = benchmark_model(model, device, input_tensor, num_repeats)

    # Print results
    print(f"Samples per second: {samples_per_second:.2f}")
    print(f"Latency per sample: {avg_inference_time:.6f} seconds")

    # Save results in a JSON file
    output_data_dict = {
        "samples_per_second": samples_per_second,
        "latency_per_sample": avg_inference_time
    }

    output_json = "/path/to/output/inference_perf.json"
    with open(output_json, "w") as f:
        json.dump(output_data_dict, f, indent=4)
    print(f"Results saved to {output_json}")

if __name__ == "__main__":
    main()
