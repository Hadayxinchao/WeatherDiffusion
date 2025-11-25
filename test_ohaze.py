#!/usr/bin/env python3
"""
Quick test script to evaluate the O-HAZE model
This will process a few validation images and save the results
"""

import argparse
import os
import yaml
import torch
import numpy as np
import datasets
from models import DenoisingDiffusion, DiffusiveRestoration


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    parser = argparse.ArgumentParser(description='Quick test for O-HAZE dehazing model')
    parser.add_argument("--checkpoint", type=str, default="WeatherDiff64.pth.tar",
                        help="Path to checkpoint file")
    parser.add_argument("--test_set", type=str, default="val",
                        help="Test set: 'val' or 'test'")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of sampling steps (default: 25)")
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width (lower=better quality, try 4-16)")
    parser.add_argument("--num_images", type=int, default=3,
                        help="Number of images to process (default: 3)")
    args_test = parser.parse_args()

    # Load config
    print("Loading O-HAZE configuration...")
    with open("configs/ohaze.yml", "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config.device = device

    # Set seeds
    torch.manual_seed(61)
    np.random.seed(61)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(61)

    # Create args for the model
    args = argparse.Namespace(
        resume=args_test.checkpoint,
        sampling_timesteps=args_test.sampling_timesteps,
        grid_r=args_test.grid_r,
        image_folder='results/images/',
        seed=61
    )

    # Load dataset
    print(f"Loading O-HAZE {args_test.test_set} dataset...")
    DATASET = datasets.OHaze(config)
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args_test.test_set)
    print(f"Found {len(val_loader.dataset)} images in {args_test.test_set} set")

    # Create model
    print(f"Loading model from: {args_test.checkpoint}")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)

    # Process images
    print(f"\nProcessing {args_test.num_images} images...")
    print(f"Settings:")
    print(f"  - Sampling timesteps: {args_test.sampling_timesteps}")
    print(f"  - Grid size (r): {args_test.grid_r}")
    print(f"  - Output folder: {args.image_folder}")
    print()

    # Limit to specified number of images
    limited_loader = []
    for i, batch in enumerate(val_loader):
        if i >= args_test.num_images:
            break
        limited_loader.append(batch)

    # Create a simple iterator
    class LimitedDataset:
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            return iter(self.data)
        def __len__(self):
            return len(self.data)

    limited_dataset = LimitedDataset(limited_loader)
    
    # Run restoration
    model.restore(limited_dataset, validation=args_test.test_set, r=args_test.grid_r)

    print("\n" + "="*50)
    print("✓ Testing completed!")
    print("="*50)
    print(f"\nRestored images saved to: {args.image_folder}")
    print(f"Check the output folder to see the dehazed results.")


if __name__ == "__main__":
    main()
