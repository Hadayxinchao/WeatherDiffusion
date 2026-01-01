# WeatherDiffusion AI Agent Instructions

## Project Overview
This is a **patch-based denoising diffusion model** for image restoration under adverse weather conditions (snow, rain, fog, raindrops). The core innovation is processing full-size images through overlapping patches during inference while training on fixed-size patches. Published in IEEE TPAMI 2023.

## Architecture Components

### Three-Layer System Design
1. **DenoisingDiffusion** (`models/ddm.py`): Core diffusion model managing training loop, noise schedules (beta), timesteps, and EMA
2. **DiffusiveRestoration** (`models/restoration.py`): Inference wrapper handling full-image patch-based restoration with overlapping grids
3. **DiffusionUNet** (`models/unet.py`): U-Net backbone for noise estimation (adapted from DDIM/DDRM)

### Key Data Flow
- **Training**: Fixed patches (64×64 or 128×128) → conditional input (degraded image 3ch) + noisy target (clean image 3ch) → noise estimation loss
- **Inference**: Full image → overlapping grid patches → diffusion sampling per patch → weighted averaging by overlap count → restored image

## Critical Implementation Details

### Patch-Based Restoration Pattern
The `generalized_steps_overlapping()` function in `utils/sampling.py` is central:
- Creates overlapping grid with stride `r` (default 16, smaller = better quality but slower)
- Maintains `x_grid_mask` counting overlaps per pixel
- Aggregates noise estimates across all patches covering each pixel
- Divides by overlap count for averaging: `et = torch.div(et_output, x_grid_mask)`

### Conditional Diffusion Convention
Input format is **always 6-channel concatenation**: `[degraded_image(3ch), noisy_image(3ch)]`
- First 3 channels: conditioning (weather-degraded input) - remains constant
- Last 3 channels: denoising target - evolves through diffusion timesteps
- See `noise_estimation_loss()` in `models/ddm.py` line 98

### Data Transform Pattern
Always use paired transforms for consistency:
- Forward: `data_transform(X) = 2 * X - 1.0` (normalize [0,1] → [-1,1])
- Inverse: `inverse_data_transform(X) = clamp((X + 1.0) / 2.0, 0.0, 1.0)`

## Dataset Structure Conventions

Each dataset class (e.g., `AllWeather`, `Snow100K`) follows this pattern:
- `get_loaders()`: Returns `(train_loader, val_loader)` tuple
- Dataset reads paired images from `.txt` file lists with format: `input_path gt_path`
- Training uses random patches; validation uses full images or parse_patches=False
- AllWeather is multi-task training set; individual datasets for task-specific evaluation

## Development Workflows

### Training
```bash
python train_diffusion.py --config "allweather.yml" [--resume 'checkpoint.pth.tar']
```
- Config paths are relative to `configs/` directory (no need for full path)
- Checkpoints save to `{data_dir}/ckpts/{dataset}_ddpm_*.pth.tar`
- Validation patches save every `validation_freq` steps to `args.image_folder`

### Evaluation
```bash
python eval_diffusion.py --config "allweather.yml" --resume 'model.pth.tar' \
    --test_set 'raindrop' --sampling_timesteps 25 --grid_r 16
```
- `--test_set` options: `raindrop`, `snow`, `rainfog` (corresponds to validation dataset splits)
- `--grid_r`: Patch overlap stride (4 or 8 for best quality, 16 for speed)
- `--sampling_timesteps`: DDIM sampling steps (25 is good balance; 1000 is full DDPM)

### Metrics Calculation
Use `calculate_psnr_ssim.py` after generating outputs - requires updating paths to ground truth and results directories.

## Configuration File Structure

YAML configs have nested sections matching code namespaces:
- `data`: dataset name (must match `datasets/__dict__` key), image_size, data_dir
- `model`: U-Net architecture (channels, multipliers, attention resolutions)
- `diffusion`: beta schedule parameters, num_diffusion_timesteps (always 1000 for training)
- `training`: patch_n (patches per image), batch_size, n_epochs
- `optim`: Adam parameters

**Critical**: `data.data_dir` must point to parent directory containing `data/` subdirectory with datasets.

## Common Pitfalls

1. **Config paths**: Use filename only (e.g., `"allweather.yml"`) not full path - script auto-prefixes `configs/`
2. **EMA requirement**: Always use `ema=True` when loading for inference; EMA weights provide better results
3. **Single GPU only**: Multi-GPU works for training but evaluation currently supports single GPU only
4. **Manual batching**: When processing many patches, `manual_batching=True` batches 64 patches at a time to avoid OOM
5. **Conditional flag**: Config must have `data.conditional: True` for weather restoration (vs unconditional generation)

## Code Attribution
Adapted from:
- DDIM (sampling schedule)
- DDRM (conditional restoration)
- SwinIR (metrics calculation)

Always preserve attribution comments in files.
