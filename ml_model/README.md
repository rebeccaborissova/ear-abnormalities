# Ear Landmark Training Pipeline

A streamlined pipeline for training ear landmark detection models with transfer learning from adult to infant ears.

## Overview

This pipeline automates the 4-stage training process:
1. **Adult Training**: Train on 55 landmarks
2. **Transfer 55→22**: Reduce to 22 landmarks for infant ears
3. **Expand 22→23**: Add one additional landmark
4. **Infant Training**: Fine-tune on infant dataset

## Quick Start

### Run the Full Pipeline
```bash
python pipeline.py --config config.yaml
```

### Run Specific Stages
```bash
# Only run transfer stages (skip training)
python pipeline.py --config config.yaml --stages transfer_55_22 transfer_22_23 infant

# Skip adult training (use existing checkpoint)
python pipeline.py --config config.yaml --skip adult
```

### Customize Configuration
Edit `config.yaml` to adjust:
- Learning rates, batch sizes, epochs
- GPU selection
- Dataset paths
- Checkpoint names

## Configuration File (`config.yaml`)

The YAML config centralizes all parameters:

```yaml
# Adult training (55 landmarks)
adult_training:
  enabled: true
  num_landmarks: 55
  dataset_path: "/path/to/dataset/train"
  batch_size: 8
  num_epochs: 100
  learning_rate: 1e-4
  # ... more parameters

# Transfer configurations
transfer_55_to_22:
  enabled: true
  landmark_mapping:  # Which adult landmarks map to infant
    0: 0
    1: 1
    # ...

# Infant training (23 landmarks)
infant_training:
  enabled: true
  num_landmarks: 23
  batch_size: 4
  # ... more parameters

# GPU settings
gpu:
  cuda_visible_devices: "7"
```

## Command-Line Options

```bash
python pipeline.py [OPTIONS]

Options:
  --config PATH              Path to YAML config (default: config.yaml)
  
  --stages {adult,transfer_55_22,transfer_22_23,infant} [...]
                            Run specific stages only
  
  --skip {adult,transfer_55_22,transfer_22_23,infant} [...]
                            Skip specific stages

Examples:
  # Full pipeline
  python pipeline.py
  
  # Use existing adult checkpoint
  python pipeline.py --skip adult
  
  # Only run transfers
  python pipeline.py --stages transfer_55_22 transfer_22_23
  
  # Custom config file
  python pipeline.py --config experiments/config_v2.yaml
```

## Output Files

The pipeline generates:
- `ear_landmark_model_best.pth` - Best adult model
- `infant_ear_model_init.pth` - 22-landmark transfer
- `infant_ear_model_23lm_init.pth` - 23-landmark expansion
- `infant_ear_model_23lm_best_v2.pth` - Final infant model
- Checkpoint files every N epochs (configurable)

All checkpoint names are customizable in `config.yaml`.
