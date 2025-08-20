# DisplayVQA
This repository contains the model proposed in the paper "Subjective and Objective Quality Assessment of Display Content Videos." It is coming soon.

## Features

- **Swin Transformer Backbone**: Spatial feature extraction
- **Modular Architecture**: Separate spatial and temporal quality rectifiers
- **Enhanced Laplacian Pyramid**: Multi-scale spatial feature processing
- **Color Attention Mechanism**: Enhanced color perception
- **Motion Feature Integration**: SlowFast network-based temporal features
- **Multi-Database Support**: 8K-Pro, KoNViD-1k, LiveVQC

## Architecture

```
The model's architecture will be available soon.
```

## Project Structure

```
DisplayVQA/
├── model.py                    # VQA model architecture
├── dataset.py                  # Dataset loading
├── trainer.py                  # Training framework
├── utils.py                    # Utility functions
├── preprocessing/
│   ├── frame_extractor.py      # Video frame extraction
│   ├── laplacian_extractor.py  # Laplacian pyramid features
│   └── motion_extractor.py     # SlowFast motion features
└── README.md
```

## Quick Start


### Data Preparation

1. **Extract Video Frames**:
```bash
python preprocessing/frame_extractor.py \
    --video_dir /path/to/videos \
    --output_dir /path/to/frames \
    --annotation_file /path/to/annotations.csv \
    --target_fps 1 \
    --max_frames 8
```

2. **Extract Laplacian Features**:
```bash
python preprocessing/laplacian_extractor.py \
    --database 8kpro \
    --frame_batch_size 8 \
    --num_pyramid_levels 6
```

3. **Extract Motion Features**:
```bash
python preprocessing/motion_extractor.py \
    --database 8kpro \
    --target_size 224 \
    --num_frames 32
```

### Training

```bash
python trainer.py \
    --database 8kpro_indep \
    --model_name DisplayVQA_Modular \
    --learning_rate 1e-5 \
    --num_epochs 30 \
    --batch_size 16 \
    --loss_type plcc
```

## Model Configuration

### DisplayVQA Model Parameters

- **Backbone**: Swin Transformer Base (1024-dim features)
- **Sequence Length**: 5-10 frames (database dependent)
- **Spatial Enhancement**: Laplacian pyramid with 5×384 features per frame
- **Temporal Enhancement**: SlowFast features (256-dim Fast pathway)
- **Dropout**: Configurable spatial (0.2) and temporal (0.2) dropout rates

### Supported Databases

| Database | Frames | Resolution | Domain |
|----------|--------|------------|---------|
| 8K-Pro | 5 | 4K | Professional content |
| KoNViD-1k | 8 | Various | User-generated content |
| LiveVQC | 10 | Various | Live streaming |


## Model Components

### 1. Swin Transformer
- Pre-trained Swin-B backbone for spatial feature extraction
- Global average pooling for frame-level representations
- 1024-dimensional feature vectors

### 2. Spatial Quality Module
- Processes enhanced Laplacian pyramid features
- Multi-scale spatial analysis with edge detection
- Color attention mechanism for enhanced perception

### 3. Temporal Quality Module
- Integrates SlowFast motion features
- Temporal dynamics modeling
- Fast pathway (256-dim) for fine temporal details

### 4. Quality Fusion
- Modular combination of spatial and temporal enhancements
- Learnable scale and bias parameters
- Final quality prediction through geometric mean



For questions or issues, please contact: your.email@example.com
