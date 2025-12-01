# FreeSeg Documentation

Welcome to the FreeSeg documentation! This directory contains documentation for the FreeSeg package.

## Documentation Index

### [Getting Started Guide](GETTING_STARTED.md)
A beginner guide to get you up and running with FreeSeg quickly. Includes installation instructions, quick start examples, and common workflows.

**Start here if you're new to FreeSeg!**

### [API Documentation](API.md)
API reference for all modules, classes, and functions in FreeSeg. Includes:
- Core modules (Config, Training, Prediction, Evaluation)
- Models (U-Net architecture)
- Datasets (SegmentationDataset)
- Augmentation (AugmentBase, AugmentVoxynth)
- Metrics (Dice, DiceLoss, WeightedL2Loss)
- Utilities and helper functions

**Use this for detailed API information and code examples.**

### [Configuration Guide](CONFIGURATION.md)
Comprehensive guide to configuring FreeSeg. Covers:
- Configuration file structure
- Dataset configuration
- Model architecture parameters
- Preprocessing and augmentation settings
- Training hyperparameters
- Command-line arguments

**Reference this when setting up your training configuration.**

### [Training Guide](TRAINING.md)
Detailed guide to training models with FreeSeg. Includes:
- Training workflow
- Two-stage training (Weighted L2 + Dice Loss)
- Training options and parameters
- Monitoring training (logs, TensorBoard)
- Resuming training from checkpoints

**Read this for training best practices and troubleshooting.**

### [Architecture Documentation](ARCHITECTURE.md)
System architecture and design documentation. Covers:
- System overview
- U-Net architecture
- Augmentation pipeline
- Training pipeline
- Inference pipeline
- Extension points

**Use this to understand the system design and extend FreeSeg.**

## Quick Links

- **Installation**: See [Getting Started Guide - Installation](GETTING_STARTED.md#installation)
- **First Training**: See [Getting Started Guide - Your First Training](GETTING_STARTED.md#your-first-training)
- **Configurations**: See [Configuration Guide](CONFIGURATION.md)
- **Training Options**: See [Training Guide - Training Options](TRAINING.md#training-options)
- **API Reference**: See [API Documentation](API.md)

## Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── GETTING_STARTED.md     # Quick start guide
├── API.md                 # API reference
├── CONFIGURATION.md       # Configuration guide
├── TRAINING.md            # Training guide
└── ARCHITECTURE.md        # Architecture documentation
```

## Getting Help

- **Documentation**: Browse the documentation files in this directory
- **Examples**: Check the `tutorials/` directory for example workflows
- **Configuration**: See `configs/` for example configuration files
- **Issues**: Report issues on the GitHub repository

## Additional Resources

- **FreeSurfer**: https://freesurfer.net/
- **PyTorch**: https://pytorch.org/
- **TensorBoard**: https://www.tensorflow.org/tensorboard
- **SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining** \
B. Billot, D.N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A.V. Dalca, J.E. Iglesias \
Medical Image Analysis (2023) \
[ [article](https://www.sciencedirect.com/science/article/pii/S1361841523000506) | [arxiv](https://arxiv.org/abs/2107.09559)
]


---

**Happy segmenting with FreeSeg!**

