# FreeSeg Documentation

Welcome to the FreeSeg documentation! This directory contains comprehensive documentation for the FreeSeg package.

## Documentation Index

### [Getting Started Guide](GETTING_STARTED.md)
A beginner-friendly guide to get you up and running with FreeSeg quickly. Includes installation instructions, quick start examples, and common workflows.

**Start here if you're new to FreeSeg!**

### [API Documentation](API.md)
Complete API reference for all modules, classes, and functions in FreeSeg. Includes:
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
- Best practices
- Troubleshooting common issues

**Read this for training best practices and troubleshooting.**

### [Architecture Documentation](ARCHITECTURE.md)
System architecture and design documentation. Covers:
- System overview
- U-Net architecture details
- Data pipeline
- Training pipeline
- Inference pipeline
- Module structure
- Extension points

**Use this to understand the system design and extend FreeSeg.**

## Quick Links

- **Installation**: See [Getting Started Guide - Installation](GETTING_STARTED.md#installation)
- **First Training**: See [Getting Started Guide - Your First Training](GETTING_STARTED.md#your-first-training)
- **Configuration Examples**: See [Configuration Guide](CONFIGURATION.md)
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

## Contributing to Documentation

If you find errors or want to improve the documentation:

1. Check existing documentation first
2. Follow the existing documentation style
3. Include code examples where helpful
4. Keep documentation up to date with code changes

## Getting Help

- **Documentation**: Browse the documentation files in this directory
- **Examples**: Check the `tutorials/` directory for example workflows
- **Configuration**: See `configs/` for example configuration files
- **Issues**: Report issues on the GitHub repository

## Additional Resources

- **FreeSurfer**: https://freesurfer.net/
- **PyTorch**: https://pytorch.org/
- **TensorBoard**: https://www.tensorflow.org/tensorboard

---

**Happy segmenting with FreeSeg!**

