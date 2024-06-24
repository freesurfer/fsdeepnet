import os
import json
import torch
import logging
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra

import argparse
from time import time
from utils.dataset import load_datasets
from models.model import UNet3D
from utils.data_utils import load_config, load_volume, save_volume, remap_labels
from utils.data_utils import onehot
from utils.metrics import DiceScore

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def evaluate(cfg: DictConfig):
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load datasets
    _, _, test_dataset = load_datasets(cfg)
    test_loader = DataLoader(test_dataset, batch_size=cfg.evaluation.batch_size, shuffle=False)

     # Load label mapping
    with open(os.path.join(os.path.dirname(cfg.evaluation.model_checkpoint), "label_mapping.json"), "r") as f:
        label_mapping = json.load(f)

    # Invert the label mapping for remapping predictions back to original labels
    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    # Load model
    model = UNet3D(
        input_shape=(cfg.dataset.expected_num_channels, *cfg.dataset.crop_size),
        ndims=cfg.model.ndims,
        nb_features=cfg.model.nb_features,
        nb_levels=cfg.model.nb_levels,
        nb_labels=len(cfg.dataset.label_mapping),
        feat_mult=cfg.model.feat_mult,
        nb_conv_per_level=cfg.model.nb_conv_per_level,
        use_residuals=cfg.model.use_residuals,
        use_batchnorm=cfg.model.use_batchnorm,
        activation=cfg.model.activation,
        final_pred_activation="softmax",
    ).to(device)

    checkpoint = torch.load(cfg.evaluation.model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dice_metric = DiceScore(
        num_classes=len(cfg.dataset.label_mapping),
        input_type="prob",
        dice_type="hard",
        ignore_indexes=cfg.training.ignore_indexes,
    )

    total_dice_scores = torch.zeros(len(cfg.dataset.label_mapping), device=device)
        
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Remap labels to model's expected format
            labels = remap_labels(labels, label_mapping)
            labels = onehot(labels, num_classes=len(label_mapping), device=device)
            
            outputs = model(images)
            dice_scores = dice_metric(outputs, labels)
            
            total_dice_scores += dice_scores.sum(dim=0)
            
            # Save predictions
            predicted_labels = torch.argmax(outputs, dim=1)
            
            # Remap predictions back to original label space
            predicted_labels_remapped = torch.zeros_like(predicted_labels)
            for model_label, original_label in inv_label_mapping.items():
                predicted_labels_remapped[predicted_labels == model_label] = int(original_label)
            
            for i, pred in enumerate(predicted_labels_remapped):
                save_volume(
                    pred,
                    test_dataset.image_files[idx * cfg.evaluation.batch_size + i],
                    os.path.join(cfg.output_dir, f"prediction_{idx}_{i}.nii.gz")
                )
            
            log.info(f"Processed batch {idx+1}/{len(test_loader)}")

    average_dice_scores = total_dice_scores / len(test_dataset)
    
    for i, score in enumerate(average_dice_scores):
        original_label = inv_label_mapping[str(i)]  # Convert i to string as JSON keys are strings
        log.info(f"Average Dice score for class {original_label}: {score.item():.4f}")

    log.info("Evaluation completed.")

if __name__ == "__main__":
    evaluate()