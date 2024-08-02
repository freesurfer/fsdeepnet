import os
import logging
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import make_grid

from utils.metrics import DiceScore
from utils.data_utils import remap_labels, onehot


class Training:
    """
    This class trains/validates the model.

    Attributes
    ----------

    Methods
    -------
    train_model
        run the training loop
    """
        
    def __init__(self,
                 train_output_folder,
                 labels_segmentation,
                 label_mapping,
                 validation_loader=None,
                 checkpoint=None,
                 best_model_metric="dice",                 
                 write_tensorboard_summary=False,
                 device=None):
        """
        Training Constructor.

        Parameters
        ----------
        train_output_folder : string
            path of a directory where the models will be saved during training.
        labels_segmentation : 1d numpy array
            List of labels for which to compute Dice scores. 
            It should be the same list as the segmentation_labels used in training.
        validation_loader : DataLoader
            (optional) validation DataLoader
        checkpoint : string
            (optional) path of an already saved model to load before starting the training

        """

        self._checkpoint = checkpoint        
        self._validation_loader = validation_loader
        self._best_model_metric = best_model_metric
        
        self._setup_training_directory(train_output_folder)
        
        self._labels_segmentation, self._unique_idx = np.unique(labels_segmentation, return_index=True)
        self._nlabels = len(self._labels_segmentation)
        self._label_mapping = label_mapping  # ??? todo: calculate this from labels_segmentation

        self._model = None
        self._unet_levels = None

        self._device = device
        if (self._device is None):
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._summary_writer = None
        if (write_tensorboard_summary):
            # Create TensorBoard writer
            from torch.utils.tensorboard import SummaryWriter
            self._summary_writer = SummaryWriter(train_output_folder)
            


    def _setup_training_directory(self, train_output_folder):
        self._best_model_dir = os.path.join(train_output_folder, "best_models")  # Folder for best models
        self._checkpoint_dir = os.path.join(train_output_folder, "checkpoints")  # Folder for checkpoints
        self._dice_dir = os.path.join(self._checkpoint_dir, "dices")             # Folder for training/validation dices
        
        os.makedirs(self._best_model_dir, exist_ok=True)
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        os.makedirs(self._dice_dir, exist_ok=True)


    def train_model(self,
                    input_generator,
                    model,
                    lr,
                    epochs,
                    steps_per_epoch,
                    metric_type,
                    loss_fn):
        """
        model training loop

        Parameters
        ----------
        input_generator : input_generator
            training data generator
        lr : float
            learning rate for the training.
        epochs : int
            number of epochs for which the network is trained.
        steps_per_epoch : int
            number of steps per epoch. Default is 1000. This is equivalent to the frequency at which the models are saved.
        metric_type : string
            wl2 or dice
        """
            
        self._input_generator = input_generator
        self._model = model

        self._dice_metric_hard = DiceScore(
            num_classes=self._nlabels,
            input_type="prob",
            dice_type="hard",
            # return_loss=False,
        )
        
        start_epoch = 0
        best_validation_loss = float("inf")
        best_validation_dice = 0.0

        # define optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
                    
        # load checkpoint if provided
        if (self._checkpoint is not None):
            """            
            # load the network weights
            state_dict = torch.load(path_model)
            self._model.load_state_dict(state_dict["model_state_dict"])
            """
            from utils.train_utils import load_checkpoint
            start_epoch, model_metric_type, best_validation_loss, best_validation_dice = load_checkpoint(
                self._checkpoint, self._model, optimizer
            )
            if (model_metric_type != metric_type):
                return
            
            logging.info(f"Resuming training from checkpoint: {self._checkpoint}")
            self._checkpoint = None  # the checkpoint will only be used once in the training

        # training loop
        for epoch in range(start_epoch, epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            (train_loss, train_dices)  = self._train_one_epoch(optimizer, loss_fn, epoch, steps_per_epoch,
                                                               metric_type=metric_type)
            
            train_loss /= steps_per_epoch
            train_dice_avg = np.mean(train_dices)
        
            # output training dices (n_labels x steps_per_epoch)
            f_dice_scores = os.path.join(self._dice_dir, f"train_dices_epoch{epoch+1}.npy")
            np.save(f_dice_scores, train_dices)
    
            if (self._validation_loader is None):
                logging.info(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Dice Avg: {train_dice_avg:.4f}"
                )
        
                # model dict
                checkpoint_dict = {
                    "epoch": epoch,
                    "metric_type": metric_type,
                    "model_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "dice": train_dice_avg,
                }
                # Save checkpoints every steps_per_epoch steps
                checkpoint_path = os.path.join(
                    self._checkpoint_dir,
                    f"{metric_type}_{epoch+1:03d}_train_loss{train_loss:.4f}_train_dice{train_dice_avg:.4f}.pth",
                )
                logging.info(f"Epoch {epoch+1}: saving model to {checkpoint_path}")
                torch.save(checkpoint_dict, checkpoint_path)
            else:
                # perform validation
                (validation_loss, validation_dices) = self._validate(optimizer, loss_fn, epoch, metric_type=metric_type)
                validation_loss /= len(self._validation_loader)
                validation_dice_avg = np.mean(validation_dices)
        
                # output validation dices (n_labels x len(self._validation_loader))
                f_dice_scores = os.path.join(self._dice_dir, f"validation_dices_epoch{epoch+1}.npy")
                np.save(f_dice_scores, validation_dices)

                logging.info(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Dice Avg: {train_dice_avg:.4f}, "
                    f"Val Loss: {validation_loss:.4f}, "
                    f"Val Dice Avg: {validation_dice_avg:.4f}"
                )

                # model dict
                checkpoint_dict = {
                    "epoch": epoch,
                    "metric_type": metric_type,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                    "dice": validation_dice_avg,
                }

                # pick and save the best model
                if self._best_model_metric == "loss":
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        checkpoint_path = os.path.join(
                            self._best_model_dir,
                            f"best_model_epoch{epoch+1}_val_loss{validation_loss:.4f}_val_dice{validation_dice_avg:.4f}.pth",
                        )
                        torch.save(checkpoint_dict, checkpoint_path)
                elif self._best_model_metric == "dice":
                    if validation_dice_avg > best_validation_dice:
                        best_validation_dice = validation_dice_avg
                        checkpoint_path = os.path.join(
                            self._best_model_dir,
                            f"best_model_epoch{epoch+1}_val_loss{validation_loss:.4f}_val_dice{validation_dice_avg:.4f}.pth",
                        )
                        torch.save(checkpoint_dict, checkpoint_path)
                
                # Save checkpoints every steps_per_epoch steps
                checkpoint_path = os.path.join(
                    self._checkpoint_dir,
                    f"{metric_type}_{epoch+1:03d}_val_loss{validation_loss:.4f}_val_dice{validation_dice_avg:.4f}.pth",
                )
                logging.info(f"Epoch {epoch+1}: saving model to {checkpoint_path}")
                torch.save(checkpoint_dict, checkpoint_path)
            # End of perform evaluation
        # End of training loop

            
    def _train_one_epoch(self, optimizer, loss_fn, epoch, steps_per_epoch, metric_type='dice'):
        """
        train one epoch

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            optimizer
        loss_fn   : nn.Module
            loss function
            
        Returns
        -------
        train_loss: float
            total training loss
        """

        train_loss = 0.0
        train_dices = np.zeros((self._nlabels, steps_per_epoch))

        self._model.train()        
        for step in range(steps_per_epoch):
            (batch_idx, images, labels) = next(self._input_generator)

            # Zero your gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            (outputs, penultimate) = self._model(images)

            # Compute the loss and its gradients
            if (metric_type == 'wl2'):
                loss = loss_fn(penultimate, labels)
            elif (metric_type == 'dice'):
                loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_loss += loss.item()

            # --- Metrics Calculation ---
            # Calculate hard Dice
            batch_hard_dice = self._dice_metric_hard(outputs, labels)
            train_dices[:, step] = batch_hard_dice.detach().cpu().numpy()
            logging.info(f"  {step+1}/{steps_per_epoch} loss: {loss.item():.4f}, dice avg: {np.mean(train_dices[:, step]):.4f}")

            if (self._summary_writer is not None):
                # Write to TensorBoard every batch
                self._summary_writer.add_scalar("Train/Loss", loss.item(), epoch * steps_per_epoch + batch_idx)
                self._summary_writer.add_scalar(
                    "Train/Dice",
                    np.mean(train_dices[:, step]),
                    #torch.mean(torch.tensor(batch_hard_dice)),
                    epoch * steps_per_epoch + batch_idx,
                )

                # --- TensorBoard Visualization (Inside Training Loop) ---
                if batch_idx % 10 == 0:  # Visualize every 10 batches (adjust as needed)
                    slice_index = random.randint(20, 50)  # Choose representative slice index
                    num_examples_to_visualize = min(images.size(0), 6)  # visualize upto 6 examples

                    # Get slices from different examples in the batch
                    image_slices = images[:num_examples_to_visualize, 0, slice_index].cpu()
                    label_slices = labels[:num_examples_to_visualize, 0, slice_index].cpu()
                    output_slices = outputs[:num_examples_to_visualize, 0, slice_index].cpu()

                    # Create grids (using a colormap if needed)
                    image_grid = make_grid(image_slices, nrow=num_examples_to_visualize, cmap="viridis")
                    label_grid = make_grid(label_slices, nrow=num_examples_to_visualize, cmap="viridis")
                    output_grid = make_grid(output_slices, nrow=num_examples_to_visualize, cmap="viridis")

                    # Add images to TensorBoard
                    self._summary_writer.add_image("Train/Input Image", image_grid, epoch * steps_per_epoch + batch_idx)
                    self._summary_writer.add_image("Train/Label", label_grid, epoch * steps_per_epoch + batch_idx)
                    self._summary_writer.add_image(
                        "Train/Predicted Output", output_grid, epoch * steps_per_epoch + batch_idx
                    )
            # end of tensorboard summary writer
        # end of training steps_per_epoch steps

        return train_loss, train_dices


    def _validate(self, optimizer, loss_fn, epoch, metric_type='dice'):
        """
        evaluate current model

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            optimizer
        loss_fn   : nn.Module
            loss function
            
        Returns
        -------
        validation_loss: float
            total validation loss
        """
        
        validation_loss = 0.0
        validation_dices = np.zeros((self._nlabels, len(self._validation_loader)))
        
        self._model.eval()        
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self._validation_loader):
                images, labels = images.to(self._device).float(), labels.to(self._device)
                labels = remap_labels(labels, self._label_mapping)
                labels = onehot(labels, num_classes=self._nlabels, device=self._device)

                (outputs, penultimate) = self._model(images)

                if (metric_type == 'wl2'):
                    loss = loss_fn(penultimate, labels)
                elif (metric_type == 'dice'):
                    loss = loss_fn(outputs, labels)

                validation_loss += loss.item()

                # --- Metrics Calculation ---
                # Calculate hard Dice
                batch_hard_dice = self._dice_metric_hard(outputs, labels)
                validation_dices[:, batch_idx] = batch_hard_dice.detach().cpu().numpy()
                logging.info(f"  validation {batch_idx}/{len(self._validation_loader)} val loss: {loss.item():.4f}, val dice avg: {np.mean(validation_dices[:, batch_idx]):.4f}")

                if (self._summary_writer is not None):
                    # Write validation loss and Dice to TensorBoard (once per epoch)
                    self._summary_writer.add_scalar(
                        "Validation/Loss", validation_loss, epoch * len(self._validation_loader) + batch_idx
                    )
                    self._summary_writer.add_scalar(
                        "Validation/Dice",
                        np.mean(validation_dices[:, batch_idx]),
                        #torch.mean(torch.tensor(batch_hard_dice)),
                        epoch * len(self._validation_loader) + batch_idx,
                    )

                    # --- TensorBoard Visualization (Inside Validation Loop) ---
                    if batch_idx % 3 == 0:  # Visualize every 3 batches (adjust as needed)
                        slice_index = random.randint(20, 50)
                        num_examples_to_visualize = min(images.size(0), 6)  # visualize upto 6 examples

                        # Get slices from different examples in the batch
                        image_slices = images[:num_examples_to_visualize, 0, slice_index].cpu()
                        label_slices = labels[:num_examples_to_visualize, 0, slice_index].cpu()
                        output_slices = outputs[:num_examples_to_visualize, 0, slice_index].cpu()

                        # Create grids (using a colormap if needed)
                        image_grid = make_grid(image_slices, nrow=num_examples_to_visualize, cmap="viridis")
                        label_grid = make_grid(label_slices, nrow=num_examples_to_visualize, cmap="viridis")
                        output_grid = make_grid(
                            output_slices, nrow=num_examples_to_visualize, cmap="viridis"
                        )

                        # Add images to TensorBoard
                        self._summary_writer.add_image(
                            "Validation/Input Image", image_grid, epoch * len(self._validation_loader) + batch_idx
                        )
                        self._summary_writer.add_image(
                            "Validation/Label", label_grid, epoch * len(self._validation_loader) + batch_idx
                        )
                        self._summary_writer.add_image(
                            "Validation/Predicted Output",
                            output_grid,
                            epoch * len(self._validation_loader) + batch_idx,
                        )
                # end of tensorboard summary writer
        # End of validation loop

        return validation_loss, validation_dices

