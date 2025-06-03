import os
import logging
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import make_grid

from freeseg.checkpoint import Checkpoint
from freeseg.metrics import DiceScore
from freeseg.utils import remap_labels, DataGenerator, save_framedimage, gpu_report


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
                 train_loader,  # torch.utils.data.DataLoader              
                 model,
                 model_arch_dict=None,
                 train_dataset_dict=None,
                 ctab=None,          # ascii color table
                 label_lookup=None,  # surfa.core.labels.LabelLookup
                 model_checkpoint=None,
                 validation_loader=None,                 
                 best_model_metric="dice",                 
                 write_tensorboard_summary=False,
                 device=None,
                 gpu_index=None,
                 preprocessing_device=None,
                 debug=False):
        """
        Training Constructor.

        Parameters
        ----------
        train_output_folder : string
            path of a directory where the models will be saved during training.
        train_loader : torch.utils.data.DataLoader
            DataLoader to create the training data generator
        validation_loader : DataLoader
            (optional) validation DataLoader
        model_checkpoint : string
            (optional) path of an already saved model to load before starting the training

        """

        self._debug = debug
        self._model = model
        self._model_arch_dict = model_arch_dict
        self._train_dataset_dict = train_dataset_dict
        self._model_checkpoint = model_checkpoint
        self._ctab = ctab
        self._validation_loader = validation_loader
        self._best_model_metric = best_model_metric
        self._batch_size = train_dataset_dict["batch_size"]

        self._setup_training_directory(train_output_folder)

        labels_segmentation = train_dataset_dict["segmentation_labels"]
        self._num_labels = len(labels_segmentation)
        self._label_mapping = train_dataset_dict["label_mapping"]
        self._inverse_label_mapping = train_dataset_dict["inverse_label_mapping"]

        self._device = device
        self._gpu_index = gpu_index
        self._preprocessing_device = preprocessing_device
        if (self._device is None):
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (self._preprocessing_device is None):
            self._preprocessing_device = self._device
        if (self._gpu_index is None and torch.cuda.is_available()):
            self._gpu_index = torch.cuda.current_device()
            
        self._input_generator = DataGenerator(train_loader, self._preprocessing_device)        

        self._summary_writer = None
        if (write_tensorboard_summary):
            # Create TensorBoard writer
            from torch.utils.tensorboard import SummaryWriter
            self._summary_writer = SummaryWriter(train_output_folder)
            
        self._dice_metric_hard = DiceScore(
            num_classes=self._num_labels,
            dice_type="hard",
            # return_loss=False,
        )

        self._label_lookup = None
        if (self._ctab is not None):
            import surfa as sf
            self._label_lookup = sf.load_label_lookup(self._ctab)
        else:
            self._label_lookup = label_lookup

        # create Checkpoint object
        self._checkpoint = Checkpoint(model_arch_dict=self._model_arch_dict, label_lookup=self._label_lookup, train_dataset_dict=self._train_dataset_dict)        


    def _setup_training_directory(self, train_output_folder):
        self._best_model_dir = train_output_folder
        self._checkpoint_dir = train_output_folder
        self._dice_dir = os.path.join(self._checkpoint_dir, "dices")             # Folder for training/validation dices        

        os.makedirs(self._best_model_dir, exist_ok=True)
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        os.makedirs(self._dice_dir, exist_ok=True)
        if (self._debug):
            self._debug_dir = os.path.join(train_output_folder, "debug")
            os.makedirs(self._debug_dir, exist_ok=True)


    def train_model(self, lr, epochs, steps_per_epoch, metric_type, optimizer_cls, loss_fn):
        """
        model training loop

        Parameters
        ----------
        lr : float
            learning rate for the training.
        epochs : int
            number of epochs for which the network is trained.
        steps_per_epoch : int
            number of steps per epoch. Default is 1000. This is equivalent to the frequency at which the models are saved.
        metric_type : string
            wl2 or dice
        """

        start_epoch = 0
        end_epoch = epochs
        best_validation_loss = float("inf")
        best_validation_dice = 0.0

        # set up optimizer
        optimizer = optimizer_cls(self._model.parameters(), lr=lr)

        # load checkpoint if provided
        if (self._model_checkpoint is not None):
            self._checkpoint.load(self._model_checkpoint, self._model, optimizer, self._device)
            # iterate through param_groups and update the learning rate:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            if (self._checkpoint.metric_type is not None and self._checkpoint.metric_type != metric_type):
                return
            
            start_epoch = self._checkpoint.epoch + 1
            end_epoch = start_epoch + epochs
            logging.info(f"Continue to train {end_epoch-start_epoch} ({epochs}) {self._checkpoint.metric_type} epochs, lr:{lr}")
            if (self._label_lookup is None):
                self._label_lookup = self._checkpoint.label_lookup
            self._model_checkpoint = None  # the checkpoint will only be used once in the training

        # training loop
        ncols = 2 if (self._validation_loader is None) else 4
        loss_dice_avg = np.zeros((end_epoch-start_epoch, ncols))        
        for epoch in range(start_epoch, end_epoch):
            if (self._gpu_index is not None):
                gpu_report(self._gpu_index)
            logging.info(f"Epoch {epoch+1:3d}/{end_epoch:<3d}")
            (train_loss, train_dices)  = self._train_one_epoch(optimizer, loss_fn, epoch, steps_per_epoch,
                                                               metric_type=metric_type)
            
            train_loss /= steps_per_epoch
            train_dice_avg = np.mean(train_dices)
        
            # output training dices (n_labels x steps_per_epoch)
            f_dice_scores = os.path.join(self._dice_dir, f"train_{metric_type}_{epoch+1:03d}.npy")
            np.save(f_dice_scores, train_dices)
            f_dice_dat = os.path.join(self._dice_dir, f"d.train_{metric_type}_{epoch+1:03d}.dat")
            # Save in text format as nepochs x nlabels
            np.savetxt(f_dice_dat, np.transpose(np.squeeze(train_dices)))
    
            if (self._validation_loader is None):
                loss_dice_avg[epoch-start_epoch] = np.array((train_loss, train_dice_avg))                
                logging.info(
                    f"Epoch [{epoch+1}/{end_epoch}], "
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
                    f"{metric_type}_{epoch+1:03d}.pth",
                )
                logging.info(f"Epoch {epoch+1}: saving model to {checkpoint_path}")
                self._checkpoint.save(checkpoint_path, checkpoint_dict)
            else:
                # perform validation
                (validation_loss, validation_dices) = self._validate(optimizer, loss_fn, epoch, metric_type=metric_type)
                validation_loss /= len(self._validation_loader)
                validation_dice_avg = np.mean(validation_dices)
        
                # output validation dices (n_labels x len(self._validation_loader))
                f_dice_scores = os.path.join(self._dice_dir, f"validation_{metric_type}_{epoch+1:03d}.npy")
                np.save(f_dice_scores, validation_dices)
                f_dice_dat = os.path.join(self._dice_dir, f"d.validation_{metric_type}_{epoch+1:03d}.dat")
                # Save in text format as nsubjects x nlabels
                np.savetxt(f_dice_dat, np.transpose(np.squeeze(validation_dices)))

                loss_dice_avg[epoch-start_epoch] = np.array((train_loss, train_dice_avg, validation_loss, validation_dice_avg))                
                logging.info(
                    f"Epoch [{epoch+1}/{end_epoch}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Dice Avg: {train_dice_avg:.4f}, "
                    f"Val Loss: {validation_loss:.4f}, "
                    f"Val Dice Avg: {validation_dice_avg:.4f}"
                )

                # model dict
                checkpoint_dict = {
                    "epoch": epoch,
                    "metric_type": metric_type,
                    "model_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                    "dice": validation_dice_avg,
                }
                # Save checkpoints every steps_per_epoch steps
                checkpoint_path = os.path.join(
                    self._checkpoint_dir,
                    f"{metric_type}_{epoch+1:03d}.pth",
                )
                logging.info(f"Epoch {epoch+1}: saving model to {checkpoint_path}")
                self._checkpoint.save(checkpoint_path, checkpoint_dict)                

                # pick and save the best model
                if self._best_model_metric == "loss":
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        checkpoint_path = os.path.join(
                            self._best_model_dir,
                            f"best_loss_model_{metric_type}_{epoch+1:03d}.pth",
                        )
                        logging.info(f"Epoch {epoch+1}: best {self._best_model_metric} model saved: {checkpoint_path}")
                        self._checkpoint.save(checkpoint_path, checkpoint_dict)
                elif self._best_model_metric == "dice":
                    if validation_dice_avg > best_validation_dice:
                        best_validation_dice = validation_dice_avg
                        checkpoint_path = os.path.join(
                            self._best_model_dir,
                            f"best_dice_model_{metric_type}_{epoch+1:03d}.pth",
                        )
                        logging.info(f"Epoch {epoch+1}: best {self._best_model_metric} model saved: {checkpoint_path}")
                        self._checkpoint.save(checkpoint_path, checkpoint_dict)
            # End of perform evaluation
        # End of training loop
        
        f_loss_dice_avg_dat = os.path.join(self._checkpoint_dir, f"train_validation_avg_{metric_type}_epoch{start_epoch+1}-{end_epoch}.dat")
        # Save in text format as nepoch x 2
        np.savetxt(f_loss_dice_avg_dat, loss_dice_avg)

            
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
        train_dices = np.zeros((self._batch_size, self._num_labels, steps_per_epoch))

        self._model.train()        
        for step in range(steps_per_epoch):
            (batch_idx, images, onehot_labels, priors, dataset_indices) = next(self._input_generator)
            # training device and preprocessing device could be different
            images, onehot_labels, priors = images.to(self._device), onehot_labels.to(self._device), priors.to(self._device)
            
            # Zero your gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            (outputs, penultimate) = self._model(images, priors)

            # Compute the loss and its gradients
            if (metric_type == 'wl2'):
                loss = loss_fn(penultimate, onehot_labels)
            elif (metric_type == 'dice'):
                loss = loss_fn(outputs, onehot_labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_loss += loss.item()

            # --- Metrics Calculation ---
            # Calculate hard Dice
            batch_hard_dice = self._dice_metric_hard(outputs, onehot_labels)
            train_dices[:, :, step] = batch_hard_dice.detach().cpu().numpy()
            logging.info(f"  {step+1:4d}/{steps_per_epoch:<4d} loss: {loss.item():.4f}, dice avg: {np.mean(train_dices[:, :, step]):.4f}")
            if (self._debug and step == steps_per_epoch-1):
                # begin of debugging volumes output
                # output augmented images/labels/priors, onehot encoded labels, posteriors, prediciton from each batch (batch_size x [C, H, W(, D)])
                logging.debug(f"output augmented images/labels, onehot encoded labels, posteriors, prediciton ...")                
                for n, idx in enumerate(dataset_indices):
                    out_image = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.augmented_image.mgz")
                    save_framedimage(images[n], out_image)

                    out_label_onehot = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.augmented_label_onehot.mgz")
                    save_framedimage(onehot_labels[n], out_label_onehot, onehotencoded=True)

                    # convert onehot encoding back to label segmentation
                    out_label = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.augmented_label.mgz")
                    label_seg = torch.zeros(onehot_labels[n].shape[1:]).int()
                    for ch in range(onehot_labels[n].shape[0]):
                        label_seg[onehot_labels[n][ch] == 1] = ch
                    save_framedimage(label_seg.unsqueeze(0), out_label)

                    if (priors is not None and priors.numel() != 0):
                        out_priors = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.augmented_priors.mgz")
                        save_framedimage(priors[n], out_priors)

                    out_posteriors = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.posteriors_loss{loss.item():.4f}_dice{np.mean(train_dices[:, :, step]):.4f}.mgz")
                    posteriors = outputs[n]  # non-batched tensor [C, H, W (,D)]
                    np.save(os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.posteriors_loss{loss.item():.4f}_dice{np.mean(train_dices[:, :, step]):.4f}.npy"), posteriors.movedim(0, -1).cpu().detach().numpy())
                    save_framedimage(posteriors, out_posteriors, onehotencoded=True)

                    out_segmentation = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.prediction_loss{loss.item():.4f}_dice{np.mean(train_dices[:, :, step]):.4f}.mgz")
                    predicted_segmentation = torch.argmax(outputs[n], dim=0)
                    np.save(os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.prediction_loss{loss.item():.4f}_dice{np.mean(train_dices[:, :, step]):.4f}.npy"), predicted_segmentation.cpu().int())
                    segmentation = remap_labels(predicted_segmentation, self._inverse_label_mapping)
                    save_framedimage(segmentation.unsqueeze(0), out_segmentation)
            # end of debugging volumes output     

            if (self._summary_writer is not None):
                # Write to TensorBoard every batch
                self._summary_writer.add_scalar("Train/Loss", loss.item(), epoch * steps_per_epoch + batch_idx)
                self._summary_writer.add_scalar(
                    "Train/Dice",
                    np.mean(train_dices[:, :, step]),
                    #torch.mean(torch.tensor(batch_hard_dice)),
                    epoch * steps_per_epoch + batch_idx,
                )

                # --- TensorBoard Visualization (Inside Training Loop) ---
                if batch_idx % 10 == 0:  # Visualize every 10 batches (adjust as needed)
                    slice_index = random.randint(20, 50)  # Choose representative slice index
                    num_examples_to_visualize = min(images.size(0), 6)  # visualize upto 6 examples

                    # Get slices from different examples in the batch
                    image_slices = images[:num_examples_to_visualize, 0, slice_index].cpu()
                    label_slices = onehot_labels[:num_examples_to_visualize, 0, slice_index].cpu()
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
        validation_dices = np.zeros((self._batch_size, self._num_labels, len(self._validation_loader)))

        self._model.eval()        
        with torch.no_grad():
            for batch_idx, (dataset_indices, images, onehot_labels, priors) in enumerate(self._validation_loader):
                images, onehot_labels, priors = images.to(self._device), onehot_labels.to(self._device), priors.to(self._device)

                (outputs, penultimate) = self._model(images, priors)
                if (self._debug):
                    np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_validate_image_to_predict.npy"), images.cpu())
                    if (priors is not None and priors.numel() != 0):
                        np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_validate_prior_to_predict.npy"), priors.cpu())

                    # predicted, labels from 0 .. N
                    predicted = torch.argmax(outputs, dim=1)
                    np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_validate_predicted.npy"), predicted.cpu().int())
                    
                    # map labels to original id
                    predicted_remap = remap_labels(predicted, self._inverse_label_mapping)
                    np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_validate_predicted_remap.npy"), predicted_remap.cpu().int())

                    # posteriors
                    posteriors = outputs  #.squeeze(0)  # remove batch axis => non-batched tensor [C, H, W (,D)]
                    np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_validate_posteriors.npy"), posteriors.cpu().movedim(1, -1))

                    np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_validate_onehot_labels.npy"), onehot_labels.cpu().movedim(1, -1))
                    
                    
                if (metric_type == 'wl2'):
                    loss = loss_fn(penultimate, onehot_labels)
                elif (metric_type == 'dice'):
                    loss = loss_fn(outputs, onehot_labels)

                validation_loss += loss.item()

                # --- Metrics Calculation ---
                # Calculate hard Dice
                batch_hard_dice = self._dice_metric_hard(outputs, onehot_labels)
                validation_dices[:, :, batch_idx] = batch_hard_dice.detach().cpu().numpy()
                logging.info(f"  validation {batch_idx+1:4d}/{len(self._validation_loader):<4d} val loss: {loss.item():.4f}, val dice avg: {np.mean(validation_dices[:, :, batch_idx]):.4f}")

                if (self._summary_writer is not None):
                    # Write validation loss and Dice to TensorBoard (once per epoch)
                    self._summary_writer.add_scalar(
                        "Validation/Loss", validation_loss, epoch * len(self._validation_loader) + batch_idx
                    )
                    self._summary_writer.add_scalar(
                        "Validation/Dice",
                        np.mean(validation_dices[:, :, batch_idx]),
                        #torch.mean(torch.tensor(batch_hard_dice)),
                        epoch * len(self._validation_loader) + batch_idx,
                    )

                    # --- TensorBoard Visualization (Inside Validation Loop) ---
                    if batch_idx % 3 == 0:  # Visualize every 3 batches (adjust as needed)
                        slice_index = random.randint(20, 50)
                        num_examples_to_visualize = min(images.size(0), 6)  # visualize upto 6 examples

                        # Get slices from different examples in the batch
                        image_slices = images[:num_examples_to_visualize, 0, slice_index].cpu()
                        label_slices = onehot_labels[:num_examples_to_visualize, 0, slice_index].cpu()
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
    
