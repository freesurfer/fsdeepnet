import os
import logging
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from freeseg.config import Config
from freeseg.checkpoint import Checkpoint
from freeseg.metrics import DiceScore
from freeseg.utils import utility as utils


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

    stage_order = {'wl2': 0, 'dice': 1}
        
    def __init__(self,
                 dnn=None,                 # deep neural network
                 train_loader=None,        # torch.utils.data.DataLoader
                 fn_data_generator=None,   # data generator
                 model_arch_dict=None,     # network architecture dictionary
                 train_dataset_dict=None,  # training dataset dictionary
                 train_output_folder=None, # training output directory              
                 validation_loader=None,
                 accuracy_fn=None,
                 ctab=None,     # ascii color table
                 model_checkpoint=None,
                 best_model_metric="dice",                 
                 write_tensorboard_summary=False,
                 report_moving_avg=False,
                 device=None,
                 gpu_index=None,
                 debug=False,
                 **kwargs):
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

        self._report_moving_avg = report_moving_avg
        self._debug = debug
        self._model = dnn
        self._model_arch_dict = model_arch_dict
        self._train_dataset_dict = train_dataset_dict
        self._model_checkpoint = model_checkpoint
        self._ctab = ctab
        self._validation_loader = validation_loader
        self._best_model_metric = best_model_metric
        self._batch_size = train_dataset_dict["batch_size"]

        self._setup_training_directory(train_output_folder)

        labels_segmentation = train_dataset_dict["segmentation_labels"]
        self._num_labels = model_arch_dict["nb_labels"]
        self._label_mapping = train_dataset_dict["label_mapping"]
        self._inverse_label_mapping = train_dataset_dict["inverse_label_mapping"]

        self._device = device
        self._gpu_index = gpu_index
        if (self._device is None):
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (self._gpu_index is None and torch.cuda.is_available()):
            self._gpu_index = torch.cuda.current_device()

        self._data_generator = fn_data_generator

        self._summary_writer = None
        if (write_tensorboard_summary):
            # Create TensorBoard writer
            from torch.utils.tensorboard import SummaryWriter
            self._summary_writer = SummaryWriter(train_output_folder)

        self._dice_metric_hard = accuracy_fn
        if (self._validation_loader is not None and self._best_model_metric == "dice"):
            assert (self._dice_metric_hard is not None), "Need 'model_metrics_accuracy' to pick best 'dice' model"

        # surfa.core.labels.LabelLookup
        self._label_lookup = train_dataset_dict.pop("label_lookup", None)
        if (self._ctab is not None):
            import surfa as sf
            self._label_lookup = sf.load_label_lookup(self._ctab)

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


    def train_model(self, lr=0.0001, epochs=100, steps_per_epoch=1000, metric_type=None, optimizer_cls=None, loss_fn=None):
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

            # continue training of the same metric_type, or restart from next stage 
            if (self._checkpoint.metric_type is not None and \
                Training.stage_order[metric_type] < Training.stage_order[self._checkpoint.metric_type]):
                return

            if (self._checkpoint.metric_type == metric_type):
                start_epoch = self._checkpoint.epoch + 1
            else:
                start_epoch = 0
            end_epoch = start_epoch + epochs
            logging.info(f"Continue to train {end_epoch-start_epoch} ({epochs}) {metric_type} epochs, lr:{lr}")
            if (self._label_lookup is None):
                self._label_lookup = self._checkpoint.label_lookup
                # update label_lookup with the one saved in pre-trained model
                self._checkpoint.update({"label_lookup" : self._label_lookup})
            self._model_checkpoint = None  # the checkpoint will only be used once in the training

        # save last epoch number
        self._end_epoch = end_epoch
        
        # training loop
        ncols = 2 if (self._validation_loader is None) else 4
        loss_dice_avg = np.zeros((end_epoch-start_epoch, ncols))        
        for epoch in range(start_epoch, end_epoch):
            if (self._gpu_index is not None):
                utils.gpu_report(self._gpu_index)
            logging.info(f"Epoch {epoch+1:>3d}/{end_epoch:<3d}")
            (train_loss, train_dices)  = self._train_one_epoch(optimizer, loss_fn, epoch, steps_per_epoch,
                                                               metric_type=metric_type)
            
            train_loss /= steps_per_epoch
            train_dice_avg = 0.0            
            if (self._dice_metric_hard is not None):
                train_dice_avg = np.mean(train_dices)
        
                # output training dices batch_size x (n_labels x steps_per_epoch)
                f_dice_scores = os.path.join(self._dice_dir, f"train_{metric_type}_{epoch+1:03d}.npy")
                np.save(f_dice_scores, train_dices)
                # Save in text format as (steps_per_epoch x n_labels)
                if (self._batch_size == 1):
                    """
                    Known Issue: np.savetxt() is designed for saving 1D and 2D arrays to text files.
                                 It does not directly support saving 3D arrays.
                                 Attempting to use savetxt() on a 3D array will result in a ValueError
                    """
                    f_dice_dat = os.path.join(self._dice_dir, f"d.train_{metric_type}_{epoch+1:03d}.dat")            
                    np.savetxt(f_dice_dat, np.transpose(np.squeeze(train_dices)))
    
            if (self._validation_loader is None):
                loss_dice_avg[epoch-start_epoch] = np.array((train_loss, train_dice_avg))
                if (self._dice_metric_hard is not None):
                    info = f"Epoch [{epoch+1:>3d}/{end_epoch:<3d}], Train Loss: {train_loss:.4f}, Train Dice Avg: {train_dice_avg:.4f}"
                else:
                    info = f"Epoch [{epoch+1:>3d}/{end_epoch:<3d}], Train Loss: {train_loss:.4f}"
                logging.info(info)

                # model dict
                checkpoint_dict = {
                    "epoch": epoch,
                    "metric_type": metric_type,
                    "model_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "dice": train_dice_avg if (self._dice_metric_hard is not None) else None,
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
                validation_dice_avg = 0.0
                if (self._dice_metric_hard is not None):
                    validation_dice_avg = np.mean(validation_dices)
        
                    # output validation dices (n_labels x len(self._validation_loader))
                    f_dice_scores = os.path.join(self._dice_dir, f"validation_{metric_type}_{epoch+1:03d}.npy")
                    np.save(f_dice_scores, validation_dices)
                    if (self._validation_loader.batch_size == 1):
                        # Save in text format as nsubjects x nlabels
                        f_dice_dat = os.path.join(self._dice_dir, f"d.validation_{metric_type}_{epoch+1:03d}.dat")
                        np.savetxt(f_dice_dat, np.transpose(np.squeeze(validation_dices)))

                loss_dice_avg[epoch-start_epoch] = np.array((train_loss, train_dice_avg, validation_loss, validation_dice_avg))
                if (self._dice_metric_hard is not None):
                    info = f"Epoch [{epoch+1:>3d}/{end_epoch:<3d}], Train Loss: {train_loss:.4f}, Train Dice Avg: {train_dice_avg:.4f}, Val Loss: {validation_loss:.4f}, Val Dice Avg: {validation_dice_avg:.4f}"
                else:
                    info = f"Epoch [{epoch+1:>3d}/{end_epoch:<3d}], Train Loss: {train_loss:.4f}, Val Loss: {validation_loss:.4f}"
                logging.info(info)

                # model dict
                checkpoint_dict = {
                    "epoch": epoch,
                    "metric_type": metric_type,
                    "model_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": validation_loss,
                    "dice": validation_dice_avg if (self._dice_metric_hard is not None) else None,
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
        # Save in text format as nepoch x ncols: 'train_loss train_dice_avg validation_loss validation_dice_avg'
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
        train_dice_hard = 0.0
        train_dices = None
        if (self._dice_metric_hard is not None):
            train_dices = np.zeros((self._batch_size, self._num_labels, steps_per_epoch))

        self._model.train()        
        for step in range(steps_per_epoch):
            batched_sample = next(self._data_generator)
            batch_idx = batched_sample.pop(0)         # remove first item batch_idx
            dataset_indices = batched_sample.pop(-1)  # remove last item dataset_indices
            haspriors = True if (len(batched_sample) == 3) else False
            if (haspriors):
                images, onehot_labels, priors = batched_sample
            else:
                images, onehot_labels = batched_sample
                priors = None

            # training device and preprocessing device could be different
            images, onehot_labels = images.to(self._device).float(), onehot_labels.to(self._device).int()
            if (priors is not None):
                 priors = priors.to(self._device).float()
            
            # Zero your gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            (outputs, penultimate) = self._model(images, priors=priors)

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
            if (self._dice_metric_hard is not None):
                # Calculate hard Dice
                batch_hard_dice = self._dice_metric_hard(outputs, onehot_labels)
                train_dices[:, :, step] = batch_hard_dice.detach().cpu().numpy()
                train_dices_avg = np.mean(train_dices[:, :, step])
                train_dice_hard += train_dices_avg

            # report simple moving loss and dice average or loss/dice for each step
            batch_indices = ", ".join(str(item).zfill(4) for item in dataset_indices.tolist())
            if (self._report_moving_avg):
                if (self._dice_metric_hard is not None):
                    info = f"  {step+1:>4d}/{steps_per_epoch:<4d} ({batch_indices}) loss: {train_loss/(step+1):.4f}, dice avg: {train_dice_hard/(step+1):.4f}"
                else:
                    info = f"  {step+1:>4d}/{steps_per_epoch:<4d} ({batch_indices}) loss: {train_loss/(step+1):.4f}"
            else:
                if (self._dice_metric_hard is not None):
                    info = f"  {step+1:>4d}/{steps_per_epoch:<4d} ({batch_indices}) loss: {loss.item():.4f}, dice avg: {train_dices_avg:.4f}"
                else:
                    info = f"  {step+1:>4d}/{steps_per_epoch:<4d} ({batch_indices}) loss: {loss.item():.4f}"
            logging.info(info)

            # begin of debugging volumes output            
            if (self._debug and ((step == 0 and epoch > 0) or (epoch == self._end_epoch-1 and step == steps_per_epoch-1))):
                # save debug volumes for a) first step of all epoch except first epoch; b) last step of last epoch  
                # output augmented images/labels/priors, onehot encoded labels, posteriors, prediciton from each batch (batch_size x [C, H, W(, D)])
                # both augmented label and prediction are saved as npy and mgz
                # the npy files contain label ids from (0 .. N), the mgz files contain the real segmentation labels
                logging.debug(f"output augmented images/labels, onehot encoded labels, posteriors, prediciton ...")                
                for n, idx in enumerate(dataset_indices):
                    # augmented image
                    out_image = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.augmented_image.mgz")
                    utils.save_framedimage(images[n], out_image)

                    # augmented label onehot
                    out_label_onehot = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.augmented_label_onehot.mgz")
                    utils.save_framedimage(onehot_labels[n], out_label_onehot, onehotencoded=True)

                    # convert onehot encoding back to label segmentation
                    out_label = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.augmented_label.mgz")
                    label_seg = torch.zeros(onehot_labels[n].shape[1:]).int()  # [H, W(, D)]
                    for ch in range(onehot_labels[n].shape[0]):
                        label_seg[onehot_labels[n][ch] == 1] = ch
                    label_seg = utils.remap_labels(label_seg, self._inverse_label_mapping)
                    utils.save_framedimage(label_seg.unsqueeze(0), out_label, labels=self._label_lookup)

                    # augmented prior
                    if (priors is not None and priors.numel() != 0):
                        out_priors = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.augmented_priors.mgz")
                        utils.save_framedimage(priors[n], out_priors)

                    # posteriors
                    out_posteriors = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.posteriors_loss{loss.item():.4f}.mgz")
                    posteriors = outputs[n]  # non-batched tensor [C, H, W (,D)]
                    utils.save_framedimage(posteriors, out_posteriors, onehotencoded=True)

                    # prediction
                    out_segmentation = os.path.join(self._debug_dir, f"{metric_type}_{epoch+1:03d}_{idx:03d}.prediction_loss{loss.item():.4f}.mgz")
                    predicted_segmentation = torch.argmax(outputs[n], dim=0)
                    segmentation = utils.remap_labels(predicted_segmentation, self._inverse_label_mapping)
                    utils.save_framedimage(segmentation.unsqueeze(0), out_segmentation, labels=self._label_lookup)
            # end of debugging volumes output     

            # begin of tensorboard summary writer
            if (self._summary_writer is not None):
                # Write to TensorBoard every batch
                self._summary_writer.add_scalar("Train/Loss", loss.item(), epoch * steps_per_epoch + batch_idx)
                if (self._dice_metric_hard is not None):
                    self._summary_writer.add_scalar(
                        "Train/Dice",
                        train_dices_avg,
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
        validation_dices = None
        if (self._dice_metric_hard is not None):
            validation_dices = np.zeros((self._validation_loader.batch_size, self._num_labels, len(self._validation_loader)))

        self._model.eval()
        with torch.no_grad():
            for batch_idx, batched_sample in enumerate(self._validation_loader):
                dataset_indices = batched_sample.pop(0)
                haspriors = True if (len(batched_sample) == 3) else False
                if (haspriors):
                    images, onehot_labels, priors = batched_sample
                else:
                    images, onehot_labels = batched_sample
                    priors = None

                images, onehot_labels = images.to(self._device), onehot_labels.to(self._device)
                if (priors is not None):
                    priors = priors.to(self._device)

                (outputs, penultimate) = self._model(images, priors=priors)
                if (self._debug):
                    logging.debug(f"output validation debug volumes ...")
                    for n, idx in enumerate(dataset_indices):
                        # image
                        np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_{idx:03d}_validate_image_to_predict.npy"), images[n].cpu().movedim(0, -1).detach().numpy())
                        # prior
                        if (priors is not None and priors.numel() != 0):
                            np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_{idx:03d}_validate_prior_to_predict.npy"), priors[n].cpu().movedim(0, -1).detach().numpy())
                        # predicted, labels from 0 .. N
                        predicted = torch.argmax(outputs[0], dim=0)  # [C, H, W (,D)] => [H, W(, D)]
                        np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_{idx:03d}_validate_predicted.npy"), predicted.unsqueeze(0).cpu().movedim(0, -1).int())
                        # posteriors
                        posteriors = outputs[0]    # remove batch axis => non-batched tensor [C, H, W (,D)]
                        onehot = onehot_labels[0]
                        if ((posteriors.ndim - 1) == 2):
                            # for 2D, make posteriors and onehot into shape [C, H, W, 1]
                            posteriors = posteriors.unsqueeze(-1)
                            onehot = onehot.unsqueeze(-1)
                        np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_{idx:03d}_validate_posteriors.npy"), posteriors.cpu().movedim(0, -1))
                        # onehot label
                        np.save(os.path.join(self._debug_dir, f"{batch_idx:03d}_{idx:03d}_validate_onehot_labels.npy"), onehot.cpu().movedim(0, -1))
                    
                if (metric_type == 'wl2'):
                    loss = loss_fn(penultimate, onehot_labels)
                elif (metric_type == 'dice'):
                    loss = loss_fn(outputs, onehot_labels)

                validation_loss += loss.item()

                # --- Metrics Calculation ---
                batch_indices = ", ".join(str(item).zfill(4) for item in dataset_indices.tolist())
                if (self._dice_metric_hard is not None):
                    # Calculate hard Dice
                    batch_hard_dice = self._dice_metric_hard(outputs, onehot_labels)
                    validation_dices[:, :, batch_idx] = batch_hard_dice.detach().cpu().numpy()
                    logging.info(f"  validation {batch_idx+1:4d}/{len(self._validation_loader):<4d} ({batch_indices}) val loss: {loss.item():.4f}, val dice avg: {np.mean(validation_dices[:, :, batch_idx]):.4f}")
                else:
                    logging.info(f"  validation {batch_idx+1:4d}/{len(self._validation_loader):<4d} ({batch_indices}) val loss: {loss.item():.4f}")

                # begin of tensorboard summary writer
                if (self._summary_writer is not None):
                    # Write validation loss and Dice to TensorBoard (once per epoch)
                    self._summary_writer.add_scalar(
                        "Validation/Loss", validation_loss, epoch * len(self._validation_loader) + batch_idx
                    )
                    if (self._dice_metric_hard is not None):
                        self._summary_writer.add_scalar(
                            "Validation/Dice",
                            np.mean(validation_dices[:, :, batch_idx]),
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
    

    @staticmethod
    def setup(config, preload_dataset=False, create_train_dataset=True, create_loader=True, create_val_loader=True, create_model=True):
        """
        1. create training DataLoader, validation DataLoader, model, and optimizer
        2. update config

        returns config, train_loader, validation_loader, model, optimizer, train_dataset
        """

        # create a torch.utils.data.Dataset object
        def load_dataset(
                py_dataset_cls,
                dataset_profile,
                dataaugment,
                device=None,
                keep_trainset_in_memory=False,
                cohort=[],
                preload=False,
                augdir=None):
            if (device is None):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            dataset = py_dataset_cls(
                    dataaugment,
                    device=device,                    
                    cohort=cohort,
                    keep_trainset_in_memory=keep_trainset_in_memory,
                    preload=preload,
                    augdir=augdir,
                    **dataset_profile)

            return dataset


        # create augmentation object
        def create_augment_object(transforms, config, cohort='train', device=None):
            cfg_dataset = config["dataset"].copy()
            cfg_dataset.pop("class_name", None)  # remove 'class_name'
            
            if (cohort == "validation"):
                cfg_preprocess = config["evaluation"].copy()
                transfer_keys = ["augmentation_wrapper", "crop_size", "verbose", "augmentation_dir"]
                for key in transfer_keys:
                    if (key not in cfg_preprocess):
                        cfg_preprocess[key] = config["preprocessing"].get(key)
                cfg_dataset["batch_size"] = cfg_preprocess.pop("batch_size", 1)
            else:  # cohort='train'
                cfg_preprocess = config["preprocessing"].copy()                

            # retrieve and remove 'augmentation_wrapper'
            augment_classname = cfg_preprocess.pop("augmentation_wrapper", "freeseg.augmentation.augmentbase.AugmentBase")
            if ("Augment2" in augment_classname):
                logging.info("'augment2.Augment2' is specified in config.")
                logging.info("Change 'augment2.Augment2' to 'augmentbase.AugmentBase' since augmentations in augment2.Augment2 are now implemented in augmentbase.AugmentBase")
                augment_classname = "freeseg.augmentation.augmentbase.AugmentBase"
            assert (augment_classname is not None), "Must provide an data augmentation class"
            
            if (device is None):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # create data augment object                    
            py_augment_cls = utils.get_class(augment_classname)
            augment_obj = py_augment_cls(Config.list2dict(cfg_preprocess.pop("augmentations")),    # retrieve and remove 'augmentations'
                                         transforms,
                                         device=device,
                                         **cfg_preprocess,  # '**' operator unpacks 'preprocessing' key/value pairs to keyword arguments
                                         **cfg_dataset)     # '**' operator unpacks 'dataset' key/value pairs to keyword arguments

            return augment_obj


        ### retrieve dataset class
        dataset_classname = config["dataset"].get("class_name", "freeseg.datasets.segmentationdataset.SegmentationDataset")
        py_dataset_cls = utils.get_class(dataset_classname)                    

        ### retrieve dataset class static method process_dataset_attr(), process and update dataset attributes
        process_dataset_attr = getattr(py_dataset_cls, "process_dataset_attr", None)
        if (process_dataset_attr is None):
            logging.warning(f"Method 'process_dataset_attr' not found in {py_dataset_cls}. Skipped dataset attributes processing. The training will fail if there are dependency on processed attributes.")
        else:
            config["dataset"] = process_dataset_attr(config["dataset"], config["output_folder"])

        ### create training Dataset
        train_dataset = None
        if (create_train_dataset):
            train_augmentations = utils.remove_duplicates(Config.get_augmentations(config["preprocessing"].get("augmentations")))
            train_augment_obj = create_augment_object(train_augmentations, config, cohort="train", device=config["preprocessing_device"])
            train_dataset = load_dataset(py_dataset_cls, config["dataset"], train_augment_obj,
                                         device=config["preprocessing_device"],
                                         keep_trainset_in_memory=config["keep_trainset_in_memory"],
                                         cohort=config["train_cohort"], preload=preload_dataset, augdir=config["preprocessing"].get("augmentation_dir", None))
            # UPDATE config
            config.update({"train_augmentations": train_augmentations})
            # update config["dataset"]
            config["dataset"].update(train_dataset.profile)

        ### create training DataLoader
        train_loader = None        
        if (create_train_dataset and create_loader):
            train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                                      pin_memory=config["dataloader"]["pin_memory"], num_workers=config["dataloader"]["num_workers"],
                                      persistent_workers=config["dataloader"]["persistent_workers"], prefetch_factor=config["dataloader"]["prefetch_factor"])

        ### create validation DataLoader
        validation_loader = None
        perform_evaluation = False
        if (config.get("training", None)):
            perform_evaluation = config["training"].get("perform_evaluation", False)
        if (create_loader and create_val_loader and perform_evaluation):
            # enforce "centercrop"/"rescalevolume" for evaluation_augmentations
            val_augmentations = ["centercrop", "rescalevolume"]
            config["evaluation"]["augmentations"] = val_augmentations
            val_augment_obj = create_augment_object(val_augmentations, config, cohort="validation", device=config["preprocessing_device"])
            # to keep validation_dataset in memory,
            # validation_dataset.preload() needs to be called
            validation_dataset = load_dataset(py_dataset_cls, config["dataset"], val_augment_obj,
                                              device=config["preprocessing_device"],
                                              cohort=config["validation_cohort"])
            if (validation_dataset is None):
                logging.warning(f"No 'validation' set in {config['dataset']['dataset_list_file']} to perform evaluation")
                config["training"]["perform_evaluation"] = False
            else:
                validation_loader = DataLoader(validation_dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False)

        ### output segmentation_labels.npy
        train_dataset_dict = config["dataset"]
        if (config["output_folder"] is not None and preload_dataset):
            train_dataset_dict = train_dataset.profile
            generation_labels = train_dataset_dict["reported_generation_labels"]
            f_generation_labels = os.path.join(config["output_folder"], "reported_generation_labels.npy")
            np.save(f_generation_labels, np.array(sorted(generation_labels)).astype(int))
            # UPDATE config.dataset
            config["dataset"].update(train_dataset_dict)

        #### update model architecture dict
        """
        - `model_arch_dict` will be passed to create individual network object
        - the model configurable keywords need to match individual network implementations
        - it is the individual network implementation's responsibility to check their availabilities
        - `num_channels` and `nb_labels` are not required, they are set here for freeseg.models.unet.UNet
            to dataset configurables `expected_num_channels` and `len(segmentation_labels)` respectively
            if they are missing from model configurables.
        """
        model_arch_dict = config["model"]
        if ("num_channels" not in model_arch_dict and "expected_num_channels" in config["dataset"]):
            model_arch_dict["num_channels"] = config["dataset"]["expected_num_channels"]
        if ("nb_labels" not in model_arch_dict and "num_labels" in config["dataset"]):
            model_arch_dict["nb_labels"] = config["dataset"]["num_labels"]

        #### create the model to train
        model, optimizer_cls = None, None
        if (create_model):
            the_model_name = model_arch_dict.get("name", None)
            assert the_model_name is not None, "Model name is not available."

            model_class = utils.get_class(the_model_name)
            model = model_class(model_arch_dict).to(config["device"])
                
            ### retrieve optimizer class
            optimizer=config["training"].get("optimizer", "torch.optim.Adam")
            optimizer_cls = utils.get_class(optimizer)
        
        ### set_deterministic_training if requested
        deterministic = False
        if (config.get("training", None)):
            deterministic = config["training"].get("deterministic", False)
        if (deterministic):
            # ??? todo: for multi-process dataloader, use worker_init_fn() and generator to preserve reproducibility
            #           see https://pytorch.org/docs/stable/notes/randomness.html
            utils.set_deterministic_training()

        return config, train_loader, validation_loader, model, optimizer_cls, train_dataset
