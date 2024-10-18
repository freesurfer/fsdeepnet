import os
import glob
import logging
from time import time
import numpy as np
import torch
import surfa as sf

from checkpoint import Checkpoint
from models.model import UNet
from utils.data_utils import load_volume, save_volume, remap_labels, onehot, centroid
from utils.preprocessing import apply_centercrop

class Prediction:
    """
    This class run predictions of the model provided, and evaluations if ground truth is given

    Attributes
    ----------

    Methods
    -------
    load_model
        Load the trained model

    predict
        Predict with the loaded model
    """
        
    def __init__(self, device=None, ctab=None):
        """
        Prediction Constructor.
        """

        self._model = None
        self._device = device
        if (self._device is None):
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._label_lookup = None
        if (ctab is not None):
            import surfa as sf
            self._label_lookup = sf.load_label_lookup(ctab)


    def load_model(self, model_checkpoint):
        """
        load trained model

        Parameters
        ----------
        model_checkpoint : string
            path of the trained model
        """
        
        assert os.path.isfile(model_checkpoint), "The provided model path %s does not exist." % model_checkpoint

        # Load the Trained Model
        checkpoint = Checkpoint()
        checkpoint.load(model_checkpoint, device=self._device)
        assert checkpoint.model_arch_dict is not None, "Model architecture information not available."
        assert checkpoint.train_dataset_dict is not None, "Training dataset information not available."

        model_creation_string = checkpoint.model_arch_dict["name"] + "(checkpoint.model_arch_dict)"
        self._model = eval(model_creation_string)
        self._model = self._model.to(self._device)

        self._crop_size = checkpoint.train_dataset_dict["crop_size"]
        self._labels_segmentation, self._unique_idx = np.unique(checkpoint.train_dataset_dict["segmentation_labels"], return_index=True)
        self._num_labels = len(self._labels_segmentation)
        
        # compute self._label_mapping, self._inverse_label_mapping from self._labels_segmentation
        self._label_mapping = {label.item(): i for i, label in enumerate(self._labels_segmentation)}
        self._inverse_label_mapping = {v: k for k, v in self._label_mapping.items()}

        if (self._label_lookup is None):
            self._label_lookup = checkpoint.label_lookup
        self._model.load_state_dict(checkpoint.model_state_dict)
        self._model.eval()

        
    def predict(self,
                path_images,
                out_segmentations,
                crop_size=None,
                path_labels=None,
                path_gt=None, # for hard-dice calculation
                path_dice=None,
                addctab=True,
                write_posteriors=None,
                debug=False):
        
        # check inputs
        assert path_images is not None, 'please specify an input file/folder'
        assert out_segmentations is not None, 'please specify an output file/folder'

        if (crop_size is not None):
            self._crop_size = crop_size
        assert self._crop_size is not None, 'please specify cropping size'

        pred_suffix = 'prediction'
        posteriors_suffix = 'posteriors'
            
        # convert path to absolute paths
        path_images = os.path.abspath(path_images)
        if (path_labels is not None):
            path_labels = os.path.abspath(path_labels)
            
        if (os.path.isdir(path_images)):
            if (path_labels is not None):
                assert os.path.isdir(path_labels), 'both %s and %s need to be directory' % (path_images, path_labels)
            if (not os.path.exists(out_segmentations)):
                os.makedirs(out_segmentations)
            assert os.path.isdir(out_segmentations), 'both %s and %s need to be directory' % (path_images, out_segmentations)

            # get all images in the directory
            path_images = sorted(glob.glob(os.path.join(path_images, '*.nii.gz')) +
                                 glob.glob(os.path.join(path_images, '*.nii')) +
                                 glob.glob(os.path.join(path_images, '*.mgz')))

            # get all labels in the directory
            if (path_labels is not None):
                path_labels = sorted(glob.glob(os.path.join(path_labels, '*.nii.gz')) +
                                     glob.glob(os.path.join(path_labels, '*.nii')) +
                                     glob.glob(os.path.join(path_labels, '*.mgz')))            

            # pre-generate all *_predict* filenames
            out_segmentations = [os.path.join(out_segmentations, os.path.basename(p)) for p in path_images]
            out_segmentations = [p.replace('.nii', '_%s.nii' % pred_suffix) for p in out_segmentations]
            out_segmentations = [p.replace('.mgz', '_%s.mgz' % pred_suffix) for p in out_segmentations]
        else:
            # single image
            assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                                'please make sure the path and the extension are correct' % path_images
            if (path_labels is not None):
                assert os.path.isfile(path_labels), 'both %s and %s need to be file \n' % (path_images, path_labels)
                path_labels = [path_labels]
                
            path_images = [path_images]
            out_segmentations = [out_segmentations]

        if (write_posteriors):
            out_posteriors_dir = os.path.join(os.path.dirname(out_segmentations[0]), "posteriors")
            os.makedirs(out_posteriors_dir, exist_ok=True)

        # perform segmentation
        for i in range(len(path_images)):
            ### preprocessing ###
            # reorient to 'RAS'
            sfimage, image_tensor, orig_orientation = load_volume(path_images[i], orientation="RAS", device=self._device)

            label_lookup = self._label_lookup
            if (label_lookup is None):
                label_lookup = sfimage.labels
            crop_idx = None
            image_tensor_cropped = image_tensor
            
            # check if the input image already has crop_size
            # image_tensor returned from load_volume() is non-batched
            image_shape = image_tensor.shape[1:]
            if (np.any(np.array(image_shape) > np.array(self._crop_size))):
                # calculate the cropping center point if label image is available
                center_point = None
                if (path_labels is not None):
                    sflabel, label_tensor, _ = load_volume(path_labels[i], orientation="RAS", device=self._device)
                    center_point = centroid(label_tensor.cpu().squeeze(0).detach().numpy())
                    if (label_lookup is None):
                        label_lookup = sflabel.labels

                # crop the images
                # apply_centercrop() expects input image_tensor to be non-batched, output image_tensor_cropped is non-batched
                (image_tensor_cropped, crop_idx) = apply_centercrop(image_tensor_cropped, self._crop_size, center_point=center_point)
                image_tensor_cropped = image_tensor_cropped.to(self._device).float()

                if (debug):
                    # debugging
                    print("[DEBUG] output cropped image/label ...")
                    if (path_labels is not None):
                        # crop the labels
                        label_tensor_cropped = apply_centercrop(label_tensor, self._crop_size, center_point=center_point)
                        label_tensor_cropped = label_tensor_cropped.to(self._device).float()

                    crop = 'centercropped'
                    if (center_point is not None):
                        crop = 'centroidcropped'
                    out_cropped_image = os.path.join(os.path.dirname(out_segmentations[i]), os.path.splitext(os.path.basename(path_images[i]))[0])+f".image.{crop}.RAS.mgz"
                    save_volume(image_tensor_cropped, sfimage, out_cropped_image)
                    out_cropped_image = os.path.join(os.path.dirname(out_segmentations[i]), os.path.splitext(os.path.basename(path_images[i]))[0])+f".image.{crop}.mgz"
                    save_volume(image_tensor_cropped, sfimage, out_cropped_image, orientation=orig_orientation)                
                    if (path_labels is not None):
                        out_cropped_label = os.path.join(os.path.dirname(out_segmentations[i]), os.path.splitext(os.path.basename(path_labels[i]))[0])+f".label.{crop}.RAS.mgz"
                        save_volume(label_tensor_cropped, sflabel, out_cropped_label)
                        out_cropped_label = os.path.join(os.path.dirname(out_segmentations[i]), os.path.splitext(os.path.basename(path_labels[i]))[0])+f".label.{crop}.mgz"
                        save_volume(label_tensor_cropped, sflabel, out_cropped_label, orientation=orig_orientation)
                    # end of debugging

            # add batch axes
            image_tensor_cropped = image_tensor_cropped.unsqueeze(0)
                
            # normalize
            # ??? todo ???
            
            ### prediction ###
            (outputs, _) = self._model(image_tensor_cropped)
            predicted_segmentation = torch.argmax(outputs, dim=1)
            # map labels to original id
            segmentation_cropped = remap_labels(predicted_segmentation, self._inverse_label_mapping)

            ### postprocessing: align prediction back to original orientation, original image size            
            if (crop_idx is None):
                segmentation = segmentation_cropped.detach().cpu().numpy()
            else:
                # re-position predicted segmentation back to the original image indices where the image was cropped out
                segmentation = np.zeros(shape=image_tensor.shape, dtype='int32')
                segmentation[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = segmentation_cropped.detach().cpu().numpy()
            segmentation = torch.from_numpy(segmentation).to(self._device)
            
            ### save results ###
            save_volume(segmentation, sfimage, out_segmentations[i],
                        orientation=orig_orientation,
                        labels=label_lookup if (addctab) else None)
            print(f"output segmentation {out_segmentations[i]}")
            if (debug):
                print("[DEBUG] output cropped prediction ...")
                seg_noreshape = os.path.join(os.path.dirname(out_segmentations[i]), os.path.splitext(os.path.basename(out_segmentations[i]))[0])+f".cropped.mgz"
                save_volume(segmentation_cropped, sfimage, seg_noreshape,
                            orientation=orig_orientation,
                            labels=label_lookup if (addctab) else None)
            if (write_posteriors):  # ??? question: posteriors need to be re-positioned as well ???
                basename = os.path.basename(out_segmentations[i])
                out_posteriors = basename.replace(f"{pred_suffix}.", f"{posteriors_suffix}.")
                out_posteriors = os.path.join(out_posteriors_dir, out_posteriors)
                posteriors = outputs.squeeze(0)  # remove batch axis => non-batched tensor [C, H, W (,D)]
                #posteriors = movedim(1, -1)  # move channel to last axis
                save_volume(posteriors, sfimage, out_posteriors,
                            orientation=orig_orientation)
                print(f"output posteriors {out_posteriors}")
        # end of segmentation loop

        # evaluate
        if (path_gt is not None):
            # calculate hard-dice between saved segmentations and their ground truth
            from evaluation import Evaluation

            if (path_dice is None):
                path_dice = os.path.join(os.path.dirname(out_segmentations[0]), 'dices.npy')

            print(f"\nEvaluating segmentations ...")
            eval = Evaluation(self._labels_segmentation)                
            if (os.path.isdir(path_gt)):
                eval.evaluate(path_gt, os.path.dirname(out_segmentations[0]), path_dice=path_dice)
            else:
                eval.evaluate(path_gt, out_segmentations[0], path_dice=path_dice)
            
    """
    # this method is not used as of 2024-10-15. it is not in-sync with other changes.
    def evaluate_dataset(self, test_dataset, unique_output_folder,
                         addctab=True, write_posteriors=None, output_gt=None):

        from torch.utils.data import DataLoader
        from utils.metrics import DiceScore            
        dice_metric_hard = DiceScore(
            num_classes=self._num_labels,
            input_type="prob",
            dice_type="hard",
        )

        total_dice_scores = np.zeros(self._num_labels)
        num_samples = 0
        start_time = time()
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # initialize dice_scores (n_labels x n_samples)
        n_labels  = self._num_labels
        n_samples = len(test_loader)
        dice_scores = np.zeros((n_labels, n_samples))

        for idx, (images, labels) in enumerate(test_loader):
            original_image_path = test_dataset.image_files[idx]
            sfimage, _, orig_orientation = load_volume(original_image_path, orientation='RAS', device=self._device)
            base_filename = os.path.splitext(os.path.basename(original_image_path))[0]

            output_segmentation = os.path.join(unique_output_folder, f"{base_filename}_prediction.mgz")
            output_gt = os.path.join(unique_output_folder, f"{base_filename}_gt.mgz")
            output_posteriors = os.path.join(unique_output_folder, f"{base_filename}_posteriors.mgz") if (write_posteriors) else None
            
            images = images.to(self._device).float()
            labels = labels.to(self._device)
            
            (outputs, _) = self._model(images)
        
            predicted_segmentation = torch.argmax(outputs, dim=1)
            # map labels to original id
            segmentation = remap_labels(predicted_segmentation, self._inverse_label_mapping)

            # save segmentation
            save_volume(segmentation, sfimage, output_segmentation,
                        orientation=orig_orientation,
                        labels=self._label_lookup if (addctab) else None)
            save_volume(labels, sfimage, output_gt,
                        orientation=orig_orientation)
                
            if (output_posteriors is not None):
                posteriors = outputs.squeeze(0)  # remove batch axis => non-batched tensor [C, H, W (,D)]
                #posteriors = outputs.movedim(1, -1)
                save_volume(posteriors, sfimage, output_posteriors,
                        orientation=orig_orientation)

            # Remap labels for metric calculation
            remapped_labels = remap_labels(labels, self._label_mapping)
            labels_onehot = onehot(remapped_labels, num_classes=self._num_labels, device=self._device)

            # Calculate metrics
            hard_dice_scores = dice_metric_hard(outputs, labels_onehot)
            dice_scores[:, idx] = hard_dice_scores.detach().cpu().numpy()

            total_dice_scores += dice_scores[:, idx]
            num_samples += 1

            logging.info(f"Sample {idx+1} (Hard Dice):")
            for label_idx in range(self._num_labels):  #enumerate(non_ignored_label_names):
                dice_score = dice_scores[label_idx, idx]
                logging.info(f" Class {self._labels_segmentation[label_idx]}: {dice_score:.4f}")
            
        # output dice_scores (n_labels x n_samples)
        f_dice_scores = os.path.join(unique_output_folder, "dice_scores.npy")
        np.save(f_dice_scores, dice_scores)

        # Calculate average Dice scores for non-ignored classes
        avg_dice_scores = total_dice_scores / num_samples

        logging.info("Average Dice Scores:")
        for label_idx in range(self._num_labels):
            avg_dice_score = avg_dice_scores[label_idx]  #.item()
            logging.info(f"Average Dice Score for Class {self._labels_segmentation[label_idx]}: {avg_dice_score:.4f}")

        # Output summary
        logging.info(f"Total evaluation time: {time() - start_time:.2f} seconds")
    """
            
 
