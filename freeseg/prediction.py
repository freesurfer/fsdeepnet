import os
import glob
from time import time
import numpy as np
import torch
import surfa as sf

from freeseg.checkpoint import Checkpoint
from freeseg.models import UNet
from freeseg.utils import load_framedimage, save_framedimage, remap_labels, onehot, centroid
from freeseg.augmentation import apply_centercrop

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

        the_model_name = checkpoint.model_arch_dict.get("name", None)
        assert the_model_name is not None, "Model name is not available."

        model_creation_string = the_model_name + "(checkpoint.model_arch_dict)"
        self._model = eval(model_creation_string)
        self._model = self._model.to(self._device)

        self._nb_levels = checkpoint.model_arch_dict["nb_levels"]
        self._ndims = checkpoint.model_arch_dict["ndims"]
        assert (self._ndims == 3 or self._ndims == 2), "Model supports 3D or 2D"
        
        self._crop_size = checkpoint.train_dataset_dict["crop_size"]
        self._labels_segmentation, self._unique_idx = np.unique(checkpoint.train_dataset_dict["segmentation_labels"], return_index=True)
        self._num_labels = len(self._labels_segmentation)
        
        # retrieve self._label_mapping, self._inverse_label_mapping from checkpoint.train_dataset_dict
        # or compute them from self._labels_segmentation for backward compatibility
        self._label_mapping = checkpoint.train_dataset_dict.get("label_mapping", None)
        self._inverse_label_mapping = checkpoint.train_dataset_dict.get("inverse_label_mapping", None)
        if (self._label_mapping is None):
            print(f"compute label_mapping ...")
            self._label_mapping = {label:i for i, label in enumerate(self._labels_segmentation)}
        if (self._inverse_label_mapping is None):
            print(f"compute inverse_label_mapping ...")
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
                path_priors=None,
                path_gt=None, # for hard-dice calculation
                addctab=True,
                write_posteriors=False,
                debug=False):
        
        # check inputs
        assert path_images is not None, 'please specify an input file/folder'
        assert out_segmentations is not None, 'please specify an output file/folder'

        if (crop_size is not None):
            self._crop_size = crop_size
        assert self._crop_size is not None, 'please specify cropping size'
        assert (np.all(np.array(self._crop_size) % (2**(self._nb_levels-1)) == 0)), \
            f"crop_size {self._crop_size} needs to be divisible by 2^{self._nb_levels-1}"

        pred_suffix = 'prediction'
        posteriors_suffix = 'posteriors'
            
        # convert path to absolute paths
        if (not isinstance(path_images, list)):
            path_images = os.path.abspath(path_images)
            if (path_labels is not None):
                path_labels = os.path.abspath(path_labels)
            if (path_priors is not None):
                path_priors = os.path.abspath(path_priors)
            
        if (isinstance(path_images, list)):
            # list of images
            if (not os.path.exists(out_segmentations)):
                os.makedirs(out_segmentations)
            assert os.path.isdir(out_segmentations), '%s need to be a directory to segment list of images' % (out_segmentations)

            if (path_labels is not None):
                assert os.path.isdir(path_labels), '%s need to be a directory\n' % (path_images, path_labels)
                path_labels = sorted(glob.glob(os.path.join(path_labels, '*.nii.gz')) +
                                     glob.glob(os.path.join(path_labels, '*.nii')) +
                                     glob.glob(os.path.join(path_labels, '*.mgz')))
                
            # pre-generate all *_predict* filenames
            out_segmentations = [os.path.join(out_segmentations, os.path.basename(p)) for p in path_images]
            out_segmentations = [p.replace('.nii', '.%s.nii' % pred_suffix) for p in out_segmentations]
            out_segmentations = [p.replace('.mgz', '.%s.mgz' % pred_suffix) for p in out_segmentations]
        elif (os.path.isdir(path_images)):
            if (path_labels is not None):
                assert os.path.isdir(path_labels), 'both %s and %s need to be directory' % (path_images, path_labels)
            if (path_priors is not None):
                assert os.path.isdir(path_priors), 'both %s and %s need to be directory' % (path_images, path_priors)
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

            # get all priors in the directory
            if (path_priors is not None):
                path_priors = sorted(glob.glob(os.path.join(path_priors, '*.nii.gz')) +
                                     glob.glob(os.path.join(path_priors, '*.nii')) +
                                     glob.glob(os.path.join(path_priors, '*.mgz')))

            # pre-generate all *_predict* filenames
            out_segmentations = [os.path.join(out_segmentations, os.path.basename(p)) for p in path_images]
            out_segmentations = [p.replace('.nii', '.%s.nii' % pred_suffix) for p in out_segmentations]
            out_segmentations = [p.replace('.mgz', '.%s.mgz' % pred_suffix) for p in out_segmentations]
        else:
            # single image
            assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                                'please make sure the path and the extension are correct' % path_images
            assert (not os.path.isdir(out_segmentations)), 'both %s and %s need to be files' % (path_images, out_segmentations)
                
            if (path_labels is not None):
                assert os.path.isfile(path_labels), 'both %s and %s need to be file \n' % (path_images, path_labels)
                path_labels = [path_labels]
            if (path_priors is not None):
                assert os.path.isfile(path_priors), 'both %s and %s need to be file \n' % (path_images, path_priors)
                path_priors = [path_priors]
                
            path_images = [path_images]
            out_segmentations = [out_segmentations]

        # check path_images, path_labels, path_priors, and out_segmentations have the same length
        assert (len(path_images) == len(out_segmentations)), "input images and output segmentations need to be the same length"
        if (path_labels is not None):
            assert (len(path_images) == len(path_labels)), "images and labels need to be the same length"        
        if (path_priors is not None):
            assert (len(path_images) == len(path_priors)), "images and priors need to be the same length"
        
        if (write_posteriors):
            out_posteriors_dir = os.path.join(os.path.dirname(out_segmentations[0]), "posteriors")
            os.makedirs(out_posteriors_dir, exist_ok=True)
        if (debug):
            out_debug_dir = os.path.join(os.path.dirname(out_segmentations[0]), "debug")
            os.makedirs(out_debug_dir, exist_ok=True)

        list_predictions = None
        if (path_gt is not None):
            list_predictions = list()  # make an empty list

        # perform segmentation
        for i in range(len(path_images)):
            ### preprocessing ###
            # reorient to 'RAS'
            sfimage, image_tensor, orig_orientation = load_framedimage(path_images[i], orientation="RAS", device=self._device, ndims=self._ndims)
            if (path_priors is not None):
                sfprior, prior_tensor, orig_ori_prior = load_framedimage(path_priors[i], orientation="RAS", device=self._device, ndims=self._ndims)
                assert (list(prior_tensor.shape) == [self._num_labels, *image_tensor.shape[1:]]), \
                    f"Expected prior shape [self.num_classes, *image_tensor.shape[1:]], but got {list(prior_tensor.shape)}"
                
            if (list_predictions is not None):
                list_predictions.append(os.path.splitext(os.path.basename(path_images[i]))[0])  # strip file extension
            if (debug):
                print("[DEBUG] output re-oriented image/prior ...")
                out_reoriented_image = os.path.join(out_debug_dir, os.path.splitext(os.path.basename(path_images[i]))[0])+f".image.reoriented.RAS.mgz"
                save_framedimage(image_tensor, out_reoriented_image, original_framedimage=sfimage)
                if (path_priors is not None):
                    out_reoriented_prior = os.path.join(out_debug_dir, os.path.splitext(os.path.basename(path_priors[i]))[0])+f".prior.reoriented.RAS.mgz"
                    save_framedimage(prior_tensor, out_reoriented_prior, original_framedimage=sfprior)
                
            label_lookup = self._label_lookup
            if (label_lookup is None):
                label_lookup = sfimage.labels
            crop_idx = None
            label_tensor = None
            image_tensor_cropped = image_tensor
            prior_tensor_cropped = prior_tensor if (path_priors is not None) else None
            
            # check if the input image already has crop_size
            # image_tensor returned from load_framedimage() is non-batched
            image_shape = image_tensor.shape[1:]
            if (np.any(np.array(image_shape) > np.array(self._crop_size))):
                # calculate the cropping center point if label image is available
                if (path_labels is not None):
                    sflabel, label_tensor, _ = load_framedimage(path_labels[i], orientation="RAS", device=self._device, ndims=self._ndims)
                    assert (label_tensor.shape == image_tensor.shape), \
                        f"image and label need to be in the same shape. label {path_labels[i]} has shape {label_tensor.shape}, image {path_images[i]} has shape {image_tensor.shape}"

                    if (label_lookup is None):
                        label_lookup = sflabel.labels

                # crop the images
                # apply_centercrop() expects input image_tensor to be non-batched, output image_tensor_cropped is non-batched
                (image_tensor_cropped, label_tensor_cropped, prior_tensor_cropped, crop_idx) = \
                    apply_centercrop(image_tensor_cropped, self._crop_size, label=label_tensor, prior=prior_tensor_cropped)
                image_tensor_cropped = image_tensor_cropped.to(self._device).float()

                if (debug):
                    # begin of debugging
                    crop = 'centercropped'
                    if (path_labels is not None):
                        crop = 'centroidcropped'

                    print(f"[DEBUG] output {crop} image/label ...")
                    out_cropped_image = os.path.join(out_debug_dir, os.path.splitext(os.path.basename(path_images[i]))[0])+f".image.{crop}.RAS.mgz"
                    save_framedimage(image_tensor_cropped, out_cropped_image, original_framedimage=sfimage)
                    out_cropped_image = os.path.join(out_debug_dir, os.path.splitext(os.path.basename(path_images[i]))[0])+f".image.{crop}.mgz"
                    save_framedimage(image_tensor_cropped, out_cropped_image, original_framedimage=sfimage, orientation=orig_orientation)
                    if (prior_tensor_cropped is not None):
                        out_cropped_prior = os.path.join(out_debug_dir, os.path.splitext(os.path.basename(path_priors[i]))[0])+f".prior.{crop}.RAS.mgz"
                        save_framedimage(prior_tensor_cropped, out_cropped_prior, original_framedimage=sfprior, dtype=float)
                        out_cropped_prior = os.path.join(out_debug_dir, os.path.splitext(os.path.basename(path_priors[i]))[0])+f".prior.{crop}.mgz"
                        save_framedimage(prior_tensor_cropped, out_cropped_prior, original_framedimage=sfprior, orientation=orig_orientation, dtype=float)
                    if (path_labels is not None):
                        out_cropped_label = os.path.join(out_debug_dir, os.path.splitext(os.path.basename(path_labels[i]))[0])+f".label.{crop}.RAS.mgz"
                        save_framedimage(label_tensor_cropped, out_cropped_label, original_framedimage=sflabel)
                        out_cropped_label = os.path.join(out_debug_dir, os.path.splitext(os.path.basename(path_labels[i]))[0])+f".label.{crop}.mgz"
                        save_framedimage(label_tensor_cropped, out_cropped_label, original_framedimage=sflabel, orientation=orig_orientation)
                    # end of debugging

            # add batch axes
            image_tensor_cropped = image_tensor_cropped.unsqueeze(0)
            if (prior_tensor_cropped is not None):
                prior_tensor_cropped = prior_tensor_cropped.unsqueeze(0)
                
            # normalize
            # ??? todo ???
            
            ### prediction ###
            (outputs, _) = self._model(image_tensor_cropped, prior_tensor_cropped)
            predicted_segmentation = torch.argmax(outputs, dim=1)
            # map labels to original id
            segmentation_cropped = remap_labels(predicted_segmentation, self._inverse_label_mapping)

            ### postprocessing: align prediction back to original orientation, original image size            
            if (crop_idx is None):
                segmentation = segmentation_cropped.detach().cpu().numpy()
            else:
                # re-position predicted segmentation back to the original image indices where the image was cropped out
                segmentation = np.zeros(shape=(segmentation_cropped.shape[0], *image_shape), dtype='int32')
                if (self._ndims == 3):
                    segmentation[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = segmentation_cropped.detach().cpu().numpy()
                else:
                    segmentation[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] = segmentation_cropped.detach().cpu().numpy()
            segmentation = torch.from_numpy(segmentation).to(self._device)
            
            ### save results ###
            save_framedimage(segmentation, out_segmentations[i],
                        original_framedimage=sfimage,
                        orientation=orig_orientation,
                        labels=label_lookup if (addctab) else None)
            print(f"output segmentation {out_segmentations[i]}")
            if (debug):
                print("[DEBUG] output cropped prediction ...")
                seg_noreshape = os.path.join(out_debug_dir, os.path.splitext(os.path.basename(out_segmentations[i]))[0])+f".cropped.mgz"
                save_framedimage(segmentation_cropped, seg_noreshape,
                            original_framedimage=sfimage, 
                            orientation=orig_orientation,
                            labels=label_lookup if (addctab) else None)
            if (write_posteriors):  # ??? question: posteriors need to be re-positioned as well ???
                basename = os.path.basename(out_segmentations[i])
                out_posteriors = basename.replace(f"{pred_suffix}.", f"{posteriors_suffix}.")
                out_posteriors = os.path.join(out_posteriors_dir, out_posteriors)
                posteriors = outputs.squeeze(0)  # remove batch axis => non-batched tensor [C, H, W (,D)]
                #posteriors = movedim(1, -1)  # move channel to last axis
                save_framedimage(posteriors, out_posteriors, original_framedimage=sfimage, 
                                 orientation=orig_orientation, onehotencoded=True, dtype=float)
                print(f"output posteriors {out_posteriors}")
        # end of segmentation loop

        # evaluate
        if (path_gt is not None):
            # calculate hard-dice between saved segmentations and their ground truth
            from freeseg.evaluation import Evaluation

            path_dice = os.path.join(os.path.dirname(out_segmentations[0]), 'dices.npy')

            print(f"\nEvaluating segmentations ...")
            eval = Evaluation(self._labels_segmentation)                
            if (isinstance(path_gt, list)):
                eval.evaluate(path_gt, out_segmentations, path_dice=path_dice)
            elif (os.path.isdir(path_gt)):
                eval.evaluate(path_gt, os.path.dirname(out_segmentations[0]), path_dice=path_dice)
            else:
                eval.evaluate(path_gt, out_segmentations[0], path_dice=path_dice)

            # output predictions in the order they are performed
            if (list_predictions is not None):
                f_list_predictions = open(os.path.join(os.path.dirname(path_dice), 'predictions.lst'), "w")
                for idx, predict in enumerate(list_predictions):
                    f_list_predictions.write(f"{idx+1}:{predict}\n")
                f_list_predictions.close()
            
            
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
            
 
