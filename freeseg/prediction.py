import os
import logging
import glob
from time import time
import numpy as np
import torch
import surfa as sf

from freeseg.checkpoint import Checkpoint
from freeseg.utils import utility as utils
from freeseg.augmentation.augmentbase import CenterCrop, RescaleVolume, ResampleVolume, PadVolume, CropVolume

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
        
    def __init__(self, device=None, ctab=None, debug=False):
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

        self._debug = debug
        self._out_debug_dir = None

        # save any hook handlers registered
        self._forward_pre_hooks, self._forward_hooks = [], []

        self._curr_codename = None
        self._crop_size = None


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

        model_class = utils.get_class(the_model_name, "freeseg.models.unet")
        self._model = model_class(checkpoint.model_arch_dict).to(self._device)

        self._nb_levels = checkpoint.model_arch_dict["nb_levels"]
        self._ndims = checkpoint.model_arch_dict["ndims"]
        assert (self._ndims == 3 or self._ndims == 2), "Model supports 3D or 2D"
        
        self._labels_segmentation, self._unique_idx = np.unique(checkpoint.train_dataset_dict["segmentation_labels"], return_index=True)
        self._num_labels = len(self._labels_segmentation)

        # segmentation_names contains the label names corresponding to segmentation_labels
        self._names_segmentation = checkpoint.train_dataset_dict.get("segmentation_names", None)
        if (self._names_segmentation is not None):
            # segmentation_names needs to be retrieved in the same order as segmentation_labels
            self._names_segmentation = self._names_segmentation[self._unique_idx]
        
        # retrieve self._label_mapping, self._inverse_label_mapping from checkpoint.train_dataset_dict
        # or compute them from self._labels_segmentation for backward compatibility
        self._label_mapping = checkpoint.train_dataset_dict.get("label_mapping", None)
        self._inverse_label_mapping = checkpoint.train_dataset_dict.get("inverse_label_mapping", None)
        if (self._label_mapping is None):
            logging.info(f"compute label_mapping ...")
            self._label_mapping = {label:i for i, label in enumerate(self._labels_segmentation)}
        if (self._inverse_label_mapping is None):
            logging.info(f"compute inverse_label_mapping ...")
            self._inverse_label_mapping = {v: k for k, v in self._label_mapping.items()}
        

        if (self._label_lookup is None):
            self._label_lookup = checkpoint.label_lookup
        self._model.load_state_dict(checkpoint.model_state_dict)
        self._model.eval()


    def register_hook(self, module, name=''):
        def forward_hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            m_key = f"{name}.{class_name}"
            
            # forward hooks are called after the forward() call, save the output of forward()
            layer_output = os.path.join(self._out_debug_dir, f"{self._curr_codename}_layerout.{m_key}.npy")
            logging.info(f"save {self._curr_codename} layer {m_key} output {list(output.size())} : {layer_output}")
            # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead
            np.save(layer_output, output.permute(2, 3, 4, 1, 0).cpu().detach().numpy())

            
        if isinstance(module, (list, tuple)):
            it = iter(module)
            # iterate through the list two elements at a time
            for nm, mod in zip(it, it):
                name += nm if len(name) == 0 else "." + nm
                self.register_hook(mod, name)
        else:
            chld_iter = module.named_children()
            chld_count = len(list(chld_iter))
            if (chld_count == 0):
                self._forward_hooks.append(module.register_forward_hook(forward_hook))
            else:
                for idx, chld in enumerate(module.named_children()):
                   self.register_hook(chld, name)


    def unregister_hook(self):
        # remove any hooks registered
        for h in self._forward_hooks:
            h.remove()
        for h in self._forward_pre_hooks:
            h.remove()


    def predict(self,
                path_images,
                out_segmentations,
                crop_size=None,
                target_res=1.,
                path_labels=None,
                path_priors=None,
                codenames=None,
                path_gt=None, # for hard-dice calculation
                addctab=True,
                write_posteriors=False,
                path_volumes=None,
                keepgeom=True,
                keep_biggest_component=True,
                topology_classes=None,
                segmentation_names=None):
        
        # check inputs
        assert path_images is not None, 'please specify an input file/folder'
        assert out_segmentations is not None, 'please specify an output file/folder'

        self._crop_size = crop_size
        self._target_res = target_res
        self._keepgeom = keepgeom
        if (segmentation_names is not None):
            self._names_segmentation = segmentation_names[self._unique_idx]

        path_images, path_labels, path_priors, out_segmentations, codenames, out_posteriors, csv_subjects = \
            self.prepare_output_files(path_images, path_labels, path_priors, out_segmentations, codenames, write_posteriors, path_volumes)
        
        if (self._debug):
            self._out_debug_dir = os.path.join(os.path.dirname(out_segmentations[0]), "debug")
            os.makedirs(self._out_debug_dir, exist_ok=True)            
            self.register_hook(self._model)

        # create empty lists for predictions, label volumes, tivs, voxel counts
        self.list_predictions, self.volumes, self.tivs, self.vox_counts = [], [], [], []

        # create RescaleVolume, ResampleVolume, CropVolume objects
        self.apply_rescale = RescaleVolume(device=self._device)
        self.apply_resample = ResampleVolume(self._target_res, device=self._device)

        # perform segmentation
        for i in range(len(path_images)):
            ### preprocessing ###
            label_lookup = self._label_lookup
            sfimage, orig_ori, target_im_geom, target_im_shape, image_tensor_preprocessed, prior_tensor_preprocessed, crop_idx, pad_idx, label_lookup = \
                self.preprocess(i, path_images, path_labels, path_priors, codenames, label_lookup)

            ### prediction ###
            (outputs, _) = self._model(image_tensor_preprocessed, prior_tensor_preprocessed)

            ### postprocessing ###
            segmentation, posteriors, = \
                self.postprocess(outputs, target_im_geom.voxsize, target_im_shape, crop_idx, pad_idx, keep_biggest_component=True, topology_classes=None, path_volumes=path_volumes)
            
            ### save segmentation ###
            # align prediction back to original orientation, original geom (if keepgeom is True)
            resample = True if (self._keepgeom) else False
            utils.save_framedimage(segmentation, out_segmentations[i],
                        geom=target_im_geom,
                        original_framedimage=sfimage,
                        orientation=orig_ori,
                        labels=label_lookup if (addctab) else None,
                        resample=resample, method='nearest')
            logging.info(f"output segmentation {out_segmentations[i]}")
            if (self._debug):
                logging.debug("output cropped prediction ...")
                np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_prediction.cropped.npy"), segmentation.cpu().numpy().astype(np.float32))
                seg_noreshape = os.path.join(self._out_debug_dir, f"{self._curr_codename}_prediction.cropped.mgz")
                utils.save_framedimage(segmentation, seg_noreshape,
                            geom=target_im_geom,
                            original_framedimage=sfimage, 
                            orientation=orig_ori,
                            labels=label_lookup if (addctab) else None)

            ### save posteriors ###
            if (write_posteriors):  # ??? question: posteriors need to be re-positioned as well ???
                posteriors = posteriors.squeeze(0)  # remove batch axis => non-batched tensor [C, H, W (,D)]
                #posteriors = movedim(1, -1)  # move channel to last axis
                utils.save_framedimage(posteriors, out_posteriors[i], original_framedimage=sfimage, geom=target_im_geom,
                                 orientation=orig_ori, onehotencoded=True, dtype=float)
                logging.info(f"output posteriors {out_posteriors[i]}")
        # end of segmentation loop

        self.unregister_hook()

        # write volumes
        if (path_volumes):
            # create a new list with subject, tiv inserted at the beginning
            # 'csv_subjects' is a list, 'self.tivs' and 'self.volumes' are list of lists
            csv_rows = [[s] + t + vol for s, t, vol in zip(csv_subjects, self.tivs, self.volumes)]
            if (self._names_segmentation is not None):
                names_label = [label for label in self._names_segmentation]
            else:
                names_label = [f'label {str(label)}' for label in self._labels_segmentation]
            # exclude background id
            utils.write_csv(path_volumes, csv_rows, header=[['subject', 'total intracranial'] + names_label[1:]])
                
            path_stats = path_volumes.replace('.csv', '.stats')
            # build a list of label id/name pairs
            names_labels = [(id, name) for id, name in zip(self._labels_segmentation, names_label)]
            # exclude background id
            utils.write_volume_stats(path_stats, self.vox_counts, self.volumes, names_labels[1:]) 

        # evaluate
        if (path_gt is not None):
            # calculate hard-dice between saved segmentations and their ground truth
            from freeseg.evaluation import Evaluation

            path_dice = os.path.join(os.path.dirname(out_segmentations[0]), 'dices.npy')

            logging.info("")  # empty line
            logging.info(f"Evaluating segmentations ...")
            eval = Evaluation(self._labels_segmentation)                
            if (isinstance(path_gt, list)):
                eval.evaluate(path_gt, out_segmentations, path_dice=path_dice)
            elif (os.path.isdir(path_gt)):
                eval.evaluate(path_gt, os.path.dirname(out_segmentations[0]), path_dice=path_dice)
            else:
                eval.evaluate(path_gt, out_segmentations[0], path_dice=path_dice)

            # output predictions in the order they are performed
            f_list_predictions = open(os.path.join(os.path.dirname(path_dice), 'predictions.lst'), "w")
            for idx, predict in enumerate(self.list_predictions):
                f_list_predictions.write(f"{codenames[idx]}:{predict}\n")
            f_list_predictions.close()


    def preprocess(self, idx, path_images, path_labels, path_priors, codenames, label_lookup):
        # reorient to 'RAS'
        sfimage, image_tensor, orig_orientation = utils.load_framedimage(path_images[idx], orientation="RAS", device=self._device, ndims=self._ndims)
        image_tensor = image_tensor.float()
        if (label_lookup is None):
            label_lookup = sfimage.labels

        if (path_priors is not None):
            sfprior, prior_tensor, orig_ori_prior = utils.load_framedimage(path_priors[idx], orientation="RAS", device=self._device, ndims=self._ndims)
            assert (list(prior_tensor.shape) == [self._num_labels, *image_tensor.shape[1:]]), \
                f"Expected prior shape [self.num_classes, *image_tensor.shape[1:]], but got {list(prior_tensor.shape)}"

        self.list_predictions.append(path_images[idx])
        if (self._debug):
            self._curr_codename = f"{codenames[idx]}_{str(idx)}"
            logging.debug("output re-oriented image/prior ...")
            out_reoriented_image = os.path.join(self._out_debug_dir, f"{self._curr_codename}_image.reoriented.RAS.mgz")
            utils.save_framedimage(image_tensor, out_reoriented_image, original_framedimage=sfimage)
            if (path_priors is not None):
                out_reoriented_prior = os.path.join(self._out_debug_dir, f"{self._curr_codename}_prior.reoriented.RAS.mgz")
                utils.save_framedimage(prior_tensor, out_reoriented_prior, original_framedimage=sfprior)

        # resample image to target_res
        out_resample = self.apply_resample({'image':image_tensor, 'voxsize':sfimage.geom.voxsize[:image_tensor.ndim-1], 'geom':sfimage.geom})
        image_tensor_preprocessed = out_resample.get('image')
        target_im_geom = out_resample.get('geom')
        target_im_shape = image_tensor_preprocessed.shape[1:]        
        if (self._debug):
            np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_resampled_image.npy"), image_tensor_preprocessed.cpu().numpy().astype(np.float32))

        # calculate crop_size
        if (self._crop_size is not None):
            self._crop_size = [utils.find_closest_number_divisible_by_m(s, 2 ** self._nb_levels, 'higher') for s in self._crop_size]
            
        crop_idx = None
        label_tensor = None
        prior_tensor_preprocessed = prior_tensor if (path_priors is not None) else None
        
        # check if the input image already has crop_size
        # image_tensor returned from utils.load_framedimage() is non-batched
        if (self._crop_size is not None and np.any(np.array(target_im_shape) > np.array(self._crop_size))):
            # create image cropping object
            apply_cropping = CenterCrop(self._crop_size, device=self._device)                

            # apply CentroidCrop if label image is available
            if (path_labels is not None):
                apply_cropping = CentroidCrop(self._crop_size, device=self._device)
                sflabel, label_tensor, _ = utils.load_framedimage(path_labels[idx], orientation="RAS", device=self._device, ndims=self._ndims)
                assert (label_tensor.shape == image_tensor.shape), \
                    f"image and label need to be in the same shape. label {path_labels[idx]} has shape {label_tensor.shape}, image {path_images[idx]} has shape {image_tensor.shape}"

                if (label_lookup is None):
                    label_lookup = sflabel.labels

            # crop image
            out_cropping = apply_cropping({'image':image_tensor_preprocessed, 'label':label_tensor, 'prior':prior_tensor_preprocessed})
            image_tensor_preprocessed = out_cropping.get('image')
            label_tensor_preprocessed = out_cropping.get('label')
            prior_tensor_preprocessed = out_cropping.get('prior')
            crop_idx = out_cropping.get('crop_idx')                
            image_tensor_preprocessed = image_tensor_preprocessed.to(self._device).float()

            if (self._debug):
                # begin of debugging
                crop = 'centercropped'
                if (path_labels is not None):
                    crop = 'centroidcropped'

                logging.debug(f"output {crop} image/label ...")
                out_cropped_image = os.path.join(self._out_debug_dir, f"{self._curr_codename}_image.{crop}.RAS.mgz")
                utils.save_framedimage(image_tensor_preprocessed, out_cropped_image, original_framedimage=sfimage)
                out_cropped_image = os.path.join(self._out_debug_dir, f"{self._curr_codename}_image.{crop}.mgz")
                utils.save_framedimage(image_tensor_preprocessed, out_cropped_image, original_framedimage=sfimage, orientation=orig_orientation)
                if (prior_tensor_preprocessed is not None):
                    out_cropped_prior = os.path.join(self._out_debug_dir, f"{self._curr_codename}_prior.{crop}.RAS.mgz")
                    utils.save_framedimage(prior_tensor_preprocessed, out_cropped_prior, original_framedimage=sfprior, dtype=float)
                    out_cropped_prior = os.path.join(self._out_debug_dir, f"{self._curr_codename}_prior.{crop}.mgz")
                    utils.save_framedimage(prior_tensor_preprocessed, out_cropped_prior, original_framedimage=sfprior, orientation=orig_orientation, dtype=float)
                if (path_labels is not None):
                    out_cropped_label = os.path.join(self._out_debug_dir, f"{self._curr_codename}_label.{crop}.RAS.mgz")
                    utils.save_framedimage(label_tensor_preprocessed, out_cropped_label, original_framedimage=sflabel)
                    out_cropped_label = os.path.join(self._out_debug_dir, f"{self._curr_codename}_label.{crop}.mgz")
                    utils.save_framedimage(label_tensor_preprocessed, out_cropped_label, original_framedimage=sflabel, orientation=orig_orientation)
                # end of debugging

        # normalize image
        if (self._debug):
            np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_before_rescale_image.npy"), image_tensor_preprocessed.cpu().numpy().astype(np.float32))
        out_rescale = self.apply_rescale({'image':image_tensor_preprocessed})
        image_tensor_preprocessed = out_rescale.get('image')
        if (self._debug):
            np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_rescaled_image.npy"), image_tensor_preprocessed.cpu().numpy().astype(np.float32))

        # pad image
        pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** self._nb_levels, 'higher') for s in image_tensor_preprocessed.shape[1:]]
        image_tensor_preprocessed, pad_idx = PadVolume(image_tensor_preprocessed, pad_shape)
                
        # add batch axes
        image_tensor_preprocessed = image_tensor_preprocessed.unsqueeze(0)
        if (prior_tensor_preprocessed is not None):
            prior_tensor_preprocessed = prior_tensor_preprocessed.unsqueeze(0)

        return sfimage, orig_orientation, target_im_geom, target_im_shape, image_tensor_preprocessed, prior_tensor_preprocessed, crop_idx, pad_idx, label_lookup


    def postprocess(self, posteriors, target_im_res, target_im_shape, crop_idx, pad_idx, keep_biggest_component=True, topology_classes=None, path_volumes=None):
        # remove the padding
        posteriors = CropVolume(posteriors.squeeze(0), pad_idx).unsqueeze(0)

        # ??? todo: keep_biggest_component ???

        # ??? todo: reset posteriors to zero outside the largest connected component of each topological class ???

        # ??? todo: normalize posteriors before getting hard segmentation ???
        # posteriors is batched tensor [B, C, H, W (,D)], predicted_segmentation is [B, H, W (,D)]
        predicted_segmentation = torch.argmax(posteriors, dim=1)
        # map labels to original id
        if (path_volumes is None):
            segmentation_cropped = utils.remap_labels(predicted_segmentation, self._inverse_label_mapping)
        else:
            segmentation_cropped, vox_counts = utils.remap_labels(predicted_segmentation, self._inverse_label_mapping, return_counts=True)
        
        if (crop_idx is not None):
            # re-position predicted segmentation back to the original image indices where the image was cropped out
            segmentation = np.zeros(shape=(segmentation_cropped.shape[0], *target_im_shape), dtype='int32')
            if (self._ndims == 3):
                segmentation[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = segmentation_cropped.detach().cpu().numpy()
            else:
                segmentation[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] = segmentation_cropped.detach().cpu().numpy()
            segmentation = torch.from_numpy(segmentation).to(self._device)
        else:
            segmentation = segmentation_cropped

        # compute volumes
        if (path_volumes is not None):
            # skip background
            volumes = np.sum(posteriors.detach().numpy()[:, 1:, ...], tuple(range(2, 2+len(posteriors.shape[2:]))))
            volumes = volumes.squeeze(0)
            tiv = np.array([np.sum(volumes)])  # sum up all volumes except background
            volumes = np.around(volumes * np.prod(target_im_res), 3)            
            tiv = np.around(tiv * np.prod(target_im_res), 3)
            self.volumes.append(volumes.tolist())
            self.tivs.append(tiv.tolist())
            self.vox_counts.append(vox_counts)

        return segmentation, posteriors


    def prepare_output_files(self, path_images, path_labels, path_priors, out_segmentations, codenames, write_posteriors, path_volumes):
        pred_suffix = 'prediction'
        posteriors_suffix = 'posteriors'

        # convert path to absolute paths
        if (not isinstance(path_images, list)):
            path_images = os.path.abspath(path_images)
            if (path_labels is not None):
                path_labels = os.path.abspath(path_labels)
            if (path_priors is not None):
                path_priors = os.path.abspath(path_priors)

        convert_single = False
        # expand image/label into list
        if (isinstance(path_images, list)):
            # case 1: list of images
            if (path_labels is not None):
                assert os.path.isdir(path_labels), '%s need to be a directory\n' % (path_images, path_labels)
                path_labels = sorted(glob.glob(os.path.join(path_labels, '*.nii.gz')) +
                                     glob.glob(os.path.join(path_labels, '*.nii')) +
                                     glob.glob(os.path.join(path_labels, '*.mgz')))
        elif (os.path.isdir(path_images)):
            # case 2: directory of images
            if (path_labels is not None):
                assert os.path.isdir(path_labels), 'both %s and %s need to be directory' % (path_images, path_labels)
            if (path_priors is not None):
                assert os.path.isdir(path_priors), 'both %s and %s need to be directory' % (path_images, path_priors)

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
        else:
            # case 3: single image
            assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                                'please make sure the path and the extension are correct' % path_images
            assert (not os.path.isdir(out_segmentations)), 'both %s and %s need to be files' % (path_images, out_segmentations)

            # if output has a directory component, create the directory if it doesn't exist yet
            out_dir = os.path.dirname(out_segmentations)
            if (out_dir):
                os.makedirs(out_dir, exist_ok=True)

            convert_single = True
            if (path_labels is not None):
                assert os.path.isfile(path_labels), 'both %s and %s need to be file \n' % (path_images, path_labels)
                path_labels = [path_labels]
            if (path_priors is not None):
                assert os.path.isfile(path_priors), 'both %s and %s need to be file \n' % (path_images, path_priors)
                path_priors = [path_priors]
                
            path_images = [path_images]
            out_segmentations = [out_segmentations]
            
        # if 'codenames' is provided, use it as the output filename prefix;
        # otherwise, use sequential numbers (starting from 1) padded with leading zeros
        if (codenames is None):
               codenames = [str(nn).zfill(4) for nn in range(1, len(path_images)+1)]
            
        if (not convert_single):
            # case 1 & 2
            if (not os.path.exists(out_segmentations)):
                os.makedirs(out_segmentations)
            assert os.path.isdir(out_segmentations), '%s need to be a directory to segment multiple images' % (out_segmentations)

            # pre-generate all *_predict* filenames
            out_segmentations = [os.path.join(out_segmentations, f"{codenames[idx]}."+os.path.basename(p)) for idx, p in enumerate(path_images)]
            out_segmentations = [p.replace('.nii', '.%s.nii' % pred_suffix) for p in out_segmentations]
            out_segmentations = [p.replace('.mgz', '.%s.mgz' % pred_suffix) for p in out_segmentations]            
                    
        # check path_images, path_labels, path_priors, and out_segmentations have the same length
        assert (len(path_images) == len(out_segmentations)), "input images and output segmentations need to be the same length"
        if (path_labels is not None):
            assert (len(path_images) == len(path_labels)), "images and labels need to be the same length"        
        if (path_priors is not None):
            assert (len(path_images) == len(path_priors)), "images and priors need to be the same length"

        out_posteriors = []
        if (write_posteriors):
            out_posteriors_dir = os.path.join(os.path.dirname(out_segmentations[0]), "posteriors")
            os.makedirs(out_posteriors_dir, exist_ok=True)
            for out_seg in out_segmentations:
                basename = os.path.basename(out_seg)
                out_posteriors.append(os.path.join(out_posteriors_dir, basename.replace(f"{pred_suffix}.", f"{posteriors_suffix}.")))

        csv_subjects = []
        if (path_volumes):
            for im in path_images:
                im = os.path.basename(im).replace('.nii.gz', '')
                im = im.replace('.nii', '')
                im = im.replace('.mgz', '')
                csv_subjects.append(im)
                
        return path_images, path_labels, path_priors, out_segmentations, codenames, out_posteriors, csv_subjects
