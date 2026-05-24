import os
import logging
import glob
from time import time
import numpy as np
import torch
import surfa as sf

from freeseg.checkpoint import Checkpoint
from freeseg.utils import utility as utils
from freeseg.augmentation import augmentbase

class Prediction:
    """
    This class run predictions, and evaluations if ground truth is given

    Attributes
    ----------

    Methods
    -------
    build_model
        Assemble the inference model

    predict
        Predict with the assembled inference model
    """
        
    def __init__(self, device=None, ctab=None, topology_classes=None, debug=False, debug_feat=False, gc=False):
        """
        Prediction Constructor.
        """

        self._inference_model = None
        self._device = device
        if (self._device is None):
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # save user input ctab, which will override the copy saved in checkpoint
        self._label_lookup = None
        if (ctab is not None):
            import surfa as sf
            self._label_lookup = sf.load_label_lookup(ctab)

        # save user input topology classes, which will override the copy saved in checkpoint
        self._topology_classes = topology_classes

        self._debug = debug
        self._debug_feat = debug_feat
        self._gc = gc
        self._out_debug_dir = None

        # save any hook handlers registered
        self._forward_pre_hooks, self._forward_hooks = [], []

        self._curr_codename = None
        self._crop_size = None


    def build_model(self, segmentation_checkpoint, parcellation_checkpoint=None, qc_checkpoint=None, flip=False, smooth_posteriors=False, smooth_sigma=0.5):
        model_info = "segmentation"
        
        # load segmentation model
        segmentation_model = self.load_segmentation_model(segmentation_checkpoint)
        
        # create the posteriors channel indices with left-right flipped labels
        if (flip and self._left_right_corresponding is None):
            logging.info(f"No left_right_corresponding found, set 'flip=False'")
            flip = False
        if (flip):
            model_info += " +flip"
            self.get_posteriors_flipped_indices()

        if (smooth_posteriors):
            model_info += " +smooth_posteriors"

        # load parcellation/qc models if applicable
        parcellation_model, qc_model = None, None        
        if (parcellation_checkpoint is not None):
            model_info += " +parcellation"
            parcellation_model = self.load_parcellation_model(parcellation_checkpoint)
        if (qc_checkpoint is not None):
            model_info += " +qc"
            qc_model = self.load_qc_model(qc_checkpoint)

        # concatenate segmentation and parcellation
        self._labels_volume = self._labels_segmentation
        self._names_volume  = self._names_segmentation
        if (parcellation_model is not None):
            self._labels_volume = np.concatenate([self._labels_volume, self._labels_parcellation[1:]])
            self._names_volume  = np.concatenate([self._names_volume,  self._names_parcellation[1:]])

        # build inference model
        self._inference_model = InferenceModel(segmentation_model,
                                             label_mapping=self._label_mapping, posterior_flipped_indices=self._posterior_flipped_indices,
                                             smooth_sigma=smooth_sigma, device=self._device, smooth_posteriors=smooth_posteriors,
                                               parcellation_model=parcellation_model, qc_model=qc_model, gc=self._gc)
        self._inference_model.eval()
        logging.info(f"Prediction.build_model(): InferenceModel = {model_info}")

            
    def load_segmentation_model(self, model_checkpoint):
        """
        load trained model

        Parameters
        ----------
        model_checkpoint : string
            path of the trained model
        """
        
        assert os.path.isfile(model_checkpoint), "The provided model path %s does not exist." % model_checkpoint

        # Load the Trained Segmentation Model
        checkpoint = Checkpoint()
        checkpoint.load(model_checkpoint, device=self._device)
        assert checkpoint.model_arch_dict is not None, "Model architecture information not available."
        assert checkpoint.train_dataset_dict is not None, "Training dataset information not available."

        the_model_name = checkpoint.model_name
        assert the_model_name is not None, "Model name is not available."

        model_class = utils.get_class(the_model_name)
        segmentation_model = model_class(checkpoint.model_arch_dict).to(self._device)

        self._nb_levels = checkpoint.model_arch_dict["nb_levels"]
        self._ndims = checkpoint.model_arch_dict["ndims"]
        assert (self._ndims == 3 or self._ndims == 2), "Model supports 3D or 2D"

        self._posterior_flipped_indices = None
        self._left_right_corresponding = checkpoint.train_dataset_dict.get("left_right_corresponding", None)
        if (self._left_right_corresponding is not None):
            # calculate number of non-sided labels
            self._num_neutral_labels = len(checkpoint.train_dataset_dict["segmentation_labels"]) - len(self._left_right_corresponding)

        # obtain sorted unique labels and their corresponding indices to the original array
        self._labels_segmentation, self._unique_idx_seg = np.unique(checkpoint.train_dataset_dict["segmentation_labels"], return_index=True)
        self._num_labels = len(self._labels_segmentation)

        # segmentation_names contains the label names corresponding to segmentation_labels
        self._names_segmentation = checkpoint.train_dataset_dict.get("segmentation_names", None)
        # self._names_segmentation can be either str to npy, or numpy array
        if (isinstance(self._names_segmentation, str)):
            self._names_segmentation = np.load(self._names_segmentation) if (os.path.exists(self._names_segmentation)) else None
        if (self._names_segmentation is not None):            
            # segmentation_names needs to be retrieved in the same order as segmentation_labels
            self._names_segmentation = self._names_segmentation[self._unique_idx_seg]
        
        # retrieve topology_classes from checkpoint, set them to classes corresponding to segmentation_labels
        # command line input overrides the copy saved in checkpoint
        if (self._topology_classes is None):
            self._topology_classes = checkpoint.train_dataset_dict.get("topology_classes", None)
        # self._topology_classes can be either str to npy, or numpy array
        if (isinstance(self._topology_classes, str)):
            self._topology_classes = np.load(self._topology_classes) if (os.path.exists(self._topology_classes)) else None
        if (self._topology_classes is not None):
            self._topology_classes = self._topology_classes[self._unique_idx_seg]

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

        # set target resolution to the resolution of the training data
        self._target_res = checkpoint.train_dataset_dict.get("target_res", None)
        self._resample_thresh = checkpoint.train_dataset_dict.get("res_diff_thresh", None)
        if (self._label_lookup is None):
            # 'label_lookup' can be either surfa.LabelLookup object or str to LUT
            self._label_lookup = checkpoint.label_lookup
            if (isinstance(self._label_lookup, str)):
                if (not os.path.exists(self._label_lookup)):
                    self._label_lookup = None
                else:
                    import surfa as sf
                    self._label_lookup = sf.load_label_lookup(self._label_lookup)

        segmentation_model.load_state_dict(checkpoint.model_state_dict)
        segmentation_model.eval()

        return segmentation_model


    def load_parcellation_model(self, model_checkpoint):
        assert os.path.isfile(model_checkpoint), "The provided model path %s does not exist." % model_checkpoint

        # Load the Trained Parcellation Model
        checkpoint = Checkpoint()
        checkpoint.load(model_checkpoint, device=self._device)
        assert checkpoint.model_arch_dict is not None, "Model architecture information not available."
        assert checkpoint.train_dataset_dict is not None, "Training dataset information not available."

        the_model_name = checkpoint.model_name
        assert the_model_name is not None, "Model name is not available."

        model_class = utils.get_class(the_model_name)
        parcellation_model = model_class(checkpoint.model_arch_dict).to(self._device)

        ###
        self._labels_parcellation = checkpoint.train_dataset_dict.get("parcellation_labels", None)
        if (isinstance(self._labels_parcellation, str)):
            self._labels_parcellation = np.load(self._labels_parcellation) if (os.path.exists(self._labels_parcellation)) else None
        if (self._labels_parcellation is not None):            
            self._labels_parcellation, unique_idx = np.unique(self._labels_parcellation, return_index=True)                

        # parcellation_names contains the label names corresponding to parcellation_labels
        self._names_parcellation = checkpoint.train_dataset_dict.get("parcellation_names", None)
        # parcellation_names needs to be retrieved in the same order as parcellation_labels
        if (isinstance(self._names_parcellation, str)):
            self._names_parcellation = np.load(self._names_parcellation) if (os.path.exists(self._names_parcellation)) else None
        if (self._names_parcellation is not None):            
            self._names_parcellation = self._names_parcellation[unique_idx]
        
        # retrieve self._label_mapping, self._inverse_label_mapping from checkpoint.train_dataset_dict
        # or compute them from self._labels_parcellation for backward compatibility
        self._parcellation_label_mapping = checkpoint.train_dataset_dict.get("parcellation_label_mapping", None)
        self._inverse_parcellation_label_mapping = checkpoint.train_dataset_dict.get("inverse_parcellation_label_mapping", None)
        if (self._parcellation_label_mapping is None):
            logging.info(f"compute parcellation_label_mapping ...")
            self._parcellation_label_mapping = {label:i for i, label in enumerate(self._labels_parcellation)}
        if (self._inverse_parcellation_label_mapping is None):
            logging.info(f"compute inverse_parcellation_label_mapping ...")
            self._inverse_parcellation_label_mapping = {v: k for k, v in self._parcellation_label_mapping.items()}
        assert self._inverse_parcellation_label_mapping is not None, "inverse_parcellation_label_mapping information not available."
        ###

        parcellation_model.load_state_dict(checkpoint.model_state_dict)
        parcellation_model.eval()

        return parcellation_model


    def load_qc_model(self, model_checkpoint):
        qc_model = None
        return qc_model

    
    def register_hook(self, module, name=''):
        def forward_hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            m_key = f"{name}.{class_name}"
            
            # forward hooks are called after the forward() call, save the output of forward()
            layer_output = os.path.join(self._out_features_dir, f"{self._curr_codename}.{m_key}.npy")
            if (isinstance(output, torch.Tensor)):
                logging.info(f"save {self._curr_codename} {m_key} {type(output)} {list(output.size())} : {layer_output}")
                # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead
                if (output.ndim == 4):  # 2D
                    output = output.unsqueeze(-1)
                np.save(layer_output, output.permute(2, 3, 4, 1, 0).cpu().detach().numpy())
            else:
                logging.info(f"{self._curr_codename} {m_key} outputs {type(output)}")

            
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
                target_res=None,
                resample_thresh=None,
                path_labels=None,
                path_priors=None,
                codenames=None,
                path_gt=None, # for hard-dice calculation
                addctab=True,
                write_posteriors=False,
                path_volumes=None,
                keepgeom=True,
                keep_biggest_component=False,
                topology_classes=None,
                segmentation_names=None,
                use_topology_classes=False):
        
        # check inputs
        assert path_images is not None, 'please specify an input file/folder'
        assert out_segmentations is not None, 'please specify an output file/folder'
        if (use_topology_classes):
            assert (self._topology_classes is not None), "'topology classes' is not specified"

        self._crop_size = crop_size
        # use target_res if it is provided, otherwise use input image resolution
        if (target_res is not None):
            self._target_res = target_res
        if (resample_thresh is not None):
            self._resample_thresh = resample_thresh
        self._keepgeom = keepgeom
        if (segmentation_names is not None):
            self._names_segmentation = segmentation_names[self._unique_idx_seg]

        path_images, path_labels, path_priors, out_segmentations, codenames, out_posteriors, csv_subjects = \
            self.prepare_output_files(path_images, path_labels, path_priors, out_segmentations, codenames, write_posteriors, path_volumes)
        
        if (self._debug):
            self._out_debug_dir = os.path.join(os.path.dirname(out_segmentations[0]), "debug")
            os.makedirs(self._out_debug_dir, exist_ok=True)
        if (self._debug_feat):
            self._out_features_dir = os.path.join(os.path.dirname(out_segmentations[0]), "debug/features")
            os.makedirs(self._out_features_dir, exist_ok=True)
            self.register_hook(self._inference_model)

        # create empty lists for predictions, label volumes, tivs, voxel counts
        self.list_predictions, self.volumes, self.tivs, self.vox_counts = [], [], [], []

        # create RescaleVolume, ResampleVolume objects
        self.apply_rescale = augmentbase.RescaleVolume(device=self._device)
        self.apply_resample = augmentbase.ResampleVolume(self._target_res, resample_thresh=self._resample_thresh, device=self._device)

        # perform segmentation
        for i in range(len(path_images)):
            ### preprocessing ###
            label_lookup = self._label_lookup
            sfimage, orig_ori, preprocessed_im_geom, preprocessed_im_shape, target_im_geom, image_tensor_preprocessed, prior_tensor_preprocessed, crop_idx, pad_idx, label_lookup = \
                self.preprocess(i, path_images, path_labels, path_priors, codenames, label_lookup)

            ### prediction ###
            with torch.no_grad():
                (posteriors_seg, posteriors_parc, qc_score) = self._inference_model(image_tensor_preprocessed, prior_tensor_preprocessed, self)

            ### postprocessing ###
            segmentation, posteriors, = \
                self.postprocess(posteriors_seg, preprocessed_im_geom.voxsize, preprocessed_im_shape, crop_idx, pad_idx,
                                 keep_biggest_component=keep_biggest_component, use_topology_classes=use_topology_classes, path_volumes=path_volumes,
                                 posteriors_parc=posteriors_parc)
            
            ### save segmentation ###
            # align prediction back to original orientation, original geom (if keepgeom is True)
            resample = True if (self._keepgeom) else False
            utils.save_framedimage(segmentation, out_segmentations[i],
                        geom=preprocessed_im_geom,
                        original_framedimage=sfimage,
                        dtype=np.int32 if (posteriors_parc is not None) else None,
                        orientation=orig_ori,
                        labels=label_lookup if (addctab) else None,
                        resample=resample, method='nearest',
                        target_im_geom=target_im_geom)
            logging.info(f"output segmentation {out_segmentations[i]}")
            if (self._debug):
                logging.debug("output prediction ...")
                np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_prediction_noresample.npy"), segmentation.cpu().movedim(0, -1).numpy().astype(np.int32))
                seg_noreshape = os.path.join(self._out_debug_dir, f"{self._curr_codename}_prediction_noresample.mgz")
                utils.save_framedimage(segmentation, seg_noreshape,
                            geom=preprocessed_im_geom,
                            original_framedimage=sfimage,
                            dtype=np.int32 if (posteriors_parc is not None) else None,
                            orientation=orig_ori,
                            labels=label_lookup if (addctab) else None)

            ### save posteriors ###
            if (write_posteriors):
                posteriors = posteriors.squeeze(0)  # remove batch axis => non-batched tensor [C, H, W (,D)]
                #posteriors = movedim(1, -1)  # move channel to last axis
                utils.save_framedimage(posteriors, out_posteriors[i], original_framedimage=sfimage, geom=preprocessed_im_geom,
                                 orientation=orig_ori, onehotencoded=True, dtype=float)
                logging.info(f"output posteriors {out_posteriors[i]}")
        # end of segmentation loop

        # remove any hooks registered
        self.unregister_hook()

        # output predictions in the order they are performed
        f_list_predictions = open(os.path.join(os.path.dirname(out_segmentations[0]), 'predictions.lst'), "w")
        for idx, predict in enumerate(self.list_predictions):
            f_list_predictions.write(f"{codenames[idx]}:{predict}\n")
        f_list_predictions.close()

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
            eval = Evaluation(self._labels_volume)                
            if (isinstance(path_gt, list)):
                eval.evaluate(path_gt, out_segmentations, path_dice=path_dice)
            elif (os.path.isdir(path_gt)):
                eval.evaluate(path_gt, os.path.dirname(out_segmentations[0]), path_dice=path_dice)
            else:
                eval.evaluate(path_gt, out_segmentations[0], path_dice=path_dice)


    def preprocess(self, idx, path_images, path_labels, path_priors, codenames, label_lookup):
        # reorient to 'RAS'
        sfimage, image_tensor, orig_geom = utils.load_framedimage(path_images[idx], orientation="RAS", device=self._device, ndims=self._ndims)
        orig_orientation = sf.transform.orientation.rotation_matrix_to_orientation(orig_geom.vox2world.matrix)
        image_tensor = image_tensor.float()
        if (label_lookup is None):
            label_lookup = sfimage.labels

        if (path_priors is not None):
            sfprior, prior_tensor, _ = utils.load_framedimage(path_priors[idx], orientation="RAS", device=self._device, ndims=self._ndims)
            assert (list(prior_tensor.shape) == [self._num_labels, *image_tensor.shape[1:]]), \
                f"Expected prior shape [self.num_classes, *image_tensor.shape[1:]], but got {list(prior_tensor.shape)}"

        self.list_predictions.append(path_images[idx])
        if (self._debug or self._debug_feat):
            self._curr_codename = f"{codenames[idx]}_{str(idx)}"
        if (self._debug):
            logging.debug("output re-oriented image/prior ...")
            out_reoriented_image = os.path.join(self._out_debug_dir, f"{self._curr_codename}_image.reoriented.RAS.mgz")
            utils.save_framedimage(image_tensor, out_reoriented_image, original_framedimage=sfimage)
            np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_image.reoriented.RAS.npy"), image_tensor.cpu().movedim(0, -1).numpy().astype(np.float32))
            if (path_priors is not None):
                out_reoriented_prior = os.path.join(self._out_debug_dir, f"{self._curr_codename}_prior.reoriented.RAS.mgz")
                utils.save_framedimage(prior_tensor, out_reoriented_prior, original_framedimage=sfprior)

        # resample image to target_res if necessary
        # original input is returned from ResampleVolume() if no resampling is necessary
        out_resample = self.apply_resample({'image':image_tensor, 'voxsize':sfimage.geom.voxsize[:image_tensor.ndim-1], 'geom':sfimage.geom, 'target_geom':orig_geom})
        image_tensor_preprocessed = out_resample.get('image')
        preprocessed_im_geom = out_resample.get('geom')
        preprocessed_im_shape = image_tensor_preprocessed.shape[1:]
        target_im_geom = out_resample.get('target_geom')  # target network output geom
        if (self._debug):
            np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_resampled_image.npy"), image_tensor_preprocessed.cpu().movedim(0, -1).numpy().astype(np.float32))

        # calculate crop_size
        if (self._crop_size is not None):
            self._crop_size = [utils.find_closest_number_divisible_by_m(s, 2 ** self._nb_levels, 'higher') for s in self._crop_size]
            
        crop_idx = None
        label_tensor = None
        prior_tensor_preprocessed = prior_tensor if (path_priors is not None) else None
        
        # check if the input image already has crop_size
        # image_tensor returned from utils.load_framedimage() is non-batched
        if (self._crop_size is not None and np.any(np.array(preprocessed_im_shape) > np.array(self._crop_size))):
            # create image cropping object
            apply_cropping = augmentbase.CenterCrop(self._crop_size, device=self._device)                

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
                np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_image.{crop}.npy"), image_tensor_preprocessed.cpu().movedim(0, -1).numpy().astype(np.float32))
                if (prior_tensor_preprocessed is not None):
                    out_cropped_prior = os.path.join(self._out_debug_dir, f"{self._curr_codename}_prior.{crop}.RAS.mgz")
                    utils.save_framedimage(prior_tensor_preprocessed, out_cropped_prior, original_framedimage=sfprior, dtype=float)
                    out_cropped_prior = os.path.join(self._out_debug_dir, f"{self._curr_codename}_prior.{crop}.mgz")
                    utils.save_framedimage(prior_tensor_preprocessed, out_cropped_prior, original_framedimage=sfprior, orientation=orig_orientation, dtype=float)
                    np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_prior.{crop}.npy"), prior_tensor_preprocessed.cpu().movedim(0, -1).numpy().astype(np.float32))
                if (path_labels is not None):
                    out_cropped_label = os.path.join(self._out_debug_dir, f"{self._curr_codename}_label.{crop}.RAS.mgz")
                    utils.save_framedimage(label_tensor_preprocessed, out_cropped_label, original_framedimage=sflabel)
                    out_cropped_label = os.path.join(self._out_debug_dir, f"{self._curr_codename}_label.{crop}.mgz")
                    utils.save_framedimage(label_tensor_preprocessed, out_cropped_label, original_framedimage=sflabel, orientation=orig_orientation)
                    np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_label.{crop}.npy"), label_tensor_preprocessed.cpu().movedim(0, -1).numpy().astype(np.float32))
                # end of debugging

        # normalize image
        if (self._debug):
            np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_before_rescale_image.npy"), image_tensor_preprocessed.cpu().movedim(0, -1).numpy().astype(np.float32))
        out_rescale = self.apply_rescale({'image':image_tensor_preprocessed})
        image_tensor_preprocessed = out_rescale.get('image')
        if (self._debug):
            np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_rescaled_image.npy"), image_tensor_preprocessed.cpu().movedim(0, -1).numpy().astype(np.float32))

        # pad image
        pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** self._nb_levels, 'higher') for s in image_tensor_preprocessed.shape[1:]]
        image_tensor_preprocessed, pad_idx = augmentbase.PadVolume(image_tensor_preprocessed, pad_shape)
        if (self._debug):
            np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_padded_image.npy"), image_tensor_preprocessed.cpu().movedim(0, -1).numpy().astype(np.float32))
                
        # add batch axes
        image_tensor_preprocessed = image_tensor_preprocessed.unsqueeze(0)
        if (prior_tensor_preprocessed is not None):
            prior_tensor_preprocessed = prior_tensor_preprocessed.unsqueeze(0)

        return sfimage, orig_orientation, preprocessed_im_geom, preprocessed_im_shape, target_im_geom, image_tensor_preprocessed, prior_tensor_preprocessed, crop_idx, pad_idx, label_lookup


    def postprocess(self, posteriors_seg, target_im_res, preprocessed_im_shape, crop_idx, pad_idx,
                    keep_biggest_component=False, use_topology_classes=False, path_volumes=None,
                    posteriors_parc=None):
        # remove the padding
        posteriors_seg = augmentbase.CropVolume(posteriors_seg.squeeze(0), pad_idx).unsqueeze(0)

        # keep biggest connected components
        if (keep_biggest_component):
            logging.info("Prediction.postprocess(): set posteriors outside the biggest connected component to zero")
            # make a copy non-background posteriors
            tmp_posteriors = posteriors_seg[:, 1:, ...].clone()
            # obtain mask for non-background labels above the threshold
            posteriors_mask = torch.sum(tmp_posteriors, axis=1) > 0.25  # [B, H, W (,D)]
            # get the largest connected component of the mask
            posteriors_mask = utils.get_largest_connected_component(posteriors_mask.cpu())         # [B, H, W (,D)]
            if (self._debug):
                np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_mask_largest_connected_component.npy"), posteriors_mask.squeeze(0).astype(np.float32))
            posteriors_mask = np.stack([posteriors_mask] * tmp_posteriors.shape[1], axis=1)  # [B, C, H, W (,D)]
            # set posteriors outside the mask to zero            
            tmp_posteriors = utils.mask_volume(tmp_posteriors.cpu().detach().numpy(), posteriors_mask)
            # update non-background posteriors
            posteriors_seg[:, 1:, ...] = tmp_posteriors

        if (use_topology_classes and self._topology_classes is not None):
            logging.info("Prediction.postprocess(): set posteriors outside the largest connected component of each topological class to zero")
            # get posteriors mask above the threshold
            posteriors_mask = posteriors_seg > 0.25
            tmp_posteriors = posteriors_seg.detach().numpy()
            # reset posteriors to zero outside the largest connected component of each non-background topological class
            for topology_class in np.unique(self._topology_classes)[1:]:
                # self._topology_classes corresponds to unique sorted labels
                # index of self._topology_classes corresponds to posteriors channel
                tmp_topology_indices = np.where(self._topology_classes == topology_class)[0]
                # obtain mask from posteriors channels belonging to the same topological class
                tmp_mask = torch.any(posteriors_mask[:, tmp_topology_indices, ...], dim=1)  # [B, H, W (,D)]
                # get largest connected component of the mask
                tmp_mask = utils.get_largest_connected_component(tmp_mask)  # [B, H, W (,D)]
                if (self._debug):
                    np.save(os.path.join(self._out_debug_dir, f"{self._curr_codename}_mask_topology_classs{topology_class}.npy"), tmp_mask.squeeze(0).astype(np.float32))
                # apply the mask to each posteriors channel belonging to the same topological class
                for idx in tmp_topology_indices:
                    tmp_posteriors[:, idx, ...] *= tmp_mask
            posteriors_seg = torch.Tensor(tmp_posteriors)
            # the cropping needs to be done for both cases, it is done at the beginning of postprocess()
            # Note that SynthSeg does the cropping here if topology classes are used to reset posteriors
            #   to zero outside the largest connected component of each topological class;
            #   otherwise, it is done at the beginning of postprocess()
            #posteriors_seg = augmentbase.CropVolume(posteriors_seg.squeeze(0), pad_idx).unsqueeze(0)
        """
        else:
            # the following logic is for 'mri_synthseg --fast'
            posteriors_mask = posteriors_seg > 0.2
            posteriors_seg[:, 1:, ...] *= posteriors_mask[:, 1:, ...]
        """

        # normalize posteriors before getting hard segmentation
        if (keep_biggest_component or (use_topology_classes and self._topology_classes is not None)):
            posteriors_seg /= torch.sum(posteriors_seg, axis=1).unsqueeze(1)

        # posteriors is batched tensor [B, C, H, W (,D)], predicted_segmentation is [B, H, W (,D)]            
        predicted_segmentation = torch.argmax(posteriors_seg, dim=1)
        # map labels to original id
        if (path_volumes is None):
            segmentation_cropped = utils.remap_labels(predicted_segmentation, self._inverse_label_mapping)
        else:
            segmentation_cropped, vox_counts = utils.remap_labels(predicted_segmentation, self._inverse_label_mapping, return_counts=True)

        # postprocess parcellation
        if (posteriors_parc is not None):
            # remove the padding
            posteriors_parc = augmentbase.CropVolume(posteriors_parc.squeeze(0), pad_idx).unsqueeze(0)

            # obtain parcellation mask from segmentation prediction
            parcellation_mask = (segmentation_cropped == 3) | (segmentation_cropped == 42)  # [B, H, W (,D)]
            # preset background label posteriors to all 1 (the background label includes white matter)
            posteriors_parc[:, 0, ...] = torch.ones_like(posteriors_parc[:, 0, ...])
            # apply parcellation mask to background label posteriors, set posteriors outside the mask to 0
            posteriors_parc[:, 0, ...] = utils.mask_volume(posteriors_parc[:, 0, ...].clone().cpu().detach().numpy(), (parcellation_mask.numpy() < 0.1))
            # normalize posteriors
            posteriors_parc /= torch.sum(posteriors_parc, axis=1).unsqueeze(1)
            
            # get hard parcellation, posteriors is batched tensor [B, C, H, W (,D)], predicted_parcellation is [B, H, W (,D)]            
            predicted_parcellation = torch.argmax(posteriors_parc, dim=1)
            # map cortex labels to original id
            parcellation_cropped = utils.remap_labels(predicted_parcellation, self._inverse_parcellation_label_mapping)

            # paste the cortex labels to segmentation
            segmentation_cropped[parcellation_mask] = parcellation_cropped[parcellation_mask]
        
        if (crop_idx is not None):
            # re-position posteriors and predicted segmentation back to the original image indices where the image was cropped out
            segmentation = np.zeros(shape=(segmentation_cropped.shape[0], *preprocessed_im_shape), dtype='int32')
            posteriors = np.zeros(shape=(posteriors_seg.shape[0], posteriors_seg.shape[1], *preprocessed_im_shape), dtype='float32')
            posteriors[:, 0, ...] = np.ones((posteriors_seg.shape[0], *preprocessed_im_shape))  # preset background posterior
            if (self._ndims == 3):
                segmentation[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = segmentation_cropped.detach().cpu().numpy()
                posteriors[..., crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = posteriors_seg.detach().cpu().numpy()
            else:
                segmentation[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] = segmentation_cropped.detach().cpu().numpy()
                posteriors[..., crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] = posteriors_seg.detach().cpu().numpy()
            segmentation = torch.from_numpy(segmentation).to(self._device)
            posteriors_seg = torch.from_numpy(posteriors).to(self._device)
        else:
            segmentation = segmentation_cropped

        # compute volumes
        if (path_volumes is not None):
            # skip background
            if (posteriors_seg.is_cuda):
                posteriors_seg = posteriors_seg.cpu()
            volumes = np.sum(posteriors_seg.detach().numpy()[:, 1:, ...], tuple(range(2, 2+len(posteriors_seg.shape[2:]))))
            volumes = volumes.squeeze(0)
            tiv = np.array([np.sum(volumes)])  # sum up all volumes except background
            volumes = np.around(volumes * np.prod(target_im_res), 3)            
            tiv = np.around(tiv * np.prod(target_im_res), 3)
            self.volumes.append(volumes.tolist())
            self.tivs.append(tiv.tolist())
            self.vox_counts.append(vox_counts)

        return segmentation, posteriors_seg


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


    # create the posteriors channel indices with left-right labels flipped
    # it is to be called after obtaining sorted unique segmentation labels and their corresponding indices to the original array
    def get_posteriors_flipped_indices(self):
        # build left->right and right->left label mappings
        left_right_mapping = {}
        for idx in range(0, len(self._left_right_corresponding), 2):
            left_right_mapping.update({self._left_right_corresponding[idx] : self._left_right_corresponding[idx+1]})
            left_right_mapping.update({self._left_right_corresponding[idx+1] : self._left_right_corresponding[idx]})
    
        # keep track of labels processed
        labels_seg_processed = []

        # create left-right flipped posterior channels
        self._posterior_flipped_indices = np.arange(self._num_labels)
        for (posterior_channel, unique_idx) in enumerate(self._unique_idx_seg):
            if (len(labels_seg_processed) == self._num_labels):
                break
        
            label = self._labels_segmentation[posterior_channel]
            labels_seg_processed.append(label)
            if (unique_idx < self._num_neutral_labels):
                # non-sided labels, keep their posterior channels
                self._posterior_flipped_indices[posterior_channel] = posterior_channel
            else:
                # get the label on opposite side of the brain
                other_label = left_right_mapping[label]
                labels_seg_processed.append(other_label)
                # get its corresponding posterior_channel (its position index to self._labels_segmentation)
                other_posterior_channel = np.where(self._labels_segmentation == other_label)[0][0]
                # flip left-right label posterior channels
                self._posterior_flipped_indices[posterior_channel] = other_posterior_channel
                self._posterior_flipped_indices[other_posterior_channel] = posterior_channel
                #print(f"flip left-right label ({label:02d}, {other_label:02d}) posterior channels {posterior_channel:2d} <=> {other_posterior_channel:2d}")                

        # save the indices as list
        self._posterior_flipped_indices = list(self._posterior_flipped_indices)
        logging.info(f"posteriors channel indices with left-right flipped labels: {self._posterior_flipped_indices}")


class InferenceModel(torch.nn.Module):
    def __init__(self, segmentation_model, label_mapping=None, posterior_flipped_indices=None,
                 smooth_sigma=0.5, device=None, smooth_posteriors=False,
                 parcellation_model=None, qc_model=None, gc=False):
        super(InferenceModel, self).__init__()
        self._segmentation_model = segmentation_model
        self._parcellation_model = parcellation_model
        self._qc_model = qc_model
        self._gc = gc

        self._device = device
        if (self._device is None):
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._smooth_sigma = smooth_sigma
        self._gaussianblur = None
        if (smooth_posteriors):
            # ??? todo: check synthseg kernel size ???
            self._gaussianblur = augmentbase.GaussianBlur(device=self._device)

        self._posterior_flipped_indices = posterior_flipped_indices
        self._inverse_label_mapping = {v: k for k, v in label_mapping.items()}
        self._cortex_label_mapping = label_mapping.copy()
        for k, v in label_mapping.items():
            if (k == 3 or k == 42):
                self._cortex_label_mapping[k] = 1
            else:
                self._cortex_label_mapping[k] = 0


    def forward(self, x, x1=None, predict_obj=None):
        if (self._gc):
            import gc

        orig_prefix, out_debug_dir = None, None
        if (predict_obj is not None):
             orig_prefix = predict_obj._curr_codename
             out_debug_dir = predict_obj._out_debug_dir
        posteriors_parc, qc_score = None, None

        with torch.no_grad():
            (posteriors_seg, _) = self._segmentation_model(x, x1)
        if (out_debug_dir is not None):
            np.save(os.path.join(out_debug_dir, f"{predict_obj._curr_codename}_inf_posteriors_seg.npy"), posteriors_seg.squeeze(0).movedim(0, -1).cpu().numpy().astype(np.float32))

        if (self._gaussianblur is not None):
            # smooth posteriors
            blur_out = self._gaussianblur({'image':posteriors_seg.squeeze(0)}, self._smooth_sigma)
            posteriors_seg = blur_out.get("image", None).unsqueeze(0)
            if (out_debug_dir is not None):
                np.save(os.path.join(out_debug_dir, f"{predict_obj._curr_codename}_inf_posteriors_seg_smooth.npy"), posteriors_seg.squeeze(0).movedim(0, -1).cpu().numpy().astype(np.float32))

        if (self._posterior_flipped_indices is not None):
            if (predict_obj is not None):
                predict_obj._curr_codename = f"{predict_obj._curr_codename}.flip"            

            # assuming the input x and x1 ([B, C, H, W(, D)]) are in RAS
            axis = 2  # left-right axis

            # predict left-right flipped image
            x_flipped = x.flip([axis])
            x1_flipped = x1.flip([axis]) if (x1 is not None) else x1
            with torch.no_grad():
                (posteriors_seg_flipped, _) = self._segmentation_model(x_flipped, x1_flipped)
            if (out_debug_dir is not None):
                np.save(os.path.join(out_debug_dir, f"{predict_obj._curr_codename}_inf_posteriors_seg.npy"), posteriors_seg_flipped.squeeze(0).movedim(0, -1).cpu().numpy().astype(np.float32))

            # flip the posteriors back
            posteriors_seg_flipped = posteriors_seg_flipped.flip([axis])
            # re-order the posteriors channels to match left-right flipped labels
            posteriors_seg_flipped = posteriors_seg_flipped = posteriors_seg_flipped[:, self._posterior_flipped_indices, ...]
            if (out_debug_dir is not None):
                np.save(os.path.join(out_debug_dir, f"{predict_obj._curr_codename}_inf_posteriors_seg_flipped.npy"), posteriors_seg_flipped.squeeze(0).movedim(0, -1).cpu().numpy().astype(np.float32))

            # average two posteriors
            posteriors_seg = (posteriors_seg + posteriors_seg_flipped) / 2
            if (out_debug_dir is not None):
                np.save(os.path.join(out_debug_dir, f"{predict_obj._curr_codename}_inf_posteriors_seg_avg.npy"), posteriors_seg.squeeze(0).movedim(0, -1).cpu().numpy().astype(np.float32))

        if (self._gc):
            del(self._segmentation_model)
            gc.collect()

        if (self._parcellation_model is not None):
            if (predict_obj is not None):
                predict_obj._curr_codename = orig_prefix
            
            seg = torch.argmax(posteriors_seg, dim=1)
            seg = utils.remap_labels(seg, self._inverse_label_mapping)
            cortex = utils.remap_labels(seg, self._cortex_label_mapping)
            onehot_cortex = utils.onehot(cortex, num_classes=2)    

            inputs = torch.cat([x, onehot_cortex], dim=1)
            with torch.no_grad():
                (posteriors_parc, _) = self._parcellation_model(inputs)
            if (out_debug_dir is not None):
                np.save(os.path.join(out_debug_dir, f"{predict_obj._curr_codename}_inf_posteriors_parc.npy"), posteriors_parc.squeeze(0).movedim(0, -1).cpu().numpy().astype(np.float32))                
            if (self._gc):
                del(self._parcellation_model)
                gc.collect()

            if (self._gaussianblur is not None):
                # smooth posteriors
                blur_out = self._gaussianblur({'image':posteriors_parc.squeeze(0)}, self._smooth_sigma)
                posteriors_parc = blur_out.get("image", None).unsqueeze(0)
                if (out_debug_dir is not None):
                    np.save(os.path.join(out_debug_dir, f"{predict_obj._curr_codename}_inf_posteriors_parc_smooth.npy"), posteriors_parc.squeeze(0).movedim(0, -1).cpu().numpy().astype(np.float32))

        if (self._qc_model is not None):
            if (predict_obj is not None):
                predict_obj._curr_codename = orig_prefix
            with torch.no_grad():
                qc_score = self._qc_model(posteriors_seg)
            if (out_debug_dir is not None):
                np.save(os.path.join(out_debug_dir, f"{predict_obj._curr_codename}_inf_qc_score.npy"), qc_score.squeeze(0).movedim(0, -1).cpu().numpy().astype(np.float32))
            if (self._gc):
                del(self._qc_model)
                gc.collect()

        return posteriors_seg, posteriors_parc, qc_score
