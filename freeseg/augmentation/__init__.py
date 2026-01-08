from .augmentbase import AugmentBase
from .augmentvoxynth import AugmentVoxynth


def apply_augmentations(augment_obj,
                        image_tensor,
                        label_tensor,
                        original_image=None,
                        original_label=None,
                        priors_tensor=None,
                        orig_fpath=None,
                        index=None):
    output_dir = augment_obj.output_dir
    debugsaveprefix0 = None
    if (orig_fpath is not None and output_dir is not None):
        import os
        import numpy as np
        from freeseg.utils import utility as utils

        volumeprefix = os.path.basename(orig_fpath)
        for fext in ['.nii.gz', '.nii', '.mgz']:
            volumeprefix = volumeprefix.replace(fext, '')
        if (index is not None):
            volumeprefix = f"{index+1:04d}." + volumeprefix
        debugsaveprefix0 = os.path.join(output_dir, volumeprefix)        
        if (not os.path.exists(output_dir)):
            os.makedirs(output_dir)

    if (debugsaveprefix0 is not None):
        utils.save_framedimage(
            label_tensor,
            f"{debugsaveprefix0}_reoriented_label.mgz",
            original_framedimage=original_label,            
        )
        np.save(f"{debugsaveprefix0}_reoriented_label.npy", label_tensor.cpu().numpy().astype(np.float32))
        if (image_tensor is not None):
            utils.save_framedimage(
                image_tensor,
                f"{debugsaveprefix0}_reoriented_image.mgz",
                original_framedimage=original_image,            
            )
            np.save(f"{debugsaveprefix0}_reoriented_image.npy", image_tensor.cpu().numpy().astype(np.float32))
        if (priors_tensor is not None):
            utils.save_framedimage(
                priors_tensor,
                f"{debugsaveprefix0}_reoriented_prior.mgz",
                original_framedimage=original_image,
                dtype=float
            )
            np.save(f"{debugsaveprefix0}_reoriented_prior.npy", priors_tensor.cpu().numpy().astype(np.float32))

    geom = original_label.geom if (original_label is not None) else None
    augmentations_to_apply = augment_obj.transforms
    for idx, augment_name in enumerate(augmentations_to_apply):
        augment = getattr(augment_obj, augment_name, None)
        if (augment is None):
            logging.warning(f"augmentation '{augment_name}' not support, skip")
            continue
            
        debugsaveprefix = None
        if (debugsaveprefix0 is not None):
            debugsaveprefix = f"{debugsaveprefix0}_{augment_name}_{idx}"

        input = {
            'image': image_tensor,
            'label': label_tensor,
            'prior': priors_tensor,
            'geom': geom,
                }
        output = augment(input, debugsaveprefix=debugsaveprefix)
        image_tensor = output.get('image', None)
        label_tensor = output.get('label', None)
        priors_tensor = output.get('prior', None)
        geom = output.get('geom', None)

        # save augmented volumes
        if (debugsaveprefix is not None):
            utils.save_framedimage(
                label_tensor,
                f"{debugsaveprefix}_label.mgz",
                original_framedimage=original_label,
                geom=geom,
            )
            np.save(f"{debugsaveprefix}_label.npy", label_tensor.cpu().numpy().astype(np.float32))
            if (image_tensor is not None):
                utils.save_framedimage(
                    image_tensor,
                    f"{debugsaveprefix}_image.mgz",
                    original_framedimage=original_image,
                    geom=geom,
                )
                np.save(f"{debugsaveprefix}_image.npy", image_tensor.cpu().numpy().astype(np.float32))
            if (priors_tensor is not None):
                utils.save_framedimage(
                    priors_tensor,
                    f"{debugsaveprefix}_prior.mgz",
                    original_framedimage=original_image,
                    geom=geom,
                    dtype=float
                )
                np.save(f"{debugsaveprefix}_prior.npy", priors_tensor.cpu().numpy().astype(np.float32))

    # ??? todo: output final augmented label_onehot ???
    # ??? this was implemented in segmentationdataset.py ???
    
    return image_tensor, label_tensor, priors_tensor

    
def check_augmentations(augment_obj):
    """
    check if all requested augmentations are valid and any duplicated augmentations
    """

    augmentations_to_apply = augment_obj.transforms
    for augmentation in (augmentations_to_apply):
        assert (augmentation in augment_obj.valid_augmentations), \
            f"Unknown augmentation '{augmentation}'. Supported augmentations {augment_obj.valid_augmentations}. "

    if (("centroidcrop" in augmentations_to_apply) and ("centercrop" in augmentations_to_apply)):
        raise ValueError("Both 'centroidcrop' and 'centercrop' are selected. Choose one.")        
    if (("centroidcrop" in augmentations_to_apply) and ("randomcrop" in augmentations_to_apply)):
        raise ValueError("Both 'centroidcrop' and 'randomcrop' are selected. Choose one.")
    if (("centroidcrop" in augmentations_to_apply) and ("randomcentercrop" in augmentations_to_apply)):
        raise ValueError("Both 'centroidcrop' and 'randomcentercrop' are selected. Choose one.")
    if (("centercrop" in augmentations_to_apply) and ("randomcrop" in augmentations_to_apply)):
        raise ValueError("Both 'centercrop' and 'randomcrop' are selected. Choose one.")
    if (("centercrop" in augmentations_to_apply) and ("randomcentercrop" in augmentations_to_apply)):
        raise ValueError("Both 'centercrop' and 'randomcentercrop' are selected. Choose one.")
    if (("randomcrop" in augmentations_to_apply) and ("randomcentercrop" in augmentations_to_apply)):
        raise ValueError("Both 'randomcrop' and 'randomcentercrop' are selected. Choose one.")
