from .augmentbase import AugmentBase
from .augmentvoxynth import AugmentVoxynth


def apply_augmentations(augment_obj,
                        image_tensor,
                        label_tensor,
                        original_image,
                        original_label,
                        voxsize,
                        priors_tensor=None,
                        save_volumes=None,
                        augmentations_to_apply=None):
    output_dir = augment_obj.output_dir
    debugsaveprefix0 = None
    if (save_volumes is not None and output_dir is not None):
        import os
        import numpy as np
        from freeseg.utils import save_framedimage
        debugsaveprefix0 = os.path.join(output_dir, save_volumes)

    if (debugsaveprefix0 is not None):
        save_framedimage(
            image_tensor,
            f"{debugsaveprefix0}_reoriented_image.mgz",
            original_framedimage=original_image,            
        )
        np.save(f"{debugsaveprefix0}_reoriented_image.npy", image_tensor.cpu().numpy().astype(np.float32))
        save_framedimage(
            label_tensor,
            f"{debugsaveprefix0}_reoriented_label.mgz",
            original_framedimage=original_label,            
        )
        np.save(f"{debugsaveprefix0}_reoriented_label.npy", label_tensor.cpu().numpy().astype(np.float32))
        if (priors_tensor is not None):
            save_framedimage(
                priors_tensor,
                f"{debugsaveprefix0}_reoriented_prior.mgz",
                original_framedimage=original_image,
                dtype=float
            )
            np.save(f"{debugsaveprefix0}_reoriented_prior.npy", priors_tensor.cpu().numpy().astype(np.float32))

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
            'voxsize': voxsize,
            'geom': original_image.geom,
                }
        output = augment(input, debugsaveprefix=debugsaveprefix)
        image_tensor = output.get('image', None)
        label_tensor = output.get('label', None)
        priors_tensor = output.get('prior', None) 

        # save augmented volumes
        if (debugsaveprefix is not None):
            save_framedimage(
                image_tensor,
                f"{debugsaveprefix}_image.mgz",
                original_framedimage=original_image,            
            )
            np.save(f"{debugsaveprefix}_image.npy", image_tensor.cpu().numpy().astype(np.float32))
            save_framedimage(
                label_tensor,
                f"{debugsaveprefix}_label.mgz",
                original_framedimage=original_label,            
            )
            np.save(f"{debugsaveprefix}_label.npy", label_tensor.cpu().numpy().astype(np.float32))
            if (priors_tensor is not None):
                save_framedimage(
                    priors_tensor,
                    f"{debugsaveprefix}_prior.mgz",
                    original_framedimage=original_image,
                    dtype=float
                )
                np.save(f"{debugsaveprefix}_prior.npy", priors_tensor.cpu().numpy().astype(np.float32))

    return image_tensor, label_tensor, priors_tensor

    
def check_augmentations(augment_obj, augmentations_to_apply):
    """
    check if all requested augmentations are valid and any duplicated augmentations
    """

    for augmentation in (augmentations_to_apply):
        assert (augmentation in augment_obj.valid_augmentations), \
            f"Unknown augmentation '{augmentation}'. Supported augmentations {augment_obj.valid_augmentations}. "

    if ("flip" in augmentations_to_apply):
        assert augment_obj.left_right_corresponding is not None, "left_right_corresponding is required for augmentation 'flip'"
    if ("sampleConditionalgmm" in augmentations_to_apply):
        assert (augment_obj.generation_labels is not None), "generation_labels is required for augmentation 'sampleConditionalGMM'"
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
