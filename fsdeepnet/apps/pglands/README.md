# Integrate Freesurfer mri_pglands_seg

## 1. Integrate network 
- apps/pglands/models/unet3d.py

  Network classes extracted from https://github.com/freesurfer/freesurfer/blob/dev/mri_pglands_seg/mri_pglands_seg
  
- apps/pglands/models/pglands_unet.py
  - class UNet is derived from fsdeepnet.apps.pglands.models.unet3d.UNet3D
  - required _model_arch_dict keywords: `num_channels`, `nb_labels`, `nb_levels`, `ndims`
  - required methods: _setdefault_arch_dict(), _update_arch_dict(model_arch_dict), forward()
  - required property: arch_dict
	
## 2. Create config.yaml
- apps/pglands/configs/pglands_config.yaml

  based on the network parameters used to create UNet3D in https://github.com/freesurfer/freesurfer/blob/dev/mri_pglands_seg/mri_pglands_seg

## 3. Source Freesurfer environment
```bash
export FREESURFER_HOME=<freesurfer+fsdeepnet env>
source $FREESURFER_HOME/SetUpFreeSurfer.sh fs+fsdeepnet

mkdir pretrained
```

## 4. Integrate Freesurfer pglands pytorch model
- update checkpoint with model_arch_dict and train_dataset_dict information from config.yaml
- rename model_state:model_state_dict
- strip optimizer_state from checkpoint
```bash
  fspython ../../cli/fsdeepnet_checkpoint.py \
  	   --checkpoint $FREESURFER_HOME/models/pglands_seg/pglands_seg.pth \
  	   --update config:config --config configs/pglands_config.yaml \
	   --rename model_state:model_state_dict \	   
	   --strip optimizer_state \
	   --saveas pretrained/fsdeepnet.pglands_seg.pth
```

## 5. Integrate prediction script
- apps/pglands/cli/pglands_seg.py

## 6. Test integrated prediction script with converted checkpoint
```bash
   mkdir tests
   cd tests
   fspython ../cli/pglands_seg.py -i bert.orig.mgz --o fsdeepnet.pglands_seg.mgz --model ../pretrained/fsdeepnet.pglands_seg.pth
```
* notes: Comparing to Freesurfer `mri_pglands_seg` run, `apps/pglands/cli/pglands_seg.py` makes extra call to `fs-synthmorph-reg --i /usr/local/freesurfer/rocky8_x86_64/subjects/bert/mri/nu.mgz --o transforms/mni152 --mni-out-res 1.0mm --mni-targ-res 1.0mm --no-crop --pituitary --affine-only
mri_synthmorph apply -m nearest transforms/mni152/reg.targ_to_invol.lta /autofs/cluster/scratch_wednesday//yh887/freesurfer.install.py39+fsdeepnet.final/models/pglands_seg/mni152_label_template.mgz mni152_label_template.nu.mgz`
