# Integrate Freesurfer mri_synthstrip

## 1. Integrate Network 
- apps/synthstrip/models/stripmodel.py

  class StripModel and class ConvBlock are extracted from https://github.com/freesurfer/freesurfer/blob/dev/mri_synthstrip/mri_synthstrip
  
- apps/synthstrip/models/synthstrip_unet.py

  - class UNet is derived from StripModel
  - required _model_arch_dict keywords: `nb_levels`, `ndims`
  - required methods: _setdefault_arch_dict(), _update_arch_dict(model_arch_dict), forward()
  - required property: arch_dict

## 2. Create config.yaml
- apps/synthstrip/configs/synthstrip_config.yaml

  based on the network parameters used to create StripModel in https://github.com/freesurfer/freesurfer/blob/dev/mri_synthstrip/mri_synthstrip

## 3. Source Freesurfer environment
```bash
export FREESURFER_HOME=<freesurfer+fsdeepnet env>
source $FREESURFER_HOME/SetUpFreeSurfer.sh fs+fsdeepnet

mkdir pretrained
```

## 4. Integrate pre-trained pytorch model
- update checkpoint with model_arch_dict and train_dataset_dict information from configs/synthstrip_config.yaml
- strip 'optimizer_state_dict' from checkpoint
```bash
	fspython ../../cli/fsdeepnet_checkpoint.py \
        	 --checkpoint $FREESURFER_HOME/models/synthstrip.1.pt \
         	 --update config:config --config configs/synthstrip_config.yaml \
         	 --strip optimizer_state_dict \
         	 --saveas pretrained/fsdeepnet.synthstrip.1.pth
```

## 5. Integrate prediction script
- apps/synthstrip/cli/synthstrip.py

## 6. Test integrated prediction script with converted checkpoint
```bash
   mkdir tests
   cd tests
   fspython ../cli/synthstrip.py --threads 2 -i bert.orig.mgz -o fsdeepnet.synthstrip.mgz --model ../pretrained/fsdeepnet.synthstrip.1.pth
```
