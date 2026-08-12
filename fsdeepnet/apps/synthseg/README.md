# Convert SynthSeg tensorflow models to pytorch

1. source Freesurfer environment
```bash
export FREESURFER_HOME=<freesurfer+fsdeepnet env>
source $FREESURFER_HOME/SetUpFreeSurfer.sh fs+fsdeepnet

export tf2toch=../../../cli/fsdeepnet_tf2torch.py
mkdir pretrained
cd configs
```

2. convert synthseg model
```bash
fspython $tf2torch --tf_model $FREESURFER_HOME/models/synthseg_2.0.h5 \
	 --torch_model_saveas ../pretrained/pt.synthseg_2.0.pth \
	 --config synthseg_config.yaml \
	 --model_layer_mapping synthseg_model_layer_mapping.yaml \
	 --set_dataset_attr
```

3. convert synthseg_parc model
```bash
fspython $tf2torch --tf_model $FREESURFER_HOME/models/synthseg_parc_2.0.h5 \
	 --torch_model_saveas ../pretrained/pt.synthseg_parc_2.0.h5.pth \
	 --config synthseg_parc_config.yaml \
	 --model_layer_mapping synthseg_parc_model_layer_mapping.yaml \
	 --set_dataset_attr
```

4. run prediction
```bash
# segmentation
fspython ../../../cli/fsdeepnet_predict.py --i <invol> --o <seg> --threads <> \
	 --checkpoint <pt.synthseg_2.0.pth> \
	 --flip --keep_biggest_component --use_topology_classes --smooth_posteriors \
	 --cpu --logfile <fsdeepnetlog>

# segmentation + parcellation
fspython ../../../cli/fsdeepnet_predict.py --i <invol> --o <seg> --threads <> \
	 --checkpoint pt.synthseg_2.0.pth --parc pt.synthseg_parc_2.0.h5.pth \
	 --flip --keep_biggest_component --use_topology_classes --smooth_posteriors \
	 --cpu --logfile <fsdeepnetlog>
```