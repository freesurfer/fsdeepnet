# Convert Claustrum tensorflow models to pytorch

1. source Freesurfer environment
```bash
export FREESURFER_HOME=<freesurfer+fsdeepnet env>
source $FREESURFER_HOME/SetUpFreeSurfer.sh fs+fsdeepnet

export tf2toch=../../../cli/tf2torch/loadh5_synthseg.py
mkdir pretrained
cd configs
```

2. convert claustrum model
```bash
fspython $tf2torch --tf_model $FREESURFER_HOME/models/claustrum_seg_20250616.h5 \
	 --torch_model_saveas ../pretrained/pt.claustrum_seg_20250616.pth \
	 --config claustrum_config.yaml \
	 --model_layer_mapping model_layer_mapping.yaml
```

3. run prediction
```bash
fspython ../../../cli/fsdeepent_predict.py --i <involcrop> --o <seg> --threads <> 
	 --checkpoint <pt_model> --nokeepgeom --keep_biggest_component --use_topology_classes --smooth_posteriors --resamplefirst --logfile <fsdeepnetlog>
```
