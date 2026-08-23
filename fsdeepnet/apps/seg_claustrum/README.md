# Convert Freesurfer Claustrum tensorflow models to pytorch

## 1. source Freesurfer environment
```bash
export FREESURFER_HOME=<freesurfer+fsdeepnet env>
source $FREESURFER_HOME/SetUpFreeSurfer.sh fs+fsdeepnet

export tf2torch=../../../cli/fsdeepnet_tf2torch.py
mkdir pretrained
cd configs
```

## 2. convert claustrum model
```bash
fspython $tf2torch --tf_model $FREESURFER_HOME/models/claustrum_seg_20250616.h5 \
	 --torch_model_saveas ../pretrained/pt.claustrum_seg_20250616.pth \
	 --config claustrum_config.yaml \
	 --model_layer_mapping model_layer_mapping.yaml
```

## 3. run prediction
```bash
fspython ../../../cli/fsdeepnet_predict.py --i <involcrop> --o <seg> --threads <> \
	 --checkpoint <pt.claustrum_seg_20250616.pth> --nokeepgeom \
	 --keep_biggest_component --use_topology_classes --smooth_posteriors \
	 --resamplefirst --logfile <fsdeepnetlog>
```

---
# References

- **A Constrast-Agnostic Method for Ultra-High Resolution Claustrum Segmentation** \
Mauri, C., Fritz, R., Mora, J., Billot, B., Iglesias, J.E., Van Leemput, K., Augustinack, J., Greve, D.N. \
Human Brain Mapping 46.12 (2025): e70303. \
[article](https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.70303)

