# Convert sclimbic tensorflow models to pytorch

1. source Freesurfer environment
```bash
export FREESURFER_HOME=<freesurfer+fsdeepnet env>
source $FREESURFER_HOME/SetUpFreeSurfer.sh fs+fsdeepnet

export tf2toch=../../../cli/fsdeepnet_tf2torch.py
mkdir pretrained
cd configs
```

2. convert mca-dura model
```bash
fspython $tf2torch --tf_model $FREESURFER_HOME/models/mca-dura.both-lh.nstd21.fhs.h5 \
	 --torch_model_saveas ../pretrained/pt.mca-dura.both-lh.nstd21.fhs.pth \
	 --config mca-dura_config.yaml \
	 --model_layer_mapping model_layer_mapping.yaml
```

3. convert vsinus model
```bash
fspython $tf2torch --tf_model $FREESURFER_HOME/models/vsinus.no-sp.m.all.nstd10-070.h5 \
	 --torch_model_saveas ../pretrained/pt.vsinus.no-sp.m.all.nstd10-070.pth \
	 --config vsinus_config.yaml \
	 --model_layer_mapping model_layer_mapping.yaml
```
	 
4. convert entowm model
```bash
fspython $tf2torch --tf_model $FREESURFER_HOME/models/entowm.fsm31.t1.nstd00-30.nstd21-108.h5 \
	 --torch_model_saveas ../pretrained/pt.entowm.fsm31.t1.nstd00-30.nstd21-108.pth \
	 --config entowm_config.yaml \
	 --model_layer_mapping model_layer_mapping.yaml
```

5. convert sclimbic model
```bash
fspython $tf2torch --tf_model $FREESURFER_HOME/models/sclimbic.fsm+ad.t1.nstd00-50.nstd32-50.h5 \
	 --torch_model_saveas ../pretrained/pt.sclimbic.fsm+ad.t1.nstd00-50.nstd32-50.pth \
	 --config sclimbic_config.yaml \
	 --model_layer_mapping model_layer_mapping.yaml
```
