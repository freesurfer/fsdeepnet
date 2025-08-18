# Tutorials

## Get Started

1. Clone the repo
```bash
git clone https://github.com/freesurfer/freeseg.git <freeseg-repo>
```

2. Install freeseg package
```bash
a. Build and install your Freesurfer with cmake options '-DDISTRIBUTE_FSPYTHON=ON -DFSPYTHON_INSTALL_CUDA=ON'
b. Source your Freesurfer environment
c. Install freeseg package
   cd <freeseg-repo>
   fspython -m pip install .

   * For editable install:  fspython -m pip install --editable .
```

3. Setup the train/validation/test dataset
```bash
cd tutorials/
fspython ../scripts/freeseg_create_data_list.py \
	 -d testdata/dummy_binary/ \
	 -o configs/dataset_list_dummy_binary.yaml
```
Note: Image/label pairs are split into train/validation/test datasets with default ratio: train_ratio=0.7, val_ratio=0.15, test_ratio=0.15

4. Edit your config.yaml file as per your dataset and model requirements etc.
   Use configs/config.yaml as an example.


## Training the Model

5. Run the training script:
```bash
cd tutorials/
fspython ../scripts/freeseg_train.py \
	 --config configs/config_dummy_binary.yaml \
	 --dataset_list_file configs/dataset_list_dummy_binary.yaml \
	 --ctab configs/dummy_binary.ctab \
	 --train_output_folder experiment/train.dummy_binary
```


## Prediction and Evaluation

6. Run the prediction script to segment any input images
```bash
cd tutorials/
fspython ../scripts/freeseg_predict.py \
	 --checkpoint experiment/train.dummy_binary/dice_010.pth \
	 --dataset_list_file configs/dataset_list_dummy_binary.yaml --cohort test \
	 --o experiment/predict.testcohort

       * Use one of the following options to specify images to segment:
         1. --i <image_path> or 
         2. --dataset_list_file <dataset.yaml> --cohort <train|validation|test>
       * Options --i <image_path> and --dataset_list_file <dataset.yaml> are mutually exclusive.
```

## Other tools

7. Generate augmentation volumes for visualization
```bash
cd tutorials/
fspython ../tests/test_segmentationdataset.py \
	 --config configs/config_dummy_binary.yaml \
	 --dataset_list_file configs/dataset_list_dummy_binary.yaml \
	 --augment --augmentation_dir experiment/augment
```

8. Print model summary and parameters
```bash
cd tutorials/
fspython ../scripts/freeseg_checkmodel.py --config configs/config_dummy_binary.yaml
```

```bash
fspython ../scripts/freeseg_checkmodel.py --checkpoint experiment/train.dummy_binary/dice_010.pth
```