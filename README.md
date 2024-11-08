# freeseg
A generic deep-learning pipeline to accompany freesurfer adjacent models.


## Get Started

1. Clone the repo
```bash
git clone <freeseg-repo>
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
fspython scripts/freeseg_create_data_list.py -d data/pgland_cropped/ -o "pgland_cropped_dataset_list.yaml"
```
Note: this is a simple script to help create your ```dataset_list.yaml``` file. It may not cover your specific use-case. But as long as your ```dataset_list.yaml``` file looks like the following, you should be good to go:
```yaml
test:
- image_filepath: path/to/test_image1
  label_filepath: path/to/test_segmentation1
- image_filepath: path/to/test_image2
  label_filepath: path/to/test_segmentation2
...
train:
- image_filepath: path/to/train_image1
  label_filepath: path/to/train_segmentation1
- image_filepath: path/to/train_image2
  label_filepath: path/to/train_segmentation2
...
validation:
- image_filepath: path/to/validation_image1
  label_filepath: path/to/validation_segmentation1
- image_filepath: path/to/validation_image2
  label_filepath: path/to/validation_segmentation2
...
```

4. Edit your config.yaml file as per your dataset and model requirements etc.
   Use configs/config/yaml as an example.


## Training the Model


5. Run the training script:
```bash
fspython scripts/freeseg_train.py
         --config <config.yaml>
         [--checkpoint <checkpoint>]
         [--ctab <ctab>]
	 [--check_augment]
         [--dataset_list_file <dataset_list_file>]	 
         [--train_root_folder <train_root_folder>]
         [--run_name <--run_name>]
	 [--model_name <model_class>]	 
         [--crop_size <W H (D)>]
         [--write_tensorboard_summary]
         [--perform_evaluation]
         [--best_model_metric <loss|dice>]
         [--cpu]	 
```


## Prediction and Evaluation


6. Run the prediction script to segment any input images
```bash
fspython scripts/freeseg_predict.py 
       	 --i <input_images>
      	 --o <output_segmentations>
    	 --checkpoint <checkpoint>
    	 [--crop_size <W H (D)>]
	 [--ctab <ctab>]
	 [--label <input_labels>]
       	 [--gt <ground_truth>] 
       	 [--path_dice <path_dice>]
       	 [--noaddctab]
       	 [--write_posteriors]
       	 [--cpu]
```

7. Run the evaluation script to compute dice between ground truth and segmentation
```bash
fspython scripts/freeseg_evaluate.py 
       	 --gt <ground_truth>
       	 --seg <segmentation>
         [--segmentation_labels <segmentation_labels.npy>]
         [--evaluation_labels <label1 label2 ...>]
         [--path_dice <path_dice>]

       * specify labels for dice evaluation using either --segmentation_labels <segmentation_labels.npy> or --evaluation_labels <label1 label2 ...>.
       * <segmentation_labels.npy> can be found in the training output directory.
```