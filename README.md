# freeseg
A generic pyTorch deep-learning pipeline to accompany freesurfer adjacent models.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Quick start guide
- **[API Documentation (WIP)](docs/API.md)** - API reference for modules and classes
- **[Configuration Guide](docs/CONFIGURATION.md)** - Detailed configuration file documentation
- **[Training Guide](docs/TRAINING.md)** - Training options
- **[Architecture Documentation](docs/ARCHITECTURE.md)** - System architecture and design principles

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
fspython scripts/freeseg_create_data_list.py -d data/pgland_cropped/ -o "pgland_cropped_dataset_list.yaml"
```
Note: this is a simple script to help create your ```dataset_list.yaml``` file. It may not cover your specific use-case. But as long as your ```dataset_list.yaml``` file looks like the following, you should be good to go:
```yaml
test:
- image_filepath: /path/test_image1
  label_filepath: /path/test_segmentation1
- image_filepath: /path/test_image2
  label_filepath: /path/test_segmentation2
...
train:
- image_filepath: /path/train_image1
  label_filepath: /path/train_segmentation1
- image_filepath: /path/train_image2
  label_filepath: /path/train_segmentation2
...
validation:
- image_filepath: /path/validation_image1
  label_filepath: /path/validation_segmentation1
- image_filepath: /path/validation_image2
  label_filepath: /path/validation_segmentation2
...

  * To train the network with priors, the dataset_list.yaml needs to have 3 entries for each subject:
      image_filepath: /path/image1
      label_filepath: /path/segmentation1
      prior_filepath: /path/prior1

  * The input data directory is expected to be arranged as following, Place images, labels, and priors under their corresponding directories with same filename for each subject.
      data_folder/
        |---------- images/
        |---------- labels/
        |---------- priors/

  * The "--ignore_prior" option can be used to not generate prior entries even if the priors/ subdirectory exists.
```

4. Edit your config.yaml file as per your dataset and model requirements etc.
   Use configs/config.yaml as an example.


## Training the Model


5. Run the training script:
```bash
fspython scripts/freeseg_train.py
         --config <config.yaml>
         [--train_output_folder <train_output_folder>]	 
	 [--keep_trainset_in_memory]
         [--deterministic]
         [--checkpoint <checkpoint>]
         [--ctab <ctab>]
         [--dataset_list_file <dataset_list_file>]
	 [--model_name <model_classname>]
         [--crop_size <W H (D)>]
         [--write_tensorboard_summary]
         [--perform_evaluation]
         [--best_model_metric <loss|dice>]
         [--cpu]
	 [--vmp]
	 [--logfile <logfile>]

       * default logfile is 'freeseg_train.log'
```


## Prediction and Evaluation


6. Run the prediction script to segment any input images
```bash
fspython scripts/freeseg_predict.py 
       	 [--i <image_path> | --dataset_list_file <dataset.yaml> --cohort <train|validation|test>]
      	 --o <output_segmentations>
    	 --checkpoint <checkpoint>
    	 [--crop_size <W H (D)>]
	 [--ctab <ctab>]
	 [--label <input_labels>]
	 [--prior <input_priors>]
       	 [--gt <ground_truth>] 
       	 [--noaddctab]
       	 [--write_posteriors]
       	 [--cpu]
	 [--vmp]
	 [--logfile <logfile>]

       * Use one of the following options to specify images to segment:
         1. --i <image_path> or 
         2. --dataset_list_file <dataset.yaml> --cohort <train|validation|test>
       * Options --i <image_path> and --dataset_list_file <dataset.yaml> are mutually exclusive.
       * default logfile is 'freeseg_predict.log'
```

7. Run the evaluation script to compute dice between ground truth and segmentation
```bash
fspython scripts/freeseg_evaluate.py 
       	 --gt <ground_truth>
       	 --seg <segmentation>
         [--segmentation_labels <segmentation_labels.npy>]
         [--evaluation_labels <label1 label2 ...>]
         [--path_dice <path_dice>]
	 [--logfile <logfile>]

       * specify labels for dice evaluation using either --segmentation_labels <segmentation_labels.npy> or --evaluation_labels <label1 label2 ...>.
       * <segmentation_labels.npy> can be found in the training output directory.
       * default logfile is 'freeseg_evaluate.log'
```