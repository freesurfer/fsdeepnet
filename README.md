# freeseg
A generic deep-learning pipeline to accompany freesurfer adjacent models.


## Get Started

1. Clone teh repo
```bash
git clone <freeseg-repo>
```

2. Source the accompanying Conda environment:
```bash
source /autofs/space/curv_001/users/avnish/miniconda3/bin/activate /autofs/space/curv_001/users/avnish/miniconda3/envs/pgland/
```

3. Copy voxynth library code into the freeseg directory
```bash
cp /autofs/space/curv_001/users/avnish/for_Doug/Pituitary-segmentation/voxynth/ <freeseg-repo>
```

4. Setup the train/validation/test dataset
```bash
python create_data_list.py -d data/pgland_cropped/ -o "pgland_cropped_dataset_list.yaml"
```
Note: this is a rudimentary scrip to help create your ```dataset_list.yaml``` file. It may not cover your specific dataset. But as long as your ```dataset_list.yaml``` file looks like the following, you should be good to go:
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
5. Edit your configs/config.yaml file as per your dataset and model requirements etc.


## Training the Model

To train the model on your dataset, follow these steps:


6. Run the training script:
```bash
python train.py --config configs/config.yaml
```

## Evaluating the Model

To evaluate the model on your dataset, follow these steps:

7. Run the evaluation script:
```bash
python evaluate.py --config configs/config.yaml
```