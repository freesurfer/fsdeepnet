import pytest
import torch
import yaml
from utils.dataset import SegmentationDataset

with open('configs/dataset_config.yaml', 'r') as file:
    dataset_config = yaml.safe_load(file)

expected_num_channels = dataset_config['dataset']['expected_num_channels']
expected_classes = dataset_config['dataset']['expected_classes']

@pytest.fixture
def dataset():
    return SegmentationDataset('configs/dataset_list.yaml')

def test_dataset_length(dataset):
    assert len(dataset) > 0, "Dataset is empty"

def test_dataset_shape(dataset):
    image, label = dataset[0]
    assert image.shape[1:] == label.shape[1:], "Image and label shapes do not match"

def test_dataset_channels(dataset):
    image, _ = dataset[0]
    assert image.shape[0] == expected_num_channels, f"Expected {expected_num_channels} channels, but got {image.shape[0]}"

def test_dataset_classes(dataset):
    unique_classes = set()
    for _, label in dataset:
        unique_values = torch.unique(label).tolist()
        unique_classes.update(unique_values)

    assert len(unique_classes) > 0, "No classes found in the dataset"
    assert sorted(unique_classes) == expected_classes, f"Expected classes {expected_classes}, but got {sorted(unique_classes)}"

# def test_data_loading():
#     """Tests if the dataset loads data correctly."""
#     dataset = MedicalImageSegmentationDataset(data_list_path="data/dataset.csv", n_input=1, n_class=2)
#     image, label = dataset[0]
#     assert image.shape == (1, 160, 160, 160)  # Replace with expected image shape
#     assert label.shape == (1, 160, 160, 160)  # Replace with expected label shape

# def test_augmentations():
#     """Tests if augmentations are applied correctly (if applicable)."""
#     # ... Implement test cases for your augmentations