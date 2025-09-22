from setuptools import setup

requirements = [
    'torch',
    'torchvision<0.17',  # to satisfy torch 2.1.2
    #'pynvml',
    #'voxynth@git+https://github.com/dalcalab/voxynth.git@main'
]

setup(
    name = "freeseg",
    version = "0.1",
    description = ("A generic deep-learning pipeline to accompany freesurfer adjacent models."),
    url = "https://github.com/freesurfer/freeseg",
    packages=['freeseg', 'freeseg.utils', 'freeseg.augmentation', 'freeseg.models', 'freeseg.datasets', 'freeseg.voxynth'],
    install_requires=requirements,
    scripts=['scripts/freeseg_split_dataset.py', 'scripts/freeseg_evaluate.py', 'scripts/freeseg_predict.py', 'scripts/freeseg_train.py'],
)


"""
# project directory structure

freeseg/
|------ README.md
|------ setup.py
|------ freeseg/
|       |------ __init__.py
|       |------ training.py
|       |------ prediction.py
|       |------ evaluation.py
|       |------ checkpoint.py
|       |------ metrics.py
|       |------ models/
|       |       |------ __init__.py
|       |       |------ unet.py
|       |------ datasets/
|       |       |------ __init__.py
|       |       |------ segmentationdataset.py
|       |------ augmentation/
|       |       |------ __init__.py
|       |       |------ preprocessing.py
|       |------ utils/
|       |       |------ __init__.py
|       |       |------ utility.py
|       |------ voxynth/
|       |       |------ readme.md
|       |       |------ __init__.py
|       |       |------ augment.py
|       |       |------ filter.py
|       |       |------ noise.py
|       |       |------ synth.py
|       |       |------ transform.py
|       |       |------ utility.py
|------ scripts/
|       |------ freeseg_split_dataset.py
|       |------ freeseg_evaluate.py
|       |------ freeseg_predict.py
|       |------ freeseg_train.py
|------ configs/
|       |------ config.yaml
|       |------ data_list.yaml

# 1. freeseg package gets intalled into python/lib/python3.8/site-package/freeseg
# 2. scripts/freeseg_*.py are copied to python/bin
#    ??? todo: during Freesurfer python install (python/CMakeLists.txt),
#              a. move them to python/scripts before they are removed
#              b. make their fspython wrappers in $FREESURFER_HOME/bin/
# 3. freeseg.voxynth is only temporary until we get our changes into their git repo
#    make voxynth an install_requires, pip install from its git repo
"""
