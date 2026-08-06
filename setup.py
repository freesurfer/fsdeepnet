import setuptools

setuptools.setup(
    name = "fsdeepnet",
    version = "0.1",
    description = ("A generic deep-learning pytorch pipeline and freesurfer pre-trained models."),
    url = "https://github.com/freesurfer/fsdeepnet",
    packages=setuptools.find_packages(),
    #python_requires='==3.9',
)


"""
# project directory structure

fsdeepnet/
|------ README.md
|------ setup.py
|------ fsdeepnet/
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
|       |------ cli/
|       |       |------ __init__.py
|       |       |------ fsdeepnet_split_dataset.py
|       |       |------ fsdeepnet_evaluate.py
|       |       |------ fsdeepnet_predict.py
|       |       |------ fsdeepnet_train.py
|       |       |------ fsdeepnet_checkpoint.py
|       |       |------ tf2torch/
|       |       |       |------ __init__.py
|       |       |       |------ loadh5_synthseg.py
|------ configs/
|       |------ config.yaml
|       |------ data_list.yaml

# 1. fsdeepnet package gets intalled into python/lib/python3.8/site-package/fsdeepnet
# 2. scripts/fsdeepnet_*.py are copied to python/bin
#    ??? todo: during Freesurfer python install (python/CMakeLists.txt),
#              a. move them to python/scripts before they are removed
#              b. make their fspython wrappers in $FREESURFER_HOME/bin/
# 3. fsdeepnet.voxynth is only temporary until we get our changes into their git repo
#    make voxynth an install_requires, pip install from its git repo
"""


"""
To publish package on testpypi (need separated account for pypi):
(https://packaging.python.org/en/latest/guides/using-testpypi/)
(https://packaging.python.org/en/latest/tutorials/packaging-projects/)
(https://setuptools.pypa.io/en/latest/userguide/quickstart.html)
1. using testpypi
   - create an account on testpypi
   - create API token
   - create ~/.pypirc with the following line:
     [testpypi]
       username = __token__
       password = <api-token>
     * the <api-token> is prefixed with 'pypi-'
2. build distribution artifacts
   - python -m pip install build  # install build package
   - python -m build .            # build source and built distributions
     python -m build --sdist .    # build only the source distribution
     python -m build --wheel .    # build only the built distribution
     * check 'Requires-Python': unzip -p dist/*.whl */METADATA | grep "Requires-Python"
3. upload the distribution to testpypi
   - python -m pip install twine  # install twine package
   - python -m twine upload --verbose --repository testpypi dist/*
     * upload dist/fsdeepnet-0.1-py3-none-any.whl and fsdeepnet-0.1.tar.gz
4. install from testpypi
   python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fsdeepnet
"""
