
# Installation manual


## Anaconda or Miniconda on Linux

```
conda update --all
conda create -n py36-wetb python=3.6
source activate py36-wetb
conda install setuptools_scm mock h5py pytables pytest nose sphinx blosc psutil
conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl paramiko
conda install -c https://conda.anaconda.org/conda-forge pyscaffold pytest-cov
```

## Anaconda or Miniconda on Windows

```
conda update --all
conda create -n py36-wetb python=3.6
source activate py36-wetb
conda install setuptools_scm mock h5py pytables pytest nose sphinx psutil
conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl paramiko
conda install -c https://conda.anaconda.org/conda-forge pyscaffold pytest-cov
```

