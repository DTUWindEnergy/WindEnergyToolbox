
# Installation manual


## Anaconda or Miniconda on Linux

```
conda update --all
conda create -n wetb_py3 python=3.5
source activate wetb_py3
conda install setuptools_scm future h5py pytables pytest nose sphinx blosc psutil
conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl paramiko
conda install -c https://conda.anaconda.org/conda-forge pyscaffold pytest-cov
```

## Anaconda or Miniconda on Windows

```
conda update --all
conda create -n wetb_py3 python=3.4
source activate wetb_py3
conda install setuptools_scm future h5py pytables pytest nose sphinx psutil
conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl paramiko
conda install -c https://conda.anaconda.org/conda-forge pyscaffold pytest-cov
```

