
# Installation manual


## Anaconda or Miniconda

```
conda update --all
conda create -n wetb_py3 python=3.5
source activate wetb_py3
conda install setuptools_scm future h5py pytables pytest nose sphinx
conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl
pip install pyscaffold pytest-cov
```

