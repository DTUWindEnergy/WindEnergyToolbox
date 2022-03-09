FROM continuumio/anaconda3:latest
MAINTAINER Mikkel Friis-Møller <mikf@dtu.dk>

RUN apt-get update && \
    apt-get install make && \
    apt-get install libgl1-mesa-glx -y && \
    apt-get install gcc gfortran -y

RUN conda update -y conda && \
    conda install -y sphinx_rtd_theme && \
	conda install setuptools_scm mock h5py pytables pytest pytest-cov nose sphinx blosc pbr paramiko && \
	conda install scipy pandas matplotlib cython xlrd coverage xlwt openpyxl psutil pandoc twine pypandoc && \
	conda install -c conda-forge pyscaffold sshtunnel --no-deps && \
    conda clean -y --all

RUN pip install --upgrade pip && \
	pip install --no-cache-dir git+https://gitlab.windenergy.dtu.dk/toolbox/WindEnergyToolbox.git
