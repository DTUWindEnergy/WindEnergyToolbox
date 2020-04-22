# Update conda ```py36-wetb``` environment and ```wetb```

There are pre-configured miniconda/anaconda python environments installed on
Gorm and Jess at:

```
/home/python/miniconda3/envs/py36-wetb
```
Note that these refer to the home drives of Gorm and Jess respectively and thus
refer to two different directories (but are named the same).

Update the root Anaconda environment:

```
conda update --all
```

Activate the ```py36-wetb``` environment:

```
source activate py36-wetb
```


Update the ```py36-wetb``` environment:

```
conda update --all
```

Pull latest wetb changes and create re-distributable binary wheel package for ```py36-wetb```:

```
cd /home/MET/repositories/tooblox/WindEnergyToolbox
git pull
python setup.py bdist_wheel -d dist/
```

And install the wheel package (```*.whl```)

```
pip install --no-deps -U dist/wetb-X.Y.Z.post0.devXXXXXXXX-cp35m-linux_x86_64.whl
```

The option ```--no-deps``` is used here to avoid pip installing possible newer
versions of packages that should be managed by conda. This only works when all
dependencies of ```wetb``` are met (which is assumed by default for the
```py36-wetb``` environment).

