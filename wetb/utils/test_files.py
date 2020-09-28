'''
Created on 20. jul. 2017

@author: mmpe
'''
import os
import wetb
import inspect
from urllib.request import urlretrieve

wetb_rep_path = os.path.abspath(os.path.dirname(wetb.__file__) + "/../").replace("\\", "/") + "/"
local_TestFiles_path = wetb_rep_path + "TestFiles/"
remote_TestFiles_url = "https://gitlab.windenergy.dtu.dk/toolbox/TestFiles/raw/master/"


def _absolute_filename(filename):
    if not os.path.isabs(filename):
        index = [os.path.realpath(s[1]) for s in inspect.stack()].index(__file__) + 2
        caller_module_path = os.path.dirname(inspect.stack()[index][1])
        tfp = caller_module_path + "/test_files/"
        filename = tfp + filename
    return os.path.abspath(filename).replace("\\", "/")


def get_test_file(filename):
    filename = _absolute_filename(filename)
    if not os.path.exists(filename):
        rel_path = os.path.relpath(filename, wetb_rep_path).replace("\\", "/")
        filename = local_TestFiles_path + rel_path
        if not os.path.exists(filename):
            urlretrieve(remote_TestFiles_url + rel_path, filename)
    return filename


def move2test_files(filename):
    filename = _absolute_filename(filename)
    assert os.path.isfile(filename), filename
    dst_filename = os.path.join(wetb_rep_path, 'TestFiles', os.path.relpath(filename, wetb_rep_path))
    folder = os.path.dirname(dst_filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.rename(filename, dst_filename)
