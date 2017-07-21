'''
Created on 20. jul. 2017

@author: mmpe
'''
import os
import wetb
import inspect
wetb_rep_path = os.path.join(os.path.dirname(wetb.__file__), "../")                                   


def _absolute_filename(filename):
    if not os.path.isabs(filename):
        index = [os.path.realpath(s[1]) for s in inspect.stack()].index(__file__) + 2
        caller_module_path = os.path.dirname(inspect.stack()[index][1])
        tfp = caller_module_path + "/test_files/"
        filename = tfp + filename
    return filename

def get_test_file(filename):
    filename = _absolute_filename(filename) 
    if os.path.exists(filename):
        return filename
    else:
        return os.path.join(wetb_rep_path, 'TestFiles', os.path.relpath(filename, wetb_rep_path))
        



def move2test_files(filename):
    filename = _absolute_filename(filename)
    assert os.path.isfile(filename), filename
    dst_filename = os.path.join(wetb_rep_path, 'TestFiles', os.path.relpath(filename, wetb_rep_path))
    folder = os.path.dirname(dst_filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.rename(filename, dst_filename)
    
    
    
    
    