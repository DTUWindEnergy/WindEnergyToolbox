'''
Created on 20. jul. 2017

@author: mmpe
'''
import os
import wetb
import urllib.request
from fileinput import filename
import inspect
wetb_rep_path = os.path.join(os.path.dirname(wetb.__file__), "../")                                   
default_TestFile_rep_path=os.path.join(os.path.dirname(wetb.__file__) + "/../../TestFiles/")

def get_test_file(filename):
    if not os.path.isabs(filename):
        index = [os.path.realpath(s[1]) for s in inspect.stack()].index(__file__) + 1
        tfp = os.path.dirname(inspect.stack()[index][1]) + "/test_files/"
        filename = tfp + filename
    
    if os.path.exists(filename):
        return filename
    else:
        filename2 = os.path.realpath(os.path.join(wetb_rep_path, 'downloaded_test_files', os.path.relpath(filename, wetb_rep_path)))
        if not os.path.isfile(filename2):
            #url = 'https://gitlab.windenergy.dtu.dk/toolbox/TestFiles/%s'%os.path.relpath(filename, wetb_rep_path)
            url = 'http://tools.windenergy.dtu.dk/TestFiles/%s.txt'%os.path.relpath(filename, wetb_rep_path).replace("\\","/")
            print ("download %s\nfrom %s"%(filename, url))
            if not os.path.exists(os.path.dirname(filename2)):
                os.makedirs(os.path.dirname(filename2))
            urllib.request.urlretrieve(url, filename2)
        return filename2



def move2test_files(filename,TestFile_rep_path=default_TestFile_rep_path):
    wetb_rep_path = os.path.join(os.path.dirname(wetb.__file__), "../")
    folder = os.path.dirname(TestFile_rep_path + os.path.relpath(filename, wetb_rep_path))
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.rename(filename, os.path.join(folder, os.path.basename(filename)+'.txt'))
    
    
    
    
    