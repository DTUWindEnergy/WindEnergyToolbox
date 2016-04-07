'''
Created on 29/01/2016

@author: mmpe
'''
import pytest
import os
p = os.path.abspath(os.path.join(os.path.dirname(__file__) + "./../")).replace("\\", "/")
pytest.main(p)
