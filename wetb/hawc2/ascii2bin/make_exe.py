'''
Created on 06/09/2013

@author: Mads M. Pedersen (mmpe@dtu.dk)
'''
from __future__ import division, print_function, absolute_import, unicode_literals
from future import standard_library
standard_library.install_aliases()
from build_exe.cx.build_cx_exe import NUMPY


if __name__=="__main__":
    from build_exe.cx import build_cx_exe
    build_cx_exe.build_exe("ascii2bin.py", version="3.0.1", includes=["'pandas'"], modules=['email', NUMPY])
