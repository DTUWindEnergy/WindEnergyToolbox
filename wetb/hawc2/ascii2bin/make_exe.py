'''
Created on 06/09/2013

@author: Mads M. Pedersen (mmpe@dtu.dk)
'''


if __name__=="__main__":
    from mmpe.build_exe.cx.build_cx_exe import NUMPY
    from mmpe.build_exe.cx import build_cx_exe
    build_cx_exe.build_exe("ascii2bin.py", version="3.0.1", includes=["'pandas'"], modules=['email', NUMPY])
