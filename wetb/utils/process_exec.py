'''
Created on 10/03/2014

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import str
from future import standard_library
standard_library.install_aliases()

import os


DEBUG = False
def pexec(args, cwd=None):
    """
    usage: errorcode, stdout, stderr, cmd = pexec("MyProgram.exe arg1, arg2", r"c:\tmp\")

    """
    import subprocess
    if not isinstance(args, (list, tuple)):
        args = [args]
    args = [str(arg) for arg in args]
    for i in range(len(args)):
        if os.path.exists(args[i]):
            args[i] = str(args[i]).replace('/', os.path.sep).replace('\\', os.path.sep).replace('"', '')

    cmd = "%s" % '{} /c "{}"'.format (os.environ.get("COMSPEC", "cmd.exe"), subprocess.list2cmdline(args))
    if os.path.isfile(cwd):
        cwd = os.path.dirname(cwd)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=cwd)
    stdout, stderr = proc.communicate()
    errorcode = proc.returncode

    return errorcode, stdout.decode('cp1252'), stderr.decode('cp1252'), cmd


def process(args, cwd=None):
    import subprocess
    if not isinstance(args, (list, tuple)):
        args = [args]
    args = [str(arg) for arg in args]
    for i in range(len(args)):
        if os.path.exists(args[i]):
            args[i] = str(args[i]).replace('/', os.path.sep).replace('\\', os.path.sep).replace('"', '')

    cmd = "%s" % '{} /c "{}"'.format (os.environ.get("COMSPEC", "cmd.exe"), subprocess.list2cmdline(args))
    if cwd is not None and os.path.isfile(cwd):
        cwd = os.path.dirname(cwd)
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, cwd=cwd)

def exec_process(process):
    stdout, stderr = process.communicate()
    errorcode = process.returncode

    return errorcode, stdout.decode(), stderr.decode()

def unix_filename(filename):
    """Convert case insensitive filename into unix case sensitive filename

    If more than one case insensitive matching file or folder is found, case sensitive matching is used

    Parameters
    ---------
    x : str
        Case insensitive filename

    Returns
    -------
    Filename

    """
    filename = os.path.realpath(filename.replace("\\", "/")).replace("\\", "/")
    ufn, rest = os.path.splitdrive(filename)
    ufn += "/"
    for f in rest[1:].split("/"):
        f_lst = [f_ for f_ in os.listdir(ufn) if f_.lower() == f.lower()]
        if len(f_lst) > 1:
            f_lst = [f_ for f_ in f_lst if f_ == f]
        elif len(f_lst) == 0:
            raise IOError("'%s' not found in '%s'" % (f, ufn))
        else: # one match found
            ufn = os.path.join(ufn, f_lst[0])
    return ufn.replace("\\", "/")


