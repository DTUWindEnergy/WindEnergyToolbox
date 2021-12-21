'''
Created on 10/03/2014

@author: MMPE
'''
import glob
import re

import os


DEBUG = False


def pexec(args, cwd=None, shell=False):
    """
    usage: errorcode, stdout, stderr, cmd = pexec("MyProgram.exe arg1, arg2", r"c:\tmp\")

    """
    proc, cmd = process(args, cwd)
    errorcode, stdout, stderr = exec_process(proc)
    return errorcode, stdout, stderr, cmd


def process(args, cwd=None, shell=False):
    import subprocess
    if not isinstance(args, (list, tuple)):
        args = [args]
    args = [str(arg) for arg in args]
    for i in range(len(args)):
        if os.path.exists(args[i]):
            args[i] = str(args[i]).replace('/', os.path.sep).replace('\\', os.path.sep).replace('"', '')

    cmd_args = subprocess.list2cmdline(args)
    if cwd and os.path.isfile(cwd):
        cwd = os.path.dirname(cwd)
    if cwd:
        cmd_cwd = "cd %s && " % cwd
    else:
        cmd_cwd = ""

    if os.name == 'nt':
        cmd = '%s /c "%s%s"' % (os.environ.get("COMSPEC", "cmd.exe"), cmd_cwd, cmd_args)
    else:
        cmd = '%s%s' % (cmd_cwd, cmd_args)

    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell, cwd=cwd), cmd


def exec_process(process):
    stdout, stderr = process.communicate()
    errorcode = process.returncode

    return errorcode, stdout.decode(), stderr.decode()
