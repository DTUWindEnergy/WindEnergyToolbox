
import os


def repl(path):
    return path.replace("\\", "/")


def abspath(path):
    return repl(os.path.abspath(path))


def relpath(path, start=None):
    return repl(os.path.relpath(path, start))


def pjoin(*path):
    return repl(os.path.join(*path))


def cluster_path(path):
    try:
        import win32wnet
        drive, folder = os.path.splitdrive(abspath(path))
        path = abspath(win32wnet.WNetGetUniversalName(drive, 1) + folder)
    except Exception:
        path = repl(path)
    path = path.replace("//jess.dtu.dk", "/home")
    path = path.replace("//mimer.risoe.dk/aiolos", "/mnt/aiolos")
    path = path.replace("//mimer.risoe.dk", "/mnt/mimer")
    return path
