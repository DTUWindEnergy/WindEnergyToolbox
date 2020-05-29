
import os


def repl(path):
    return path.replace("\\", "/")


def abspath(path):
    return repl(os.path.abspath(path))


def relpath(path, start=None):
    return repl(os.path.relpath(path, start))


def realpath(path):
    return repl(os.path.realpath(path))


def pjoin(*path):
    return repl(os.path.join(*path))


def fixcase(path):
    path = realpath(str(path)).replace("\\", "/")
    p, rest = os.path.splitdrive(path)
    p += "/"
    for f in rest[1:].split("/"):
        f_lst = [f_ for f_ in os.listdir(p) if f_.lower() == f.lower()]
        if len(f_lst) > 1:
            # use the case sensitive match
            f_lst = [f_ for f_ in f_lst if f_ == f]
        if len(f_lst) == 0:
            raise IOError("'%s' not found in '%s'" % (f, p))
        # Use matched folder
        p = pjoin(p, f_lst[0])
    return p


def normpath(path):
    return repl(os.path.normpath(path))


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
