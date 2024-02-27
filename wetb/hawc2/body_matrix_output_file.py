# -*- coding: utf-8 -*-
"""
Read HAWC2 results files produced by the `new_htc_structure` commands.

@author: ricriv
"""

# %% Import.

from glob import glob
import numpy as np


# %% Functions.


def read_body_matrix_output(file_name):
    """
    Read output of `new_htc_structure` / `body_matrix_output`, which is the mass, damping and stiffness matrices for each body.

    Parameters
    ----------
    file_name : str
        File name entered in the
        `new_htc_structure` / `body_matrix_output`
        command. A glob patter will be used to find all the files.

    Returns
    -------
    bodies : dict
        Dictionary containing the structural matrices for each body.
        Its keys are:
            mass : ndarray of shape (ndof, ndof)
                Mass matrix.
            damping : ndarray of shape (ndof, ndof)
                Damping matrix.
            stiffness : ndarray of shape (ndof, ndof)
                Stiffness matrix.
    """
    # Get all files with the mass matrix.
    files = glob(f"{file_name}*_m.bin")

    # We get the body names by taking the part between
    # the file_name and _m.bin.
    names = [s[len(file_name) : -6] for s in files]

    # Dict that will contain all the results.
    bodies = dict.fromkeys(names)

    # Loop over the bodies.
    for name in names:
        bodies[name] = {}

        # Read mass matrix.
        with open(f"{file_name}{name}_m.bin", "rb") as fid:
            ndof = np.fromfile(fid, dtype=np.int32, count=2)[0]
            bodies[name]["mass"] = np.fromfile(
                fid, dtype=np.float64, count=ndof * ndof
            ).reshape(ndof, ndof, order="F")

        # Read damping matrix.
        with open(f"{file_name}{name}_c.bin", "rb") as fid:
            ndof = np.fromfile(fid, dtype=np.int32, count=2)[0]
            bodies[name]["damping"] = np.fromfile(
                fid, dtype=np.float64, count=ndof * ndof
            ).reshape(ndof, ndof, order="F")

        # Read stiffness matrix.
        with open(f"{file_name}{name}_k.bin", "rb") as fid:
            ndof = np.fromfile(fid, dtype=np.int32, count=2)[0]
            bodies[name]["stiffness"] = np.fromfile(
                fid, dtype=np.float64, count=ndof * ndof
            ).reshape(ndof, ndof, order="F")

    return bodies
