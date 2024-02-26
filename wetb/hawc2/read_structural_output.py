# -*- coding: utf-8 -*-
"""
Read HAWC2 results files produced by the `new_htc_structure` commands.

@author: ricriv
"""

def read_body_matrix_output(file_name):
    """
    Read output of `new_htc_structure` / `body_matrix_output`.

    Parameters
    ----------
    file_name : str
        File path.

    Returns
    -------
    mass : ndarray of shape (ndof, ndof)
        Mass matrix.
    damping : ndarray of shape (ndof, ndof)
        Damping matrix.
    stiffness  : ndarray of shape (ndof, ndof)
        Stiffness matrix.

    """
    pass
