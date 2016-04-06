'''
Created on 05/11/2015

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import unittest
import os
import filecmp
import pickle

from wetb.prepost import dlctemplate as tmpl


class TestGenerateInputs(unittest.TestCase):

    def setUp(self):
        self.basepath = os.path.dirname(__file__)

    def test_launch_dlcs_excel(self):
        # manually configure paths, HAWC2 model root path is then constructed as
        # p_root_remote/PROJECT/sim_id, and p_root_local/PROJECT/sim_id
        # adopt accordingly when you have configured your directories differently
        p_root = os.path.join(self.basepath, 'data/')
        # project name, sim_id, master file name
        tmpl.PROJECT = 'demo_dlc'
        tmpl.MASTERFILE = 'demo_dlc_master_A0001.htc'
        # MODEL SOURCES, exchanche file sources
        tmpl.P_RUN = os.path.join(p_root, tmpl.PROJECT, 'remote/')
        tmpl.P_SOURCE = os.path.join(p_root, tmpl.PROJECT, 'source/')
        # location of the master file
        tmpl.P_MASTERFILE = os.path.join(p_root, tmpl.PROJECT,
                                         'source', 'htc', '_master/')
        # location of the pre and post processing data
        tmpl.POST_DIR = os.path.join(p_root, tmpl.PROJECT, 'remote',
                                     'prepost/')
        tmpl.force_dir = tmpl.P_RUN
        tmpl.launch_dlcs_excel('remote', silent=True)

        # we can not check-in empty dirs in git
        for subdir in ['control', 'data', 'htc', 'pbs_in']:
            remote = os.path.join(p_root, tmpl.PROJECT, 'remote', subdir)
            ref = os.path.join(p_root, tmpl.PROJECT, 'ref', subdir)
            cmp = filecmp.dircmp(remote, ref)
            self.assertTrue(len(cmp.diff_files)==0)
            self.assertTrue(len(cmp.right_only)==0)
            self.assertTrue(len(cmp.left_only)==0)

        # for the pickled file we can just read it
        remote = os.path.join(p_root, tmpl.PROJECT, 'remote', 'prepost')
        ref = os.path.join(p_root, tmpl.PROJECT, 'ref', 'prepost')
        with open(os.path.join(remote, 'remote.pkl'), 'rb') as FILE:
            pkl_remote = pickle.load(FILE)
        with open(os.path.join(ref, 'remote.pkl'), 'rb') as FILE:
            pkl_ref = pickle.load(FILE)
        self.assertTrue(pkl_remote == pkl_ref)


if __name__ == "__main__":
    unittest.main()
