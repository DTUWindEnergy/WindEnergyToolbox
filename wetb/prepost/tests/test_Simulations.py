'''
Created on 05/11/2015

@author: MMPE
'''
import unittest
import os
import filecmp
import shutil
from zipfile import ZipFile

import numpy as np
import pandas as pd

from wetb.prepost import dlctemplate as tmpl
from wetb.prepost import Simulations as sim
from wetb.prepost.misc import DictDiff


class Template(unittest.TestCase):
    def setUp(self):
        self.basepath = os.path.dirname(__file__)


class TestErrorLogs(Template):

    def test_loganalysis(self):
        # select a few log cases and do the analysis, save to csv and convert
        # saved result to DataFrame
        pass


class TestGenerateInputs(Template):

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

        # make sure the remote dir is empty so a test does not pass on data
        # generated during a previous cycle
        if os.path.exists(os.path.join(p_root, tmpl.PROJECT, 'remote')):
            shutil.rmtree(os.path.join(p_root, tmpl.PROJECT, 'remote'))

        tmpl.force_dir = tmpl.P_RUN
        tmpl.launch_dlcs_excel('remote', silent=True, runmethod='pbs',
                               pbs_turb=True, zipchunks=True, ppn=17,
                               postpro_node_zipchunks=False,
                               postpro_node=False, update_model_data=True)

        def cmp_dir(dir1, dir2):
            lst1, lst2 = map(os.listdir, (dir1, dir2))
            self.assertEqual(";".join(lst1), ";".join(lst2))
            for f1, f2 in zip(lst1, lst2):
                if f1.endswith(".zip") or f1.endswith(".xlsx"):
                    continue
                if os.path.isdir(os.path.join(dir1, f1)):
                    cmp_dir(os.path.join(dir1, f1), os.path.join(dir2, f2))
                else:
                    try:
                        with open(os.path.join(dir1, f1)) as fid1:
                            l1 = fid1.readlines()
                        with open(os.path.join(dir2, f2)) as fid2:
                            l2 = fid2.readlines()

                        self.assertEqual(len(l1), len(l2))
                        self.assertTrue(all([l1_ == l2_ for l1_, l2_ in zip(l1, l2)]))
                    except:
                        print("=" * 30)
                        print(os.path.join(dir1, f1))
                        print(os.path.join(dir2, f2))
                        print(dir1[[d1 != d2 for d1, d2 in zip(dir1, dir2)].index(True):])
                        print(f1)
                        for i in range(len(l1)):
                            if l1[i] != l2[i]:
                                print("%03d, rem: %s" % (i, l1[i].strip()))
                                print("%03d, ref: %s" % (i, l2[i].strip()))
                                print()
                        raise

        # we can not git check-in empty dirs so we can not compare the complete
        # directory structure withouth manually creating the empty dirs here
        for subdir in ['control', 'data', 'htc', 'pbs_in', 'pbs_in_turb',
                       'htc/_master', 'htc/dlc01_demos', 'pbs_in/dlc01_demos',
                       'zip-chunks-jess']:
            remote = os.path.join(p_root, tmpl.PROJECT, 'remote', subdir)
            ref = os.path.join(p_root, tmpl.PROJECT, 'ref', subdir)
            # the zipfiles are taken care of separately
            ignore = ['remote_chnk_00000.zip']
            cmp = filecmp.dircmp(remote, ref, ignore=ignore)
            cmp_dir(remote, ref)
            self.assertEqual(len(cmp.diff_files), 0,
                             "{} {}".format(subdir, cmp.diff_files))
            self.assertEqual(len(cmp.right_only), 0,
                             "{} {}".format(subdir, cmp.right_only))
            self.assertEqual(len(cmp.left_only), 0,
                             "{} {}".format(subdir, cmp.left_only))

        # compare the zip files
        for fname in ['demo_dlc_remote.zip',
                      'zip-chunks-jess/remote_chnk_00000.zip']:
            remote = os.path.join(p_root, tmpl.PROJECT, 'remote', fname)
            ref = os.path.join(p_root, tmpl.PROJECT, 'ref', fname)

            with ZipFile(remote) as zrem, ZipFile(ref) as zref:
                self.assertEqual(len(zrem.infolist()), len(zref.infolist()))
                frem = {f.filename:f.file_size for f in zrem.infolist()}
                fref = {f.filename:f.file_size for f in zref.infolist()}
                dd = DictDiff(frem, fref)
                self.assertEqual(len(dd.added()), 0,
                                 "{} {}".format(fname, dd.added()))
                self.assertEqual(len(dd.removed()), 0,
                                 "{} {}".format(fname, dd.removed()))
                self.assertEqual(len(dd.changed()), 0,
                                 "{} {}".format(fname, dd.changed()))

        # for the pickled file we can just read it
        remote = os.path.join(p_root, tmpl.PROJECT, 'remote', 'prepost')
        ref = os.path.join(p_root, tmpl.PROJECT, 'ref', 'prepost')
        cmp = filecmp.cmp(os.path.join(remote, 'remote_tags.txt'),
                          os.path.join(ref, 'remote_tags.txt'), shallow=False)
        self.assertTrue(cmp)
#        with open(os.path.join(remote, 'remote.pkl'), 'rb') as FILE:
#            pkl_remote = pickle.load(FILE)
#        with open(os.path.join(ref, 'remote.pkl'), 'rb') as FILE:
#            pkl_ref = pickle.load(FILE)
#        self.assertTrue(pkl_remote == pkl_ref)


class TestFatigueLifetime(Template):

    def test_leq_life(self):
        """Verify if prepost.Simulation.Cases.fatigue_lifetime() returns
        the expected life time equivalent load.
        """
        # ---------------------------------------------------------------------
        # very simple case
        cases = {'case1':{'[post_dir]':'no-path', '[sim_id]':'A0'},
                 'case2':{'[post_dir]':'no-path', '[sim_id]':'A0'}}
        cc = sim.Cases(cases)

        fh_list = [('case1', 10/3600), ('case2', 20/3600)]
        dfs = pd.DataFrame({'m=1.0' : [2, 3],
                            'channel' : ['channel1', 'channel1'],
                            '[case_id]' : ['case1', 'case2']})
        neq_life = 1.0
        df_Leq = cc.fatigue_lifetime(dfs, neq_life, fh_lst=fh_list,
                                     save=False, update=False, csv=False,
                                     xlsx=False, silent=False)
        np.testing.assert_allclose(df_Leq['m=1.0'].values, 2*10 + 3*20)
        self.assertTrue(df_Leq['channel'].values[0]=='channel1')

        # ---------------------------------------------------------------------
        # slightly more complicated
        neq_life = 3.0
        df_Leq = cc.fatigue_lifetime(dfs, neq_life, fh_lst=fh_list,
                                     save=False, update=False, csv=False,
                                     xlsx=False, silent=False)
        np.testing.assert_allclose(df_Leq['m=1.0'].values,
                                   (2*10 + 3*20)/neq_life)

        # ---------------------------------------------------------------------
        # a bit more complex and also test the sorting of fh_lst and dfs
        cases = {'case1':{'[post_dir]':'no-path', '[sim_id]':'A0'},
                 'case2':{'[post_dir]':'no-path', '[sim_id]':'A0'},
                 'case3':{'[post_dir]':'no-path', '[sim_id]':'A0'},
                 'case4':{'[post_dir]':'no-path', '[sim_id]':'A0'}}
        cc = sim.Cases(cases)

        fh_list = [('case3', 10/3600), ('case2', 20/3600),
                   ('case1', 50/3600), ('case4', 40/3600)]
        dfs = pd.DataFrame({'m=3.0' : [2, 3, 4, 5],
                            'channel' : ['channel1']*4,
                            '[case_id]' : ['case4', 'case2', 'case3', 'case1']})
        neq_life = 5.0
        df_Leq = cc.fatigue_lifetime(dfs, neq_life, fh_lst=fh_list,
                                     save=False, update=False, csv=False,
                                     xlsx=False, silent=False)
        expected = ((2*2*2*40 + 3*3*3*20 + 4*4*4*10 + 5*5*5*50)/5)**(1/3)
        np.testing.assert_allclose(df_Leq['m=3.0'].values, expected)

        # ---------------------------------------------------------------------
        # more cases and with sorting
        base = {'[post_dir]':'no-path', '[sim_id]':'A0'}
        cases = {'case%i' % k : base for k in range(50)}
        cc = sim.Cases(cases)
        # reverse the order of how they appear in dfs and fh_lst
        fh_list = [('case%i' % k, k*10/3600) for k in range(49,-1,-1)]
        dfs = pd.DataFrame({'m=5.2' : np.arange(1,51,1),
                            'channel' : ['channel1']*50,
                            '[case_id]' : ['case%i' % k for k in range(50)]})
        df_Leq = cc.fatigue_lifetime(dfs, neq_life, fh_lst=fh_list,
                                     save=False, update=False, csv=False,
                                     xlsx=False, silent=False)
        expected = np.sum(np.power(np.arange(1,51,1), 5.2)*np.arange(0,50,1)*10)
        expected = np.power(expected/neq_life, 1/5.2)
        np.testing.assert_allclose(df_Leq['m=5.2'].values, expected)


if __name__ == "__main__":
    unittest.main()
