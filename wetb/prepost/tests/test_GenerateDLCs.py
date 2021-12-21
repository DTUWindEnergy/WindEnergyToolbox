import unittest
import os
#import shutil

import numpy as np
import pandas as pd

from wetb.prepost.GenerateDLCs import GenerateDLCCases


class Template(unittest.TestCase):
    def setUp(self):
        self.basepath = os.path.dirname(__file__)


class TestGenerateDLCCases(Template):

    def test_dlcs(self):
        # manually configure paths, HAWC2 model root path is then constructed as
        # p_root_remote/PROJECT/sim_id, and p_root_local/PROJECT/sim_id
        # adopt accordingly when you have configured your directories differently
        p_root = os.path.join(self.basepath, 'data/demo_gendlc/')

        # project name, sim_id, master file name
        dlc_master = os.path.join(p_root, 'DLCs.xlsx')
        dlc_folder = os.path.join(p_root, 'DLCs')
        dlc_gen12 = os.path.join(dlc_folder, 'DLC12.xlsx')
        dlc_gen13 = os.path.join(dlc_folder, 'DLC13.xlsx')

        DLB = GenerateDLCCases()
        DLB.execute(filename=dlc_master, folder=dlc_folder)

        df12 = pd.read_excel(dlc_gen12)
        # df12.to_csv('data/demo_gendlc/ref/DLC12.csv', index=False)
        df12_ref = pd.read_csv(os.path.join(p_root, 'ref/DLC12.csv'))
        # df12_ref2 = pd.read_excel(p2)[df12.columns]
        pd.testing.assert_frame_equal(df12, df12_ref)

        # df2 = df[['[Case id.]', '[wdir]', '[wsp]', '[seed]', '[wave_seed]']]
        self.assertEqual(df12['[ReferenceWindSpeed]'].unique(), np.array([44]))
        self.assertEqual(df12['[t0]'].unique(), np.array([100]))
        self.assertEqual(len(df12['[Case id.]'].unique()), 2*3*3*2)
        self.assertEqual(df12['[Case id.]'].values[0],
                         'DLC12_wsp04_wdir010_sa1001_sw0101')

        df13 = pd.read_excel(dlc_gen13)
        # df13.to_csv('data/demo_gendlc/ref/DLC13.csv', index=False)
        df13_ref = pd.read_csv(os.path.join(p_root, 'ref/DLC13.csv'))
        pd.testing.assert_frame_equal(df13, df13_ref)


if __name__ == "__main__":
    unittest.main()
