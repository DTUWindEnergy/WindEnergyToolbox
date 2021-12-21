'''
Created on 05/11/2015

@author: MMPE
'''

import os
import unittest

import mock

from wetb.hawc2 import ae_file
from wetb.hawc2.ae_file import AEFile
import numpy as np


testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path


class TestAEFile(unittest.TestCase):

    def test_aefile(self):
        ae = AEFile(testfilepath + "NREL_5MW_ae.txt")
        self.assertEqual(ae.thickness(38.950), 21)
        self.assertEqual(ae.chord(38.950), 3.256)
        self.assertEqual(ae.pc_set_nr(38.950), 1)

    def test_aefile_interpolate(self):
        ae = AEFile(testfilepath + "NREL_5MW_ae.txt")
        self.assertEqual(ae.thickness(32), 23.78048780487805)
        self.assertEqual(ae.chord(32), 3.673)
        self.assertEqual(ae.pc_set_nr(32), 1)

    def test_ae_file_main(self):
        def no_print(s):
            pass
        with mock.patch.object(ae_file, "__name__", "__main__"):
            with mock.patch.object(ae_file, "print", no_print):
                getattr(ae_file, 'main')()

    def test_add_set(self):
        ae = AEFile(testfilepath + "NREL_5MW_ae.txt")

        ae.add_set(radius=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   chord=[1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                   thickness=[100.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
                   pc_set_id=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        ae.add_set(radius=[0.0, 0.1],
                   chord=[1.1, 1.0],
                   thickness=[100.0, 100.0],
                   pc_set_id=[1.0, 1.0],
                   set_id=4)

        self.assertEqual(ae.thickness(38.950), 21)
        self.assertEqual(ae.chord(38.950), 3.256)
        self.assertEqual(ae.pc_set_nr(38.950), 1)
        np.testing.assert_array_equal(ae.chord(None, 2), [1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        np.testing.assert_array_equal(ae.ae_sets[2][:2], ae.ae_sets[4])

    def test_str(self):
        ae = AEFile(testfilepath + "NREL_5MW_ae.txt")
        ref = """1 r[m]           Chord[m]    T/C[%]  Set no.
1 19
  0.00000000000000000e+00   3.54199999999999982e+00   1.00000000000000000e+02     1"""

        self.assertEqual(str(ae)[:len(ref)], ref)

    def test_save(self):
        ae = AEFile()

        ae.add_set(radius=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   chord=[1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                   thickness=[100.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
                   pc_set_id=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        fn = testfilepath + "tmp/ae_file.txt"
        ae.save(fn)
        ae2 = AEFile(fn)
        assert str(ae) == str(ae2)

    def test_multiple_sets(self):
        ae = AEFile(testfilepath + 'ae_files/HAWC2_ae.dat')
        self.assertEqual(len(ae.ae_sets), 2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
