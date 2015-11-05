'''
Created on 05/11/2015

@author: MMPE
'''
import unittest
from wetb.hawc2.ae_file import AEFile

testfilepath = 'test_files/'
class Test(unittest.TestCase):


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


        print (ae.thickness(32))
        print (ae.chord(32))
        print (ae.pc_set_nr(32))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
