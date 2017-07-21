'''
Created on 20. jul. 2017

@author: mmpe
'''
import unittest
from wetb.utils.test_files import move2test_files, get_test_file
import os
from wetb.utils import test_files
import wetb


tfp = os.path.join(os.path.dirname(__file__) + "/test_files/")
class Test_test_files(unittest.TestCase):
    def test_move2test_files(self):
        dst = test_files.default_TestFile_rep_path+ "wetb/utils/tests/test_files/tmp_test_file.txt"
        src = tfp+'tmp_test_file.txt'
        if os.path.isdir(test_files.default_TestFile_rep_path):
            if os.path.isfile(dst):
                os.remove(dst)
            if not os.path.isfile(src):
                with open(src,'w') as fid:
                    fid.write("This is a test file")
            move2test_files(src)
            self.assertTrue(os.path.isfile(dst))

    def test_test_files(self):
        fn = os.path.realpath(os.path.dirname(wetb.__file__) + '/../downloaded_test_files/wetb/utils/tests/test_files/test_file.txt')
        if os.path.isfile(fn):
            os.remove(fn)
        fn1 = get_test_file(tfp+'test_file.txt')
        self.assertTrue(fn1)
        os.remove(fn1)
        fn2 = get_test_file('test_file.txt')
        self.assertEqual(fn2, fn)
        self.assertTrue(os.path.isfile(fn2))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()