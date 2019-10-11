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
import numpy as np
from wetb.hawc2.Hawc2io import ReadHawc2, ModifyHawc2
import os


testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/hawc2io/')  # test file path

class TestHAWC2IO(unittest.TestCase):


    def test_doc_example(self):
        # if called with ReadOnly = 1 as
        file = ReadHawc2(testfilepath + "Hawc2bin", ReadOnly=1)
        # no channels a stored in memory, otherwise read channels are stored for reuse

        # channels are called by a list
#        file([0,2,1,1])  => channels 1,3,2,2
        self.assertEqual(file([0, 2, 1, 1]).shape, (800, 4))

#        # if empty all channels are returned
#        file()  => all channels as 1,2,3,...
        self.assertEqual(file().shape, (800, 28))
#        file.t => time vector
        np.testing.assert_array_almost_equal(file.t, file([0])[:, 0])


    def test_read_binary_file(self):
        file = ReadHawc2(testfilepath + "Hawc2bin", ReadOnly=1)
        self.assertAlmostEqual(file()[0, 0], 0.025)
        self.assertEqual(file()[799, 0], 20)
        self.assertAlmostEqual(file()[1, 0], .05)

    def test_read_ascii_file(self):
        file = ReadHawc2(testfilepath + "Hawc2ascii", ReadOnly=1)
        self.assertAlmostEqual(file()[0, 0], 0.025)
        self.assertEqual(file()[799, 0], 20)
        self.assertAlmostEqual(file()[1, 0], .05)
        
    def test_writing_with_no_modifications(self):
        '''
        Read a HAWC2 binary result file, rewrite it, then read it again.
        Check that both the original and rewritten result files return the
        same results.
        '''
        ref_fn = testfilepath + "Hawc2bin"
        out_fn = testfilepath + "output_results"
        
        read_obj = ModifyHawc2(ref_fn)
        read_obj.WriteBinary(out_fn)
        
        out_data = ReadHawc2(out_fn)()
        self.assertEqual(((out_data - read_obj.data)**2).sum(), 0)
    
    def test_write_new_channel_and_read(self):
        '''
        Read a HAWC2 Binary result file, add a row, write it, and read it again.
        '''
        
        ref_fn = testfilepath + "Hawc2bin"
        out_fn = testfilepath + "output_results2"
        
        write_obj = ModifyHawc2(ref_fn)
        write_obj.add_channel(np.ones(write_obj.NrSc), 'the_name', 'the_unit', 'the_description')
        write_obj.WriteBinary(out_fn)
        
        read_obj = ReadHawc2(out_fn)
        out_data = read_obj()
        
        self.assertEqual( read_obj.NrCh, write_obj.NrCh)
        self.assertEqual(((write_obj.data - out_data)**2).sum(), 0)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
