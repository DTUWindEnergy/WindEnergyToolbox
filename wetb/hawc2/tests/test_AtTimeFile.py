'''
Created on 17/07/2014

@author: MMPE
'''
import unittest
from wetb.hawc2.at_time_file import AtTimeFile
import numpy as np
import os


tfp = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class TestAtTimeFile(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path


    def test_doc_examples(self):
        atfile = AtTimeFile(self.testfilepath + "at_time.dat", bladetip_radius=20.501)  # load file
        self.assertEqual(atfile.attribute_names, ['radius_s', 'twist', 'chord'])
        np.testing.assert_array_equal(atfile[:3, 1], [ 0., -0.775186, -2.91652 ])
        np.testing.assert_array_equal(atfile.twist()[:3], [ 0. , -0.775186 , -2.91652 ])
        self.assertAlmostEqual(atfile.twist(10), -5.34743208242399)  # Twist at radius = 10 (interpolated)
        self.assertEqual(atfile.twist(10,curved_length=True), -5.34743208242399)  # Twist at radius = 10 (interpolated)



    def test_at_time_file(self):
        atfile = AtTimeFile(self.testfilepath + "at_time.dat")
        self.assertEqual(atfile.attribute_names, ['radius_s', 'twist', 'chord'])
        self.assertEqual(atfile.radius_s()[9], 6.32780)
        self.assertEqual(atfile.twist()[9], -13.5373)
        self.assertEqual(atfile.chord()[9], 1.54999)


    def test_at_time_file_at_curved_radius(self):
        atfile = AtTimeFile(self.testfilepath + "at_time.dat")
        self.assertEqual(atfile.radius_s(9, curved_length=True), 9)
        self.assertEqual(atfile.twist(9, curved_length=True), -6.635983309665461)
        self.assertEqual(atfile.chord(9, curved_length=True), 1.3888996578373045)

    def test_at_time_file_at_radius(self):
        atfile = AtTimeFile(self.testfilepath + "at_time.dat", bladetip_radius=20.501/2)
        self.assertEqual(atfile.radius_s(9/2), 9)
        self.assertEqual(atfile.twist(9/2), -6.635983309665461)
        self.assertEqual(atfile.chord(9/2), 1.3888996578373045)


    def test_at_time_file_radius(self):
        atfile = AtTimeFile(self.testfilepath + "at_time.dat")
        self.assertEqual(atfile.radius_curved_ac()[12], 10.2505)
        self.assertEqual(atfile.radius_curved_ac(10), 10.2505)
        self.assertEqual(atfile.radius_curved_ac(10.5), 10.2505)

    def test_rotor_name_null(self):
        atfile = AtTimeFile(self.testfilepath + "at_time/test_rotor_name_null.dat", bladetip_radius=20.501)  # load file


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
