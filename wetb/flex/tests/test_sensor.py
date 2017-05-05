'''
Created on 9. sep. 2016

@author: mmpe
'''
import os
import unittest
from wetb.flex import read_sensor_info


tfp = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class Test(unittest.TestCase):


    def test_sensor_load(self):
        sensor_info = read_sensor_info(tfp + "test_sensor_info/sensor")
        nr, name, unit, description, _, _ = sensor_info[17]
        self.assertEqual(nr, 18)
        self.assertEqual(name, "Mz coo:")
        self.assertEqual(unit, "kNm")
        self.assertEqual(description, "MomentMz Mbdy:tower nodenr: 1 coo: tower tower base flange")

    def test_sensor_load_name_stop(self):
        sensor_info = read_sensor_info(tfp + "test_sensor_info/sensor"," ")
        nr, name, unit, description, _, _ = sensor_info[17]
        self.assertEqual(nr, 18)
        self.assertEqual(name, "Mz")
        self.assertEqual(unit, "kNm")
        self.assertEqual(description, "coo: MomentMz Mbdy:tower nodenr: 1 coo: tower tower base flange")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_sensor_load']
    unittest.main()
