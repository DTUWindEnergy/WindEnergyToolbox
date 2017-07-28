'''
Created on 02/10/2015

@author: MMPE
'''
import unittest

import numpy as np
from wetb.wind.turbulence.constraint_file import ConstraintFile
from wetb.utils.test_files import get_test_file
class TestConstraintFile(unittest.TestCase):


    def testConstraintFile(self):
        time = [0, 1, 2]
        u = [5, 7, 9]
        v = [1.5, 1, .5]
        w = [3, 1, 5]
        tu = np.array([time, u]).T
        tuvw = np.array([time, u, v, w]).T
        glpos = (0, 0, -85)
        constraint_file = ConstraintFile(center_gl_xyz=(-5, 0, -90), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.add_constraints(glpos, tu)
        np.testing.assert_array_equal(constraint_file.constraints['u'][1], (6, 4, 4, 0))
        constraint_file = ConstraintFile(center_gl_xyz=(-5, 0, -90), box_transport_speed=5, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.add_constraints(glpos, tuvw)
        np.testing.assert_array_equal(constraint_file.constraints['u'][2], (6, 4, 4, 2))
        np.testing.assert_array_equal(constraint_file.constraints['v'][2], (6, 4, 4, -.5))
        np.testing.assert_array_equal(constraint_file.constraints['w'][2], (6, 4, 4, 2))
        constraint_file = ConstraintFile(center_gl_xyz=(-5, 0, -90), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.add_constraints((-10, -4, -95), tu)
        np.testing.assert_array_equal(constraint_file.constraints['u'][1], (8, 5, 5, 0))
        constraint_file = ConstraintFile(center_gl_xyz=(-5, 4, -90), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.add_constraints((-10, 0, -95), tu)
        np.testing.assert_array_equal(constraint_file.constraints['u'][1], (8, 5, 5, 0))
        #constraint_file.save("./test_files", "constraints_test",'')

    def testConstraintFile_Mxyz(self):
        time = [0]
        u = [5]
        tu = np.array([time, u]).T

        glpos = (0, 0, -85)
        constraint_file = ConstraintFile(center_gl_xyz=(0, 0, 0), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.add_constraints([0,0,0], tu, subtract_mean=False)
        np.testing.assert_array_equal(constraint_file.constraints['u'][-1], (1, 5, 5, 5))
        constraint_file.add_constraints([5,0,0], tu, subtract_mean=False)
        np.testing.assert_array_equal(constraint_file.constraints['u'][-1], (1, 4, 5, 5))
        constraint_file.add_constraints([0,0,5], tu, subtract_mean=False)
        np.testing.assert_array_equal(constraint_file.constraints['u'][-1], (1, 5, 4, 5))
        constraint_file.add_constraints([0,-10,0], tu, subtract_mean=False)
        np.testing.assert_array_equal(constraint_file.constraints['u'][-1], (6, 5, 5, 5))
        
    def testConstraintFile_center_gl_xyz(self):
        time = [0]
        u = [5]
        tu = np.array([time, u]).T

        constraint_file = ConstraintFile(center_gl_xyz=(0, 0, 0), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.add_constraints([0,0,0], tu, subtract_mean=False)
        np.testing.assert_array_equal(constraint_file.constraints['u'][-1], (1, 5, 5, 5))
        
        constraint_file = ConstraintFile(center_gl_xyz=(0, 0, -5), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.add_constraints([0,0,0], tu, subtract_mean=False)
        np.testing.assert_array_equal(constraint_file.constraints['u'][-1], (1, 5, 4, 5))
        
        constraint_file = ConstraintFile(center_gl_xyz=(-5, 0, 0), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.add_constraints([0,0,0], tu, subtract_mean=False)
        np.testing.assert_array_equal(constraint_file.constraints['u'][-1], (1, 4, 5, 5))
        
        constraint_file = ConstraintFile(center_gl_xyz=(0, 10, 0), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.add_constraints([0,0,0], tu, subtract_mean=False)
        np.testing.assert_array_equal(constraint_file.constraints['u'][-1], (6, 5, 5, 5))
        

    def test_outputbox_errors(self):
        time = [0, 1, 2]
        u = [5, 7, 9]
        v = [1.5, 1, .5]
        w = [3, 1, 5]
        tu = np.array([time, u]).T
        tuvw = np.array([time, u, v, w]).T
        glpos = (0, 0, -85)
        constraint_file = ConstraintFile(center_gl_xyz=(0, 0, -85), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        self.assertRaisesRegex(ValueError, "At time, t=0, global position \(0,0,-44\)", constraint_file.add_constraints, (0, 0, -85 + 35 + 6), tu)
        self.assertRaisesRegex(ValueError, "At time, t=0, global position \(0,0,-126\)", constraint_file.add_constraints, (0, 0, -85 - 35 - 6), tu)
        self.assertRaisesRegex(ValueError, "At time, t=0, global position \(-41,0,-85\)", constraint_file.add_constraints, (-35 - 6, 0, -85), tu)
        self.assertRaisesRegex(ValueError, "At time, t=0, global position \(41,0,-85\)", constraint_file.add_constraints, (35 + 6, 0, -85), tu)
        self.assertRaisesRegex(ValueError, "At time, t=0, global position \(0,2,-85\)", constraint_file.add_constraints, (0, 2, -85), tu)
        self.assertRaisesRegex(ValueError, "At time, t=2, global position \(0,-13,-85\)", constraint_file.add_constraints, (0, -13, -85), tu)

    def test_hawc2_cmd(self):
        constraint_file = ConstraintFile(center_gl_xyz=(0, 0, -85), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        mann = constraint_file.hawc2_mann_section("test",1)
        self.assertEqual(mann.box_dim_u.values, [16,1.875])
        
    def test_load(self):
        constraint_file = ConstraintFile(center_gl_xyz=(-5, 4, -90), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.load(get_test_file('constraints_test.con'))
        with open(get_test_file('constraints_test.con')) as fid:
            self.assertEqual(str(constraint_file), fid.read())
        
    def test_time_series(self):
        constraint_file = ConstraintFile(center_gl_xyz=(-5, 4, -90), box_transport_speed=10, no_grid_points=(16, 8, 8), box_size=(30, 70, 70))
        constraint_file.load(get_test_file('constraints_test.con'))
        time,uvw = constraint_file.time_series(-4)
        u,v,w = uvw.T
        m = ~np.isnan(u)
        np.testing.assert_array_equal(time[m],[0,1,2])
        np.testing.assert_array_equal(u[m],[-2,0,2])
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testConstraintFile']
    unittest.main()
