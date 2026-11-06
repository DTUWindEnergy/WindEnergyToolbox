"""
Created on Fri Oct  9 11:25:38 2026

@author: nicgo
"""

import os
import unittest
from wetb.gtsdf.gtsdf import collect_postproc
from wetb.utils.postprocs import (extreme_loads,
                                  directional_extreme_loads,
                                  directional_equivalent_loads,
                                  directional_markov_matrices)

class Test_postproc(unittest.TestCase):
            
    def test_sensors(self):
        sensors_info_indices = [('Tower base', 15, 16, 17, 18, 19, 20),
                                ('Blade 1 root', 27, 28, 29, 30, 31, 32)]
        sensors_info_names = [('Tower base',
                               'local-tower-node-001-forcevec-x',
                               'local-tower-node-001-forcevec-y',
                               'local-tower-node-001-forcevec-z',
                               'local-tower-node-001-momentvec-x',
                               'local-tower-node-001-momentvec-y',
                               'local-tower-node-001-momentvec-z'),
                              ('Blade 1 root',
                               'forcemomentvec_interp-blade1-c2def-blade1-0.00-Fx',
                               'forcemomentvec_interp-blade1-c2def-blade1-0.00-Fy',
                               'forcemomentvec_interp-blade1-c2def-blade1-0.00-Fz',
                               'forcemomentvec_interp-blade1-c2def-blade1-0.00-Mx',
                               'forcemomentvec_interp-blade1-c2def-blade1-0.00-My',
                               'forcemomentvec_interp-blade1-c2def-blade1-0.00-Mz')]
        sensors_info_directional_indices = [('Tower top shear force', 21, 22),
                                            ('Tower top bending moment', 24, 25),
                                            ('Blade 1 mid-span shear force', 33, 34),
                                            ('Blade 1 mid-span bending moment', 36, 37)]
        sensors_info_directional_names = [('Tower top shear force', 
                                           'local-tower-node-031-forcevec-x',
                                           'local-tower-node-031-forcevec-y'),
                                          ('Tower top bending moment', 
                                           'local-tower-node-031-momentvec-x',
                                           'local-tower-node-031-momentvec-y'),
                                          ('Blade 1 mid-span shear force',
                                           'forcemomentvec_interp-blade1-c2def-blade1-68.93-Fx',
                                           'forcemomentvec_interp-blade1-c2def-blade1-68.93-Fy'),
                                          ('Blade 1 mid-span bending moment',
                                           'forcemomentvec_interp-blade1-c2def-blade1-68.93-Mx',
                                           'forcemomentvec_interp-blade1-c2def-blade1-68.93-My')]

        config_indices = {extreme_loads: {'sensors_info': sensors_info_indices},
                          directional_extreme_loads: {'sensors_info': sensors_info_directional_indices},
                          directional_equivalent_loads: {'sensors_info': sensors_info_directional_indices},
                          directional_markov_matrices: {'sensors_info': sensors_info_directional_indices}}
        
        config_names = {extreme_loads: {'sensors_info': sensors_info_names},
                        directional_extreme_loads: {'sensors_info': sensors_info_directional_names},
                        directional_equivalent_loads: {'sensors_info': sensors_info_directional_names},
                        directional_markov_matrices: {'sensors_info': sensors_info_directional_names}}
        
        folder = os.path.dirname(os.path.abspath(__file__)) + "/test_files/test_sensors"
        postprocs_indices = collect_postproc(folder, config=config_indices, force_recompute=True)
        postprocs_names = collect_postproc(folder, config=config_names, force_recompute=True)
        
        for i in range(len(postprocs_names)):
            # Postprocs using indices or names must be the same for the default case
            assert postprocs_indices[i].isel(filename=0).equals(postprocs_names[i].isel(filename=0))
            
            # For the case with repeated sensors the indices have changed, so the postprocs computed with the default case indices will not match
            assert not postprocs_indices[i].isel(filename=1).equals(postprocs_names[i].isel(filename=1)) # Sensor indices have changed
            
            # The important thing is that postprocs using names are the same for the default case and the case with repeated sensors
            assert postprocs_names[i].isel(filename=0).drop_vars('filename').equals(postprocs_names[i].isel(filename=1).drop_vars('filename'))

if __name__ == "__main__":
    unittest.main()