'''
Created on 18/11/2015

@author: MMPE
'''
import unittest
from wetb.hawc2.log_file import LogFile, is_file_open, INITIALIZATION_ERROR, \
    INITIALIZATION, SIMULATING, DONE, SIMULATION_ERROR, GENERATING_TURBULENCE, \
    PENDING
import time
from wetb.hawc2 import log_file
import threading
import os

def simulate(file, wait):
    with open(file, 'r') as fin:
        lines = fin.readlines()
    file = file + "_"
    with open(file, 'w'):
        pass
    time.sleep(.1)
    for l in lines:
        with open(file, 'a+') as fout:
            fout.write(l)
        if "Turbulence generation starts" in l or "Log file output" in l:
            time.sleep(0.2)
        time.sleep(wait)

class Test(unittest.TestCase):


    def test_missing_logfile(self):
        f = 'test_files/logfiles/missing.log'
        logfile = LogFile(f, 200)
        status = logfile.status()
        self.assertEqual(status[0], 0)
        self.assertEqual(status[1], log_file.MISSING)


    def test_is_file_open(self):
        f = 'test_files/logfiles/test.log'
        with open(f, 'a+'):
            self.assertTrue(is_file_open(f))
        with open(f, 'r'):
            self.assertTrue(is_file_open(f))
        self.assertFalse(is_file_open(f))

    def test_simulation_init_error(self):
        f = 'test_files/logfiles/init_error.log'
        logfile = LogFile(f, 2)
        code, txt, err = logfile.status()
        self.assertEqual(code, 100)
        self.assertEqual(txt, DONE)
        self.assertEqual(err, [' *** ERROR *** No line termination in command line            8'])

    def test_init(self):
        f = 'test_files/logfiles/init.log'
        logfile = LogFile(f, 200)
        code, txt, err = logfile.status()
        self.assertEqual(code, 0)
        self.assertEqual(txt, INITIALIZATION)
        self.assertEqual(err, [])

    def test_turbulence_generation(self):
        f = 'test_files/logfiles/turbulence_generation.log'
        logfile = LogFile(f, 200)
        code, txt, err = logfile.status()
        self.assertEqual(code, 0)
        self.assertEqual(txt, GENERATING_TURBULENCE)
        self.assertEqual(err, [])

    def test_simulation(self):
        f = 'test_files/logfiles/simulating.log'
        logfile = LogFile(f, 2)
        code, txt, err = logfile.status()
        self.assertEqual(code, 25)
        self.assertEqual(txt, SIMULATING)
        self.assertEqual(err, [])


    def test_finish(self):
        f = 'test_files/logfiles/finish.log'
        logfile = LogFile(f, 200)
        code, txt, err = logfile.status()
        self.assertEqual(code, 100)
        self.assertEqual(txt, DONE)
        self.assertEqual(err, [])


    def test_simulation_error(self):
        f = 'test_files/logfiles/simulation_error.log'
        logfile = LogFile(f, 2)
        code, txt, err = logfile.status()
        self.assertEqual(code, 100)
        self.assertEqual(txt, DONE)
        self.assertEqual(err, [' *** ERROR *** Error opening out .dat file'])

    def test_simulation_error2(self):
        f = 'test_files/logfiles/simulation_error2.log'
        logfile = LogFile(f, 2)
        code, txt, err = logfile.status()
        self.assertEqual(code, 100)
        self.assertEqual(txt, DONE)
        self.assertEqual(err, ['30 x  *** ERROR *** Out of limits in user defined shear field - limit value used'])


    def check(self, logfilename, phases, end_status):
        logfile = LogFile(logfilename + "_", 2)
        if os.path.isfile(logfile.filename):
            os.remove(logfile.filename)
        status = logfile.status()
        t = threading.Thread(target=simulate, args=(logfilename, 0.0001))
        t.start()
        while status[0] >= 0 and status[1] != DONE:
            new_status = logfile.status()
            if new_status[1] != status[1] or new_status[0] != status[0]:
                status = new_status
                #print(status)
            if status[1] in phases:
                phases.remove(status[1])
            time.sleep(0.01)
        code, txt, err = logfile.status()
        self.assertEqual(code, end_status[0])
        self.assertEqual(txt, end_status[1])
        self.assertEqual(err, end_status[2])
        self.assertFalse(phases)
        t.join()
        os.remove(logfile.filename)


    def test_realtime_test(self):
        self.check('test_files/logfiles/finish.log',
                   phases=[PENDING, INITIALIZATION, SIMULATING, DONE],
                   end_status=(100, DONE, []))

    def test_realtime_test2(self):
        self.check('test_files/logfiles/init_error.log',
           phases=[PENDING, INITIALIZATION, SIMULATING, DONE],
           end_status=(100, DONE, [' *** ERROR *** No line termination in command line            8']))

    def test_realtime_test_simulation_error(self):
        self.check('test_files/logfiles/simulation_error.log',
                   [PENDING, INITIALIZATION, SIMULATING, DONE],
                   (100, DONE, [' *** ERROR *** Error opening out .dat file']))

    def test_realtime_test_turbulence(self):
        self.check('test_files/logfiles/finish_turbulencegeneration.log',
                   phases=[PENDING, INITIALIZATION, GENERATING_TURBULENCE, SIMULATING, DONE],
                   end_status=(100, DONE, []))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_logfile']
    unittest.main()
