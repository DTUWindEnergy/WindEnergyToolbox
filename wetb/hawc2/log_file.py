'''
Created on 18/11/2015

@author: MMPE
'''
import os
from wetb.hawc2.htc_file import HTCFile
from collections import OrderedDict
MISSING = "Log file cannot be found"
PENDING = "Simulation not started yet"
INITIALIZATION = 'Initializing simulation'
GENERATING_TURBULENCE = "Generating turbulence"
SIMULATING = "Simulating"
ABORTED = ""
DONE = "Simulation succeded"
INITIALIZATION_ERROR = "Initialization error"
SIMULATION_ERROR = "Simulation error"
ERROR = "Error"

def is_file_open(filename):
    try:
        os.rename(filename, filename + "_")
        os.rename(filename + "_", filename)
        return False
    except OSError as e:
        if "The process cannot access the file because it is being used by another process" not in str(e):
            raise

        if os.path.isfile(filename + "_"):
            os.remove(filename + "_")
        return True

class LogFile(object):
    _status = (0, MISSING)
    def __init__(self, log_filename, time_stop):
        self.filename = log_filename
        self.time_stop = time_stop
        self.position = 0

    def clear(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'w'):
            pass
        self.position = 0

    def status(self):
        if not os.path.isfile(self.filename):
            self._status = (0, MISSING, [])
        else:
            if self._status[1] == MISSING:
                self._status = (0, PENDING, [])
            with open(self.filename) as fid:
                fid.seek(self.position)
                txt = fid.read()
            self.position += len(txt)
            if self._status[1] == PENDING and self.position > 0:
                self._status = (0, INITIALIZATION, [])
            error_lst = self._status[2]
            if len(txt) > 0:
                if self._status[1] == INITIALIZATION or self._status[1] == GENERATING_TURBULENCE:
                    init_txt, *rest = txt.split("Starting simulation")
                    if "*** ERROR ***" in init_txt:
                        error_lst.extend([l for l in init_txt.strip().split("\n") if "ERROR" in l])
                    if "Turbulence generation starts" in init_txt[init_txt.strip().rfind("\n"):]:
                        self._status = (0, GENERATING_TURBULENCE, error_lst)
                    if rest:
                        txt = rest[0]
                        self._status = (0, SIMULATING, error_lst)
                if self._status[1] == SIMULATING:

                    simulation_txt, *rest = txt.split('Elapsed time')
                    if "*** ERROR ***" in simulation_txt:
                        error_lst.extend([l for l in simulation_txt.strip().split("\n") if "ERROR" in l])
                    i1 = simulation_txt.rfind("Global time")
                    i2 = simulation_txt[:i1].rfind('Global time')
                    time_line = simulation_txt[i1:]
                    try:
                        time = float(time_line[time_line.index('=') + 1:time_line.index('Iter')])
                        self._status = (int(100 * time // self.time_stop), SIMULATING, error_lst)
                    except:
                        self._status = (self._status[0], SIMULATING, error_lst)
                    if rest:
                        self._status = (100, DONE, error_lst)
            #return self._status

        error_lst = self._status[2]
        error_dict = OrderedDict()
        for error in error_lst:
            error_dict[error] = error_dict.get(error, 0) + 1
        error_lst = [("%d x %s" % (v, k), k)[v == 1] for k, v in error_dict.items()]

        return (self._status[0], self._status[1], error_lst)

