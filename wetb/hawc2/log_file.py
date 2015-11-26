'''
Created on 18/11/2015

@author: MMPE
'''
import os
from wetb.hawc2.htc_file import HTCFile
from collections import OrderedDict
import time
import math
UNKNOWN = "Unknown"
MISSING = "Log file cannot be found"
PENDING = "Simulation not started yet"
INITIALIZATION = 'Initializing simulation'
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
    def __init__(self, log_filename, time_stop):
        self.filename = log_filename
        self.time_stop = time_stop
        self.reset()
        self.update_status()

    @staticmethod
    def from_htcfile(htcfile, modelpath):
        logfilename = htcfile.simulation.logfile[0]
        if not os.path.isabs(logfilename):
            logfilename = os.path.join(modelpath, logfilename)
        return LogFile(logfilename, htcfile.simulation.time_stop[0])

    def reset(self):
        self.position = 0
        self.lastline = ""
        self.status = UNKNOWN
        self.pct = 0
        self.errors = []
        self.info = []
        self.start_time = None
        self.current_time = 0
        self.remaining_time = None


    def clear(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'w'):
            pass
        self.reset()

    def extract_time(self, time_line):
        time_line = time_line.strip()
        if 'Starting simulation' == time_line:
            return 0
        if time_line == "":
            return self.current_time
        try:
            return float(time_line[time_line.index('=') + 1:time_line.index('Iter')])
        except:
            print ("#" + time_line + "#")
            pass

    def update_status(self):
        if not os.path.isfile(self.filename):
            self.status = MISSING
        else:
            if self.status == UNKNOWN or self.status == MISSING:
                self.status = PENDING
            with open(self.filename, 'rb') as fid:
                fid.seek(self.position)
                txt = fid.read()
            self.position += len(txt)
            txt = txt.decode(encoding='utf_8', errors='strict')
            if self.status == PENDING and self.position > 0:
                self.status = INITIALIZATION

            if len(txt) > 0:
                self.lastline = (txt.strip()[max(0, txt.strip().rfind("\n")):]).strip()
                if self.status == INITIALIZATION:
                    init_txt, *rest = txt.split("Starting simulation")
                    if "*** ERROR ***" in init_txt:
                        self.errors.extend([l.strip() for l in init_txt.strip().split("\n") if "error" in l.lower()])
                    if rest:
                        txt = rest[0]
                        self.status = SIMULATING
                        if not 'Elapsed time' in self.lastline:
                            self.start_time = (self.extract_time(self.lastline), time.time())

                if self.status == SIMULATING:
                    simulation_txt, *rest = txt.split('Elapsed time')
                    if "*** ERROR ***" in simulation_txt:
                        self.errors.extend([l.strip() for l in simulation_txt.strip().split("\n") if "error" in l.lower()])
                    i1 = simulation_txt.rfind("Global time")
                    i2 = simulation_txt[:i1].rfind('Global time')
                    self.current_time = self.extract_time(simulation_txt[i1:])
                    self.pct = int(100 * self.current_time // self.time_stop)
                    if self.current_time is not None and self.start_time is not None and (self.current_time - self.start_time[0]) > 0:
                        self.remaining_time = (time.time() - self.start_time[1]) / (self.current_time - self.start_time[0]) * (self.time_stop - self.current_time)
                    if rest:
                        self.status = DONE
                        self.pct = 100
                        self.elapsed_time = float(rest[0].replace(":", "").strip())

    def error_str(self):
        error_dict = OrderedDict()
        for error in self.errors:
            error_dict[error] = error_dict.get(error, 0) + 1
        return "\n".join([("%d x %s" % (v, k), k)[v == 1] for k, v in error_dict.items()])


    def remaining_time_str(self):
        if self.remaining_time:
            if self.remaining_time < 3600:
                m, s = divmod(self.remaining_time, 60)
                return "%02d:%02d" % (m, math.ceil(s))
            else:
                h, ms = divmod(self.remaining_time, 3600)
                m, s = divmod(ms, 60)
                return "%d:%02d:%02d" % (h, m, math.ceil(s))
        else:
            return "--:--"

    def add_HAWC2_errors(self, errors):
        if errors:
            self.status = ERROR
            self.errors.extend(errors)





