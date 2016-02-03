from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import open
from builtins import str
from future import standard_library
standard_library.install_aliases()
import os
from wetb.hawc2.htc_file import HTCFile
from wetb.hawc2.log_file import LogFile
from threading import Timer, Thread
import sys
from multiprocessing.process import Process
import psutil
from wetb.utils.process_exec import process, exec_process
import subprocess
import shutil
import json
import glob
from wetb.hawc2 import log_file
import re
import threading



QUEUED = "queued"  #until start
PREPARING = "Copy to host"  # during prepare simulation
INITIALIZING = "Initializing"  #when starting
SIMULATING = "Simulating"  # when logfile.status=simulating
FINISH = "Finish"  # when finish
ERROR = "Error"  # when hawc2 returns error
ABORTED = "Aborted"  # when stopped and logfile.status != Done
CLEANED = "Cleaned"  # after copy back

class Simulation(object):
    is_simulating = False
    _status = QUEUED
    def __init__(self, modelpath, htcfilename, hawc2exe="HAWC2MB.exe"):
        self.modelpath = os.path.abspath(modelpath) + "/"
        self.folder = os.path.dirname(htcfilename)
        if not os.path.isabs(htcfilename):
            htcfilename = os.path.join(modelpath, htcfilename)
        self.htcfilename = htcfilename
        self.filename = os.path.basename(htcfilename)
        self.htcFile = HTCFile(htcfilename)
        self.time_stop = self.htcFile.simulation.time_stop[0]
        self.copy_turbulence = True
        self.simulation_id = os.path.relpath(self.htcfilename, self.modelpath).replace("\\", "_") + "_%d" % id(self)
        self.stdout_filename = "%s.out" % self.simulation_id
        if 'logfile' in self.htcFile.simulation:
            self.log_filename = self.htcFile.simulation.logfile[0]
        else:
            self.log_filename = self.stdout_filename
        if os.path.isabs(self.log_filename):
            self.log_filename = os.path.relpath(self.log_filename, self.modelpath)
        else:
            self.log_filename = os.path.relpath(self.log_filename)
        self.log_filename = self.log_filename.replace("\\", "/")

        self.logFile = LogFile(os.path.join(self.modelpath, self.log_filename), self.time_stop)
        self.logFile.clear()
        self.last_status = self._status
        self.errors = []
        self.thread = Thread(target=self.simulate_distributed)
        self.dist_thread = Thread()
        self.hawc2exe = hawc2exe
        self.simulationThread = SimulationThread(self)
        self.timer = RepeatedTimer(self.update_status)


    def __str__(self):
        return "Simulation(%s)" % self.filename

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status
        self.show_status()

    def update_status(self, *args, **kwargs):
        if self.status in [INITIALIZING, SIMULATING]:
            self.logFile.update_status()

            if self.logFile.status == log_file.SIMULATING:
                self._status = SIMULATING
            if self.logFile.status == log_file.DONE:
                self._status = FINISH

    def show_status(self):
        #print ("log status:", self.logFile.status)
        if self.logFile.status == log_file.SIMULATING:
            if self.last_status != log_file.SIMULATING:
                print ("|" + ("-"*50) + "|" + ("-"*49) + "|")
                sys.stdout.write("|")
            sys.stdout.write("."*(self.logFile.pct - getattr(self, 'last_pct', 0)))
            sys.stdout.flush()
            self.last_pct = self.logFile.pct
        elif self.last_status == log_file.SIMULATING:
            sys.stdout.write("."*(100 - self.last_pct) + "|")
            sys.stdout.flush()
            print ("\n")
        else:
            print (self.logFile.status)
        if self.logFile.status != log_file.SIMULATING:
            if self.logFile.errors:
                print (self.logFile.errors)
        self.last_status = self.logFile.status



    def additional_files(self):
        additional_files_file = os.path.join(self.modelpath, 'additional_files.txt')
        additional_files = {}
        if os.path.isfile(additional_files_file):
            with open(additional_files_file, encoding='utf-8') as fid:
                additional_files = json.load(fid)
        return additional_files

    def add_additional_input_file(self, file):
        additional_files = self.additional_files()
        additional_files['input'] = additional_files.get('input', []) + [file]
        additional_files_file = os.path.join(self.modelpath, 'additional_files.txt')
        with open(additional_files_file, 'w', encoding='utf-8') as fid:
                json.dump(additional_files, fid)

    def prepare_simulation(self):
        self.status = PREPARING

        self.tmp_modelpath = os.path.join(self.modelpath, "tmp_%s/" % self.simulation_id)


        for src in self.htcFile.input_files() + self.htcFile.turbulence_files() + self.additional_files().get('input', []):
            if not os.path.isabs(src):
                src = os.path.join(self.modelpath, src)
            for src_file in glob.glob(src):
                dst = os.path.join(self.tmp_modelpath, os.path.relpath(src_file, self.modelpath))
                # exist_ok does not exist in Python27
                if not os.path.exists(os.path.dirname(dst)):
                    os.makedirs(os.path.dirname(dst))#, exist_ok=True)
                shutil.copy(src_file, dst)
                if not os.path.isfile(dst) or os.stat(dst).st_size != os.stat(src_file).st_size:
                    print ("error copy ", dst)
                else:
                    #print (dst)
                    pass


        self.logFile.filename = os.path.join(self.tmp_modelpath, self.log_filename)
        self.simulationThread.modelpath = self.tmp_modelpath



    def finish_simulation(self):
        lock = threading.Lock()
        with lock:
            if self.status == CLEANED: return
            if self.status != ERROR:
                self.status = CLEANED

        files = self.htcFile.output_files()
        if self.copy_turbulence:
            files.extend(self.htcFile.turbulence_files())
        for dst in files:
            if not os.path.isabs(dst):
                dst = os.path.join(self.modelpath, dst)
            src = os.path.join(self.tmp_modelpath, os.path.relpath(dst, self.modelpath))

            for src_file in glob.glob(src):
                dst_file = os.path.join(self.modelpath, os.path.relpath(src_file, self.tmp_modelpath))
                # exist_ok does not exist in Python27
                if not os.path.exists(os.path.dirname(dst_file)):
                    os.makedirs(os.path.dirname(dst_file))#, exist_ok=True)
                if not os.path.isfile(dst_file) or os.path.getmtime(dst_file) != os.path.getmtime(src_file):
                    shutil.copy(src_file, dst_file)

        self.logFile.filename = os.path.join(self.modelpath, self.log_filename)


        try:
            shutil.rmtree(self.tmp_modelpath)
        except (PermissionError, OSError) as e:
            raise Warning(str(e))



    def simulate(self):
        #starts blocking simulation
        self.is_simulating = True
        self.errors = []
        self.logFile.clear()
        self.status = INITIALIZING

        self.returncode, self.stdout = 1, "Simulation failed"
        self.simulationThread.start()
        self.simulationThread.join()
        self.returncode, self.stdout = self.simulationThread.res
        if self.returncode or 'error' in self.stdout.lower():
            self.errors = (list(set([l for l in self.stdout.split("\n") if 'error' in l.lower()])))
            self.status = ERROR
        self.is_simulating = False
        self.logFile.update_status()
        if self.returncode:
            raise Exception("Simulation error:\n" + "\n".join(self.errors))
        elif self.logFile.status != log_file.DONE  or self.errors or self.logFile.errors:
            raise Warning("Simulation succeded with errors:\nLog status:%s\n" % self.logFile.status + "\n".join(self.logFile.errors))
        else:
            self.status = FINISH

    def simulate_distributed(self):
        self.prepare_simulation()
        self.simulate()
        self.finish_simulation()



    def fix_errors(self):
        def confirm_add_additional_file(folder, file):
            if os.path.isfile(os.path.join(self.modelpath, folder, file)):
                filename = os.path.join(folder, file).replace(os.path.sep, "/")
                if self.get_confirmation("File missing", "'%s' seems to be missing in the temporary working directory. \n\nDo you want to add it to additional_files.txt" % filename):
                    self.add_additional_input_file(filename)
                    self.show_message("'%s' is not added to additional_files.txt.\n\nPlease restart the simulation" % filename)
        for error in self.errors:
            m = re.compile(r".*\*\*\* ERROR \*\*\* File '(.*)' does not exist in the (.*) folder").match(error.strip())
            if m is not None:
                file, folder = m.groups()
                confirm_add_additional_file(folder, file)
                continue
            m = re.compile(r".*\*\*\* ERROR \*\*\* File '(.*)' does not exist in the working directory").match(error.strip())
            if m is not None:
                file = m.groups()[0]
                for root, folder, files in os.walk(self.modelpath):
                    if "__Thread" not in root and file in files:
                        folder = os.path.relpath(root, self.modelpath)
                        confirm_add_additional_file(folder, file)
                continue

    def get_confirmation(self, title, msg):
        return True
    def show_message(self, msg, title="Information"):
        print (msg)

    def start(self):
        """Start non blocking distributed simulation"""
        self.timer.start(1000)
        self.thread.start()

    def stop(self):
        self.timer.stop()
        self.simulationThread.process.kill()
        try:
            self.finish_simulation()
        except:
            pass
        if self.logFile.status not in [log_file.DONE]:
            self.status = ABORTED
        self.update_status()

#class SimulationProcess(Process):
#
#    def __init__(self, modelpath, htcfile, hawc2exe="HAWC2MB.exe"):
#        Process.__init__(self)
#        self.modelpath = modelpath
#        self.htcfile = os.path.abspath(htcfile)
#        self.hawc2exe = hawc2exe
#        self.res = [0, "", "", ""]
#        self.process = process([self.hawc2exe, self.htcfile] , self.modelpath)
#
#
#    def run(self):
#        p = psutil.Process(os.getpid())
#        p.nice = psutil.BELOW_NORMAL_PRIORITY_CLASS
#        exec_process(self.process)


class SimulationThread(Thread):

    def __init__(self, simulation):
        Thread.__init__(self)
        self.sim = simulation
        self.modelpath = self.sim.modelpath
        self.res = [0, "", ""]


    def start(self):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        CREATE_NO_WINDOW = 0x08000000
        modelpath = self.modelpath
        htcfile = os.path.relpath(self.sim.htcFile.filename, self.sim.modelpath)
        hawc2exe = self.sim.hawc2exe
        stdout = self.sim.stdout_filename
        self.process = subprocess.Popen("%s %s 1> %s 2>&1" % (hawc2exe, htcfile, stdout), stdout=None, stderr=None, shell=True, cwd=modelpath, creationflags=CREATE_NO_WINDOW)

        Thread.start(self)


    def run(self):
        p = psutil.Process(os.getpid())
        p.set_nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.process.communicate()
        errorcode = self.process.returncode
        with open(self.modelpath + self.sim.stdout_filename, encoding='utf-8') as fid:
            stdout = fid.read()
        self.res = errorcode, stdout



class RepeatedTimer(object):
    def __init__(self, function, *args, **kwargs):
        self._timer = None
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False


    def _run(self):
        self.is_running = False
        self.start(self.interval)
        self.function(*self.args, **self.kwargs)

    def start(self, interval_ms=None):
        self.interval = interval_ms
        if not self.is_running:
            self._timer = Timer(interval_ms / 1000, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


if __name__ == "__main__":
    sim = Simulation('C:\mmpe\HAWC2\Hawc2_model/', 'htc/error.htc')
    sim.simulate()
