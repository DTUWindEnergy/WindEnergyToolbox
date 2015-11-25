import os
from wetb.hawc2.htc_file import HTCFile
from wetb.hawc2.log_file import LogFile, SIMULATING
import threading
from wetb.hawc2 import dummy_simulator
from threading import Timer, Thread
import sys
import multiprocessing
from multiprocessing.process import Process
import psutil
from wetb.functions.process_exec import pexec, process, exec_process
import time
import subprocess
import shutil
import json
import glob





class Simulation(object):
    is_simulating = False
    def __init__(self, modelpath, htcfilename, hawc2exe="HAWC2MB.exe"):
        self.modelpath = modelpath
        self.folder = os.path.dirname(htcfilename)
        if not os.path.isabs(htcfilename):
            htcfilename = os.path.join(modelpath, htcfilename)

        self.filename = os.path.basename(htcfilename)
        self.htcFile = HTCFile(htcfilename)
        self.time_stop = self.htcFile.simulation.time_stop[0]

        self.log_filename = self.htcFile.simulation.logfile[0]
        if os.path.isabs(self.log_filename):
            self.log_filename = os.path.relpath(self.log_filename, self.modelpath)

        self.logFile = LogFile(os.path.join(self.modelpath, self.log_filename), self.time_stop)
        self.logFile.clear()
        self.last_status = (0, "Pending", [])
        self.thread = Thread(target=self.simulate)
        self.simulationThread = SimulationThread(self.modelpath, self.htcFile.filename, hawc2exe)
        self.timer = RepeatedTimer(1, self.update_status)


    def update_status(self, *args, **kwargs):
        status = self.logFile.status()
        if status[1] == SIMULATING:
            if self.last_status[1] != SIMULATING:
                print ("|" + ("-"*50) + "|" + ("-"*49) + "|")
                sys.stdout.write("|")
            #print (status)
            sys.stdout.write("."*(status[0] - self.last_status[0]))
            sys.stdout.flush()
        elif self.last_status[1] == SIMULATING and status[1] != SIMULATING:
            sys.stdout.write("."*(status[0] - self.last_status[0]) + "|")
            sys.stdout.flush()
            print ("\n")
        if status[1] != SIMULATING:

            if status[2]:
                print (status[1:])
            else:
                print (status[1])
        self.last_status = status


    def prepare_simulation(self, id):
        self.tmp_modelpath = os.path.join(self.modelpath, id + "/")
        additional_files_file = os.path.join(self.modelpath, 'additional_files.txt')
        additional_files = []
        if os.path.isfile(additional_files_file):
            with open(additional_files_file) as fid:
                additional_files = json.load(fid).get('input', [])

        for src in self.htcFile.input_files() + self.htcFile.turbulence_files() + additional_files:
            if not os.path.isabs(src):
                src = os.path.join(self.modelpath, src)
            for src_file in glob.glob(src):
                dst = os.path.join(self.tmp_modelpath, os.path.relpath(src_file, self.modelpath))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src_file, dst)


        self.logFile.filename = os.path.join(self.tmp_modelpath, self.log_filename)
        self.simulationThread.modelpath = self.tmp_modelpath



    def finish_simulation(self, copy_turbulence):
        files = self.htcFile.output_files()
        if copy_turbulence:
            files.extend(self.htcFile.turbulence_files())
        for dst in files:
            if not os.path.isabs(dst):
                dst = os.path.join(self.modelpath, dst)

            src = os.path.join(self.tmp_modelpath, os.path.relpath(dst, self.modelpath))
            if os.path.isfile(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.isfile(dst) or os.path.getmtime(dst) != os.path.getmtime(src):
                    shutil.copy(src, dst)

        self.logFile.filename = os.path.join(self.modelpath, self.log_filename)



        shutil.rmtree(self.tmp_modelpath)



    def simulate(self):

        self.is_simulating = True
        self.timer.start()
        self.simulationThread.start()
        self.simulationThread.join()

        errorcode, stdout, stderr, cmd = self.simulationThread.res
        if errorcode:
            print (errorcode)
            print (stdout)
            print (stderr)
            print (cmd)
        self.timer.stop()
        self.is_simulating = False
        self.update_status()


    def start(self):

        self.thread.start()

    def terminate(self):
        self.timer.stop()
        self.simulationThread.process.kill()
        self.simulationThread.join()

class SimulationProcess(Process):

    def __init__(self, modelpath, htcfile, hawc2exe="HAWC2MB.exe"):
        Process.__init__(self)
        self.modelpath = modelpath
        self.htcfile = os.path.abspath(htcfile)
        self.hawc2exe = hawc2exe
        self.res = [0, "", "", ""]
        self.process = process([self.hawc2exe, self.htcfile] , self.modelpath)


    def run(self):
        p = psutil.Process(os.getpid())
        p.nice = psutil.BELOW_NORMAL_PRIORITY_CLASS
        exec_process(self.process)


class SimulationThread(Thread):

    def __init__(self, modelpath, htcfile, hawc2exe):
        Thread.__init__(self)
        self.modelpath = modelpath
        self.htcfile = os.path.relpath(htcfile, self.modelpath)
        self.hawc2exe = hawc2exe
        self.res = [0, "", "", ""]


    def start(self):
        print (self.getName(), self.htcfile, self.modelpath, self.hawc2exe)
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        #self.hawc2exe = r'C:\mmpe\programming\python\MMPE\programs\getcwd\getcwd_dist\exe.win-amd64-3.3/getcwd.exe'
        CREATE_NO_WINDOW = 0x08000000
        self.process = subprocess.Popen([self.hawc2exe, self.htcfile], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, cwd=self.modelpath, creationflags=CREATE_NO_WINDOW)

        Thread.start(self)


    def run(self):
        p = psutil.Process(os.getpid())
        p.nice = psutil.BELOW_NORMAL_PRIORITY_CLASS
        self.res = exec_process(self.process)
        
        
class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False


    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


if __name__ == "__main__":
    sim = Simulation('C:\mmpe\HAWC2\Hawc2_model/', 'htc/error.htc')
    sim.simulate()
