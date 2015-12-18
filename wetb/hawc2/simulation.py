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



QUEUED = "queued"  #until start
INITIALIZING = "initializing"  #when starting
SIMULATING = "SIMULATING"  # when logfile.status=simulating
FINISH = "finish"  # when finish
ERROR = "error"  # when hawc2 returns error
ABORTED = "aborted"  # when stopped and logfile.status != Done
CLEANED = "cleaned"  # after copy back

class Simulation(object):
    is_simulating = False
    def __init__(self, modelpath, htcfilename, hawc2exe="HAWC2MB.exe"):
        self.modelpath = modelpath
        self.folder = os.path.dirname(htcfilename)
        if not os.path.isabs(htcfilename):
            htcfilename = os.path.join(modelpath, htcfilename)
        self.htcfilename = htcfilename
        self.filename = os.path.basename(htcfilename)
        self.htcFile = HTCFile(htcfilename)
        self.time_stop = self.htcFile.simulation.time_stop[0]
        self.copy_turbulence = True
        self.log_filename = self.htcFile.simulation.logfile[0]
        if os.path.isabs(self.log_filename):
            self.log_filename = os.path.relpath(self.log_filename, self.modelpath)

        self.logFile = LogFile(os.path.join(self.modelpath, self.log_filename), self.time_stop)
        self._status = QUEUED
        self.errors = []
        self.thread = Thread(target=self.simulate)
        self.simulationThread = SimulationThread(self.modelpath, self.htcFile.filename, hawc2exe)
        self.timer = RepeatedTimer(1, self.update_status)

    def __str__(self):
        return "Simulation(%s)" % self.filename

    def update_status(self, *args, **kwargs):
        if self.status in [INITIALIZING, SIMULATING]:
            self.logFile.update_status()

            if self.logFile.status == log_file.SIMULATING:
                self._status = SIMULATING
            if self.logFile.status == log_file.DONE:
                self._status = FINISH

    def show_status(self):
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


    def additional_files(self):
        additional_files_file = os.path.join(self.modelpath, 'additional_files.txt')
        additional_files = {}
        if os.path.isfile(additional_files_file):
            with open(additional_files_file) as fid:
                additional_files = json.load(fid)
        return additional_files

    def add_additional_input_file(self, file):
        additional_files = self.additional_files()
        additional_files['input'] = additional_files.get('input', []) + [file]
        additional_files_file = os.path.join(self.modelpath, 'additional_files.txt')
        with open(additional_files_file, 'w') as fid:
                json.dump(additional_files, fid)

    def prepare_simulation(self, id):
        self.tmp_modelpath = os.path.join(self.modelpath, id + "/")


        for src in self.htcFile.input_files() + self.htcFile.turbulence_files() + self.additional_files().get('input', []):
            if not os.path.isabs(src):
                src = os.path.join(self.modelpath, src)
            for src_file in glob.glob(src):
                dst = os.path.join(self.tmp_modelpath, os.path.relpath(src_file, self.modelpath))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src_file, dst)
                if not os.path.isfile(dst) or os.stat(dst).st_size != os.stat(src_file).st_size:
                    print ("error copy ", dst)
                else:
                    #print (dst)
                    pass


        self.logFile.filename = os.path.join(self.tmp_modelpath, self.log_filename)
        self.simulationThread.modelpath = self.tmp_modelpath



    def finish_simulation(self):
        if self.status == CLEANED: return
        files = self.htcFile.output_files()
        if self.copy_turbulence:
            files.extend(self.htcFile.turbulence_files())
        for dst in files:
            if not os.path.isabs(dst):
                dst = os.path.join(self.modelpath, dst)
            src = os.path.join(self.tmp_modelpath, os.path.relpath(dst, self.modelpath))

            for src_file in glob.glob(src):
                dst_file = os.path.join(self.modelpath, os.path.relpath(src_file, self.tmp_modelpath))
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                if not os.path.isfile(dst_file) or os.path.getmtime(dst_file) != os.path.getmtime(src_file):
                    shutil.copy(src_file, dst_file)

        self.logFile.filename = os.path.join(self.modelpath, self.log_filename)


        try:
            shutil.rmtree(self.tmp_modelpath)
        except (PermissionError, OSError) as e:
            raise Warning(str(e))
        if self.status != ERROR:
            self.status = CLEANED



    def simulate(self):
        self.is_simulating = True
        self.errors = []
        self.logFile.clear()
        self.status = INITIALIZING

        self.simulationThread.start()
        self.simulationThread.join()
        self.returncode, self.stdout = self.simulationThread.res
        if self.returncode or 'error' in self.stdout.lower():
            self.errors = (list(set([l for l in self.stdout.split("\n") if 'error' in l.lower()])))
            self.status = ERROR
        self.is_simulating = False
        self.logFile.update_status()
        if self.returncode or self.errors:
            raise Exception("Simulation error:\n" + "\n".join(self.errors))
        if self.logFile.status != log_file.DONE or self.logFile.errors:
            raise Warning("Simulation succeded with errors:\nLog status:%s\n" % self.logFile.status + "\n".join(self.logFile.errors))



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
        """Start non blocking simulation"""
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

    def __init__(self, modelpath, htcfile, hawc2exe):
        Thread.__init__(self)
        self.modelpath = modelpath
        self.htcfile = os.path.relpath(htcfile, self.modelpath)
        self.hawc2exe = hawc2exe
        self.res = [0, "", ""]


    def start(self):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        CREATE_NO_WINDOW = 0x08000000
        self.process = subprocess.Popen([self.hawc2exe, self.htcfile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False, cwd=self.modelpath, creationflags=CREATE_NO_WINDOW)

        Thread.start(self)


    def run(self):
        p = psutil.Process(os.getpid())
        p.nice = psutil.BELOW_NORMAL_PRIORITY_CLASS
        stdout, _ = self.process.communicate()
        errorcode = self.process.returncode
        self.res = errorcode, stdout.decode()



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
