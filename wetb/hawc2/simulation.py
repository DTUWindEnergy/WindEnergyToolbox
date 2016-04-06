from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from io import open
from builtins import str
from future import standard_library
from wetb.utils.cluster_tools.cluster_resource import LocalResource, \
    PBSClusterResource
from wetb.utils.cluster_tools.pbsjob import PBSJob
import io
import time
from wetb.utils.cluster_tools import pbsjob
from wetb.utils.cluster_tools.ssh_client import SSHClient
standard_library.install_aliases()
import os
from wetb.hawc2.htc_file import HTCFile
from wetb.hawc2.log_file import LogFile, LogInfo
from threading import Timer, Thread
import sys
from multiprocessing import Process
import psutil
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
FINISH = "Finish"  # when HAWC2 finish
ERROR = "Error"  # when hawc2 returns error
ABORTED = "Aborted"  # when stopped and logfile.status != Done
CLEANED = "Cleaned"  # after copy back

class Simulation(object):
    is_simulating = False
    status = QUEUED
    def __init__(self, modelpath, htcfilename, hawc2exe="HAWC2MB.exe"):
        self.modelpath = os.path.abspath(modelpath) + "/"
        self.folder = os.path.dirname(htcfilename)
        if not os.path.isabs(htcfilename):
            htcfilename = os.path.join(modelpath, htcfilename)
        self.filename = os.path.basename(htcfilename)
        self.htcFile = HTCFile(htcfilename)
        self.time_stop = self.htcFile.simulation.time_stop[0]
        self.copy_turbulence = True
        self.simulation_id = os.path.relpath(htcfilename, self.modelpath).replace("\\", "_") + "_%d" % id(self)
        self.stdout_filename = "stdout/%s.out" % self.simulation_id
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
        self.last_status = self.status
        self.errors = []
        self.thread = Thread(target=self.simulate_distributed)
        self.hawc2exe = hawc2exe
        self.updateStatusThread = UpdateStatusThread(self)
        self.host = LocalSimulationHost(self)

    def input_sources(self):
        def fmt(src):
            if os.path.isabs(src):
                src = os.path.relpath(os.path.abspath(src), self.modelpath)
            else:
                src = os.path.relpath (src)
            assert not src.startswith(".."), "%s referes to a file outside the model path\nAll input files be inside model path" % src
            return src
        return [fmt(src) for src in self.htcFile.input_files() + self.htcFile.turbulence_files() + self.additional_files().get('input', [])]

    def output_sources(self):
        def fmt(dst):
            if os.path.isabs(dst):
                dst = os.path.relpath(os.path.abspath(dst), self.modelpath)
            else:
                dst = os.path.relpath (dst)
            dst = dst.replace("\\", "/")
            assert not dst.startswith(".."), "%s referes to a file outside the model path\nAll input files be inside model path" % dst
            return dst
        return [fmt(dst) for dst in self.htcFile.output_files() + ([], self.htcFile.turbulence_files())[self.copy_turbulence] + [self.stdout_filename]]


    def prepare_simulation(self):
        self.status = PREPARING
        self.tmp_modelpath = os.path.join(".hawc2launcher/%s/" % self.simulation_id)
        self.set_id(self.simulation_id, str(self.host), self.tmp_modelpath)
        self.host._prepare_simulation()

    def simulate(self):
        #starts blocking simulation
        self.is_simulating = True
        self.errors = []
        self.status = INITIALIZING
        self.logFile.clear()

        self.host._simulate()

        if self.host.returncode or 'error' in self.host.stdout.lower():
            self.errors = (list(set([l for l in self.host.stdout.split("\n") if 'error' in l.lower()])))
            self.status = ERROR
        if  'HAWC2MB version:' not in self.host.stdout:
            self.errors.append(self.stdout)
            self.status = ERROR

        self.logFile.update_status()
        self.errors.extend(list(set(self.logFile.errors)))
        self.update_status()
        self.is_simulating = False
        if self.host.returncode or self.errors:
            raise Exception("Simulation error:\n" + "\n".join(self.errors))
        elif self.logFile.status != log_file.DONE or self.logFile.errors:
            raise Warning("Simulation succeded with errors:\nLog status:%s\nErrors:\n%s" % (self.logFile.status, "\n".join(self.logFile.errors)))
        else:
            self.status = FINISH


    def finish_simulation(self):
        lock = threading.Lock()
        with lock:
            if self.status == CLEANED: return
            if self.status != ERROR:
                self.status = CLEANED
        self.host._finish_simulation()
        self.set_id(self.simulation_id)




    def update_status(self, *args, **kwargs):
        self.host.update_logFile_status()
        if self.status in [INITIALIZING, SIMULATING]:
            if self.logFile.status == log_file.SIMULATING:
                self.status = SIMULATING
            if self.logFile.status == log_file.DONE and self.is_simulating is False:
                self.status = FINISH


    def __str__(self):
        return "Simulation(%s)" % self.filename

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
        elif self.logFile.status == log_file.UNKNOWN:
            print (self.status)
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
        additional_files['input'] = list(set(additional_files.get('input', []) + [file]))
        additional_files_file = os.path.join(self.modelpath, 'additional_files.txt')
        with open(additional_files_file, 'w', encoding='utf-8') as fid:
                json.dump(additional_files, fid)


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
                    self.show_message("'%s' is now added to additional_files.txt.\n\nPlease restart the simulation" % filename)
        for error in self.errors:
            for regex in [r".*\*\*\* ERROR \*\*\* File '(.*)' does not exist in the (.*) folder",
                          r".*\*\*\* ERROR \*\*\* DLL (.*)()"]:
                m = re.compile(regex).match(error.strip())
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

    def start(self, update_interval=1):
        """Start non blocking distributed simulation"""
        self.is_simulating = True
        self.updateStatusThread.start()
        self.thread.start()

    def wait(self):
        self.thread.join()
        self.update_status()

    def abort(self):
        self.host.stop()
        for _ in range(100):
            if self.is_simulating:
                break
            time.sleep(0.1)
        try:
            self.finish_simulation()
        except Exception as e:
            print (str(e))
            pass
        if self.logFile.status not in [log_file.DONE]:
            self.status = ABORTED
        self.update_status()


class SimulationThread(Thread):

    def __init__(self, simulation, low_priority=True):
        Thread.__init__(self)
        self.sim = simulation
        self.modelpath = self.sim.modelpath
        self.res = [0, "", ""]
        self.low_priority = low_priority


    def start(self):
        CREATE_NO_WINDOW = 0x08000000
        modelpath = self.modelpath
        htcfile = os.path.relpath(self.sim.htcFile.filename, self.sim.modelpath)
        hawc2exe = self.sim.hawc2exe
        stdout = self.sim.stdout_filename
        if os.name == "nt":
            self.process = subprocess.Popen('"%s" %s 1> %s 2>&1' % (hawc2exe, htcfile, stdout), stdout=None, stderr=None, shell=True, cwd=modelpath)  #, creationflags=CREATE_NO_WINDOW)
        else:
            self.process = subprocess.Popen('wine "%s" %s 1> %s 2>&1' % (hawc2exe, htcfile, stdout), stdout=None, stderr=None, shell=True, cwd=modelpath)
        Thread.start(self)


    def run(self):
        p = psutil.Process(os.getpid())
        if self.low_priority:
            p.set_nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.process.communicate()
        errorcode = self.process.returncode
        with open(self.modelpath + self.sim.stdout_filename, encoding='utf-8') as fid:
            stdout = fid.read()
        self.res = errorcode, stdout

    def stop(self):
        subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=self.process.pid))

class UpdateStatusThread(Thread):
    def __init__(self, simulation, interval=1):
        Thread.__init__(self)
        self.simulation = simulation
        self.interval = interval

    def start(self):
        Thread.start(self)

    def run(self):
        while self.simulation.is_simulating:
            self.simulation.update_status()
            time.sleep(self.interval)


class SimulationResource(object):
    def __init__(self, simulation):
        self.sim = simulation
    logFile = property(lambda self : self.sim.logFile, lambda self, v: setattr(self.sim, "logFile", v))
    errors = property(lambda self : self.sim.errors)
    modelpath = property(lambda self : self.sim.modelpath)
    tmp_modelpath = property(lambda self : self.sim.tmp_modelpath, lambda self, v: setattr(self.sim, "tmp_modelpath", v))
    simulation_id = property(lambda self : self.sim.simulation_id)
    htcFile = property(lambda self : self.sim.htcFile)
    additional_files = property(lambda self : self.sim.additional_files)
    input_sources = property(lambda self : self.sim.input_sources)
    output_sources = property(lambda self : self.sim.output_sources)
    log_filename = property(lambda self : self.sim.log_filename)

    status = property(lambda self : self.sim.status, lambda self, v: setattr(self.sim, "status", v))
    is_simulating = property(lambda self : self.sim.is_simulating, lambda self, v: setattr(self.sim, "is_simulating", v))

    def __str__(self):
        return self.host
class LocalSimulationHost(SimulationResource):
    def __init__(self, simulation):
        SimulationResource.__init__(self, simulation)
        LocalResource.__init__(self, "hawc2mb")
        self.simulationThread = SimulationThread(self.sim)

    def _prepare_simulation(self):
        # must be called through simulation object
        self.tmp_modelpath = os.path.join(self.modelpath, self.tmp_modelpath)
        self.sim.set_id(self.simulation_id, 'Localhost', self.tmp_modelpath)
        for src in self.input_sources():
            for src_file in glob.glob(os.path.join(self.modelpath, src)):
                dst = os.path.join(self.tmp_modelpath, os.path.relpath(src_file, self.modelpath))
                # exist_ok does not exist in Python27
                if not os.path.exists(os.path.dirname(dst)):
                    os.makedirs(os.path.dirname(dst))  #, exist_ok=True)
                shutil.copy(src_file, dst)
                if not os.path.isfile(dst) or os.stat(dst).st_size != os.stat(src_file).st_size:
                    print ("error copy ", dst)

        if not os.path.exists(os.path.join(self.tmp_modelpath, 'stdout')):
            os.makedirs(os.path.join(self.tmp_modelpath, 'stdout'))  #, exist_ok=True)
        self.logFile.filename = os.path.join(self.tmp_modelpath, self.log_filename)
        self.simulationThread.modelpath = self.tmp_modelpath


    def _simulate(self):
        #must be called through simulation object
        self.returncode, self.stdout = 1, "Simulation failed"
        self.simulationThread.start()
        self.simulationThread.join()
        self.returncode, self.stdout = self.simulationThread.res
        self.logFile.update_status()
        self.errors.extend(list(set(self.logFile.errors)))


    def _finish_simulation(self):
        for dst in self.output_sources():
            src = os.path.join(self.tmp_modelpath, dst)

            for src_file in glob.glob(src):
                dst_file = os.path.join(self.modelpath, os.path.relpath(src_file, self.tmp_modelpath))
                # exist_ok does not exist in Python27
                if not os.path.isdir(os.path.dirname(dst_file)):
                    os.makedirs(os.path.dirname(dst_file))  #, exist_ok=True)
                if not os.path.isfile(dst_file) or os.path.getmtime(dst_file) != os.path.getmtime(src_file):
                    shutil.copy(src_file, dst_file)

        self.logFile.filename = os.path.join(self.modelpath, self.log_filename)

        try:
            shutil.rmtree(self.tmp_modelpath)
        except (PermissionError, OSError) as e:
            raise Warning(str(e))

    def update_logFile_status(self):
        self.logFile.update_status()

    def stop(self):
        self.simulationThread.stop()
        self.simulationThread.join()



class PBSClusterSimulationHost(SimulationResource, SSHClient):
    def __init__(self, simulation, host, username, password, port=22):
        SimulationResource.__init__(self, simulation)
        SSHClient.__init__(self, host, username, password, port=port)
        self.pbsjob = PBSJob(self)

    hawc2exe = property(lambda self : os.path.basename(self.sim.hawc2exe))


    def _prepare_simulation(self):
        with self:
            self.execute(["mkdir -p .hawc2launcher/%s" % self.simulation_id], verbose=False)
            self.execute("mkdir -p %s%s" % (self.tmp_modelpath, os.path.dirname(self.log_filename)))

            for src in self.input_sources():
                for src_file in glob.glob(os.path.join(self.modelpath, src)):
                    dst = (self.tmp_modelpath + os.path.relpath(src_file, self.modelpath)).replace("\\", "/")
                    self.execute("mkdir -p %s" % os.path.dirname(dst), verbose=False)
                    self.upload(src_file, dst, verbose=False)
                    ##assert self.ssh.file_exists(dst)

            f = io.StringIO(self.pbsjobfile(self.simulation_id))
            f.seek(0)
            self.upload(f, self.tmp_modelpath + "%s.in" % self.simulation_id)
            remote_log_filename = "%s%s" % (self.tmp_modelpath, self.log_filename)
            self.execute("rm -f %s" % remote_log_filename)



    def _finish_simulation(self):
        with self:
            for dst in self.output_sources():

                src = os.path.join(self.tmp_modelpath, dst).replace("\\", "/")
                for src_file in self.glob(src):
                    dst_file = os.path.join(self.modelpath, os.path.relpath(src_file, self.tmp_modelpath))
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    self.download(src_file, dst_file, verbose=False)
            self.execute('rm -r .hawc2launcher/%s' % self.simulation_id)
            self.execute('rm .hawc2launcher/status_%s' % self.simulation_id)


    def _simulate(self):
        """starts blocking simulation"""
        self.sim.logFile = LogInfo(log_file.MISSING, 0, "None", "")

        self.pbsjob.submit("%s.in" % self.simulation_id, self.tmp_modelpath , self.sim.stdout_filename)
        sleeptime = 1
        while self.is_simulating:
            #self.__update_logFile_status()
            time.sleep(sleeptime)

        local_out_file = self.modelpath + self.sim.stdout_filename
        with self:
            try:
                self.download(self.tmp_modelpath + self.sim.stdout_filename, local_out_file)
                with open(local_out_file) as fid:
                    _, self.stdout, returncode_str, _ = fid.read().split("---------------------")
                    self.returncode = returncode_str.strip() != "0"
            except Exception:
                self.returncode = 1
                self.stdout = "error: Could not download and read stdout file"
            try:
                self.download(self.tmp_modelpath + self.log_filename, self.modelpath + self.log_filename)
            except Exception:
                raise Exception ("Logfile not found")
        self.sim.logFile = LogFile.from_htcfile(self.htcFile, self.modelpath)



    def update_logFile_status(self):
        status = self.pbsjob.status
        if status == pbsjob.NOT_SUBMITTED:
            pass
        elif status == pbsjob.DONE:
            self.is_simulating = False
            pass
        else:
            try:
                _, out, _ = self.execute("cat .hawc2launcher/status_%s" % self.simulation_id)
                out = out.split(";")
                if len(out) == 5:
                    self.status = out[0]
                    self.logFile = LogInfo(*out[1:])

            except Exception as e:
                if "No such file or directory" in str(e):
                    pass
                else:
                    raise

    def start(self):
        """Start non blocking distributed simulation"""
        self.thread.start()



    def abort(self):
        self.pbsjob.stop()
        self.stop()
        try:
            self.finish_simulation()
        except:
            pass
        if self.status != ERROR and self.logFile.status not in [log_file.DONE]:
            self.status = ABORTED

    def stop(self):
        self.is_simulating = False
        self.pbsjob.stop()

    def pbsjobfile(self, simulation_id):
        cp_back = ""
        for folder in set([os.path.relpath(os.path.dirname(f)) for f in self.htcFile.output_files() + self.htcFile.turbulence_files()]):
            cp_back += "mkdir -p $PBS_O_WORKDIR/%s/. \n" % folder
            cp_back += "cp -R -f %s/. $PBS_O_WORKDIR/%s/.\n" % (folder, folder)
        rel_htcfilename = os.path.relpath(self.htcFile.filename, self.modelpath).replace("\\", "/")
        return """
### Standard Output
#PBS -N h2l_%s
### merge stderr into stdout
#PBS -j oe
#PBS -o %s.out
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=01:00:00
###PBS -a 201547.53
#PBS -lnodes=1:ppn=1
### Queue name
#PBS -q workq
### Create scratch directory and copy data to it
cd $PBS_O_WORKDIR
pwd
cp -R . /scratch/$USER/$PBS_JOBID
### Execute commands on scratch nodes
cd /scratch/$USER/$PBS_JOBID
pwd
echo "---------------------"
python -c "from wetb.hawc2.cluster_simulation import ClusterSimulation;ClusterSimulation('.','%s', '%s')"
echo "---------------------"
echo $?
echo "---------------------"
### Copy back from scratch directory
cd /scratch/$USER/$PBS_JOBID
%s
echo $PBS_JOBID
cd /scratch/
### rm -r $PBS_JOBID
exit""" % (simulation_id, simulation_id, rel_htcfilename, self.hawc2exe, cp_back)




