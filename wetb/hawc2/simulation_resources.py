'''
Created on 22. dec. 2016

@author: mmpe
'''
from datetime import datetime
import glob
import io
import os
import shutil
from subprocess import STDOUT
import subprocess
from threading import Thread
import time

from wetb.hawc2 import log_file
from wetb.hawc2.log_file import LogInfo, LogFile
from wetb.hawc2.simulation import ERROR, ABORTED

from wetb.utils.cluster_tools.cluster_resource import LocalResource, \
    SSHPBSClusterResource
from wetb.utils.cluster_tools import pbsjob
from wetb.utils.cluster_tools.pbsjob import SSHPBSJob, NOT_SUBMITTED, DONE
from wetb.hawc2.htc_file import fmt_path
import numpy as np


class SimulationHost(object):
    def __init__(self, simulation):
        self.sim = simulation
    logFile = property(lambda self: self.sim.logFile, lambda self, v: setattr(self.sim, "logFile", v))
    errors = property(lambda self: self.sim.errors)
    modelpath = property(lambda self: self.sim.modelpath)
    exepath = property(lambda self: self.sim.exepath)
    tmp_modelpath = property(lambda self: self.sim.tmp_modelpath, lambda self, v: setattr(self.sim, "tmp_modelpath", v))
    tmp_exepath = property(lambda self: self.sim.tmp_exepath, lambda self, v: setattr(self.sim, "tmp_exepath", v))
    simulation_id = property(lambda self: self.sim.simulation_id)
    stdout_filename = property(lambda self: self.sim.stdout_filename)
    htcFile = property(lambda self: self.sim.htcFile)
    additional_files = property(lambda self: self.sim.additional_files)
    _input_sources = property(lambda self: self.sim._input_sources)
    _output_sources = property(lambda self: self.sim._output_sources)
    log_filename = property(lambda self: self.sim.log_filename)

    status = property(lambda self: self.sim.status, lambda self, v: setattr(self.sim, "status", v))
    is_simulating = property(lambda self: self.sim.is_simulating, lambda self, v: setattr(self.sim, "is_simulating", v))

    def __str__(self):
        return self.resource.host


class LocalSimulationHost(SimulationHost):
    def __init__(self, simulation, resource=None):
        SimulationHost.__init__(self, simulation)
        if resource is None:
            resource = LocalResource(1)
        self.resource = resource
        self.simulationThread = SimulationThread(self.sim)

    def get_datetime(self):
        return datetime.now()

    def glob(self, path, recursive=False):
        if isinstance(path, list):
            return [f for p in path for f in self.glob(p, recursive)]
        if recursive:
            return [os.path.join(root, f) for root, _, files in os.walk(path) for f in files]
        else:
            return glob.glob(path)

    def _prepare_simulation(self, input_files):
        # must be called through simulation object
        # self.tmp_modelpath = os.path.join(self.modelpath, self.tmp_modelpath)
        # self.tmp_exepath = os.path.join(self.tmp_modelpath, os.path.relpath(self.sim.exepath, self.sim.modelpath) ) + "/"
        self.sim.set_id(self.simulation_id, 'Localhost', self.tmp_modelpath)
        for src_file in input_files:
            dst = os.path.join(self.tmp_modelpath, os.path.relpath(src_file, self.modelpath))
            # exist_ok does not exist in Python27
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))  # , exist_ok=True)
            shutil.copy(src_file, dst)
            if not os.path.isfile(dst) or os.stat(dst).st_size != os.stat(src_file).st_size:
                print("error copy ", dst)

        stdout_folder = os.path.join(self.tmp_exepath, os.path.dirname(self.sim.stdout_filename))
        if not os.path.exists(stdout_folder):
            os.makedirs(stdout_folder)  # , exist_ok=True)
        self.logFile.filename = os.path.join(self.tmp_exepath, self.log_filename)
        self.simulationThread.modelpath = self.tmp_modelpath
        self.simulationThread.exepath = self.tmp_exepath

    def _simulate(self):
        # must be called through simulation object
        self.returncode, self.stdout = 1, "Simulation failed"
        self.simulationThread.start()
        self.sim.set_id(self.sim.simulation_id, "Localhost(pid:%d)" %
                        self.simulationThread.process.pid, self.tmp_modelpath)
        self.simulationThread.join()
        self.returncode, self.stdout = self.simulationThread.res
        self.logFile.update_status()
        self.errors.extend(list(set(self.logFile.errors)))

    def _finish_simulation(self, output_files):
        missing_result_files = []
        for src_file in output_files:
            dst_file = os.path.join(self.modelpath, os.path.relpath(src_file, self.tmp_modelpath))
            # exist_ok does not exist in Python27
            try:
                if not os.path.isdir(os.path.dirname(dst_file)):
                    os.makedirs(os.path.dirname(dst_file))  # , exist_ok=True)
                if not os.path.isfile(dst_file) or os.path.getmtime(dst_file) != os.path.getmtime(src_file):
                    shutil.copy(src_file, dst_file)
            except Exception:
                missing_result_files.append(dst_file)

        self.logFile.filename = os.path.join(self.sim.exepath, self.log_filename)
        if missing_result_files:
            raise Warning("Failed to copy %s from %s" % (",".join(missing_result_files), self.resource.host))
#         try:
#             shutil.rmtree(self.tmp_modelpath, ignore_errors=False)
#         except (PermissionError, OSError) as e:
#             try:
#                 # change permissions and delete
#                 for root, folders, files in os.walk(self.tmp_modelpath):
#                     for folder in folders:
#                         os.chmod(os.path.join(root, folder), 0o666)
#                     for file in files:
#                         os.chmod(os.path.join(root, file), 0o666)
#                 shutil.rmtree(self.tmp_modelpath)
#             except (PermissionError, OSError) as e:
#                 raise Warning("Fail to remove temporary files and folders on %s\n%s" % (self.resource.host, str(e)))

    def update_logFile_status(self):
        self.logFile.update_status()

    def stop(self):
        self.simulationThread.stop()
        if self.simulationThread.is_alive():
            self.simulationThread.join()
        print("simulatino_resources.stop joined")


class SimulationThread(Thread):

    def __init__(self, simulation, low_priority=True):
        Thread.__init__(self)
        self.sim = simulation
        self.modelpath = self.sim.modelpath
        self.exepath = self.sim.exepath
        self.res = [0, "", ""]
        self.low_priority = low_priority

    def start(self):
        CREATE_NO_WINDOW = 0x08000000
        exepath = self.exepath  # overwritten in _prepare_simulation
        modelpath = self.modelpath  # overwritten in _prepare_simulation
        htcfile = os.path.relpath(self.sim.htcFile.filename, self.sim.exepath)

        hawc2exe = self.sim.hawc2exe
        stdout = self.sim.stdout_filename
        if not os.path.isdir(os.path.dirname(exepath + self.sim.stdout_filename)):
            os.makedirs(os.path.dirname(exepath + self.sim.stdout_filename))

        with open(os.path.join(exepath, stdout), 'wb') as stdout:
            if isinstance(hawc2exe, tuple):
                wine, hawc2exe = hawc2exe
                # shell must be True to inwoke wine
                self.process = subprocess.Popen(" ".join([wine, hawc2exe, htcfile]),
                                                stdout=stdout, stderr=STDOUT, shell=True, cwd=exepath)
            else:
                self.process = subprocess.Popen([hawc2exe, htcfile], stdout=stdout,
                                                stderr=STDOUT, shell=False, cwd=exepath, creationflags=CREATE_NO_WINDOW)
            # self.process.communicate()

        import psutil
        try:
            self.sim.host.resource.process_name = psutil.Process(self.process.pid).name()
        except Exception:
            pass
        Thread.start(self)

    def run(self):
        import psutil
        p = psutil.Process(os.getpid())
        try:
            if self.low_priority:
                p.set_nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass
        self.process.communicate()
        errorcode = self.process.returncode

        with open(self.exepath + self.sim.stdout_filename, encoding='cp1252') as fid:
            stdout = fid.read()
        self.res = errorcode, stdout

    def stop(self):
        if hasattr(self, 'process'):
            try:
                subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=self.process.pid))
            except Exception:
                pass


class PBSClusterSimulationResource(SSHPBSClusterResource):
    def __init__(self, sshclient, min_cpu, min_free, init_cmd, wine_cmd, python_cmd, queue="workq"):
        SSHPBSClusterResource.__init__(self, sshclient, min_cpu, min_free)
        self.init_cmd = init_cmd
        self.wine_cmd = wine_cmd
        self.python_cmd = python_cmd
        self.queue = queue

    def is_clean(self):
        return self.execute("find .hawc2launcher/ -type f | wc -l")[1] > 0

    def clean(self):
        try:
            self.execute('rm .hawc2launcher -r -f')
        except Exception:
            pass
        try:
            self.shared_ssh.close()
        except Exception:
            pass

    def update_resource_status(self):
        try:
            _, out, _ = self.ssh.execute("find .hawc2launcher/ -name '*.out'")
            self.finished = set([f.split("/")[1] for f in out.split("\n") if "/" in f])
        except Exception as e:
            print("resource_manager.update_status, out", str(e))
            pass

        try:
            _, out, _ = self.ssh.execute("find .hawc2launcher -name 'status*' -exec cat {} \;")
            self.loglines = {l.split(";")[0]: l.split(";")[1:] for l in out.split("\n") if ";" in l}
        except Exception as e:
            print("resource_manager.update_status, status file", str(e))
            pass
        try:
            _, out, _ = self.ssh.execute("qstat -u %s" % self.username)
            self.is_executing = set([j.split(".")[0] for j in out.split("\n")[5:] if "." in j])
        except Exception as e:
            print("resource_manager.update_status, qstat", str(e))
            pass


class GormSimulationResource(PBSClusterSimulationResource):
    init_cmd = """export PATH=/home/python/miniconda3/bin:$PATH
source activate wetb_py3"""
    queue = "workq"
    host = "gorm.risoe.dk"

    def __init__(self, username, password, wine_cmd="WINEARCH=win32 WINEPREFIX=~/.wine32 wine"):

        from wetb.utils.cluster_tools.ssh_client import SSHClient
        PBSClusterSimulationResource.__init__(self, SSHClient(
            self.host, username, password, 22), 25, 100, self.init_cmd, wine_cmd, "python", self.queue)


class JessSimulationResource(PBSClusterSimulationResource):
    host = 'jess.dtu.dk'
    init_cmd = """export PATH=/home/python/miniconda3/bin:$PATH
source activate wetb_py3
WINEARCH=win32 WINEPREFIX=~/.wine32 winefix"""
    queue = "windq"

    def __init__(self, username, password, wine_cmd="WINEARCH=win32 WINEPREFIX=~/.wine32 wine"):

        from wetb.utils.cluster_tools.ssh_client import SSHClient
        PBSClusterSimulationResource.__init__(self, SSHClient(
            self.host, username, password, 22), 25, 600, self.init_cmd, wine_cmd, "python", self.queue)


class PBSClusterSimulationHost(SimulationHost):
    def __init__(self, simulation, resource):
        SimulationHost.__init__(self, simulation)
        self.ssh = resource.new_ssh_connection()
        self.pbsjob = SSHPBSJob(resource.new_ssh_connection())
        self.resource = resource

    hawc2exe = property(lambda self: os.path.basename(self.sim.hawc2exe))

    def glob(self, filepattern, cwd="", recursive=False):
        return self.ssh.glob(filepattern, cwd, recursive)

    def get_datetime(self):
        v, out, err = self.ssh.execute('date "+%Y,%m,%d,%H,%M,%S"')
        if v == 0:
            return datetime.strptime(out.strip(), "%Y,%m,%d,%H,%M,%S")

    def _prepare_simulation(self, input_files):
        with self.ssh:
            self.ssh.execute(["mkdir -p .hawc2launcher/%s" % self.simulation_id], verbose=False)
            self.ssh.execute("mkdir -p %s%s" % (self.tmp_exepath, os.path.dirname(self.log_filename)))
            self.ssh.upload_files(self.modelpath, self.tmp_modelpath, file_lst=[os.path.relpath(
                f, self.modelpath) for f in input_files], callback=self.sim.progress_callback("Copy to host"))
#             for src_file in input_files:
#                     dst = unix_path(self.tmp_modelpath + os.path.relpath(src_file, self.modelpath))
#                     self.execute("mkdir -p %s" % os.path.dirname(dst), verbose=False)
#                     self.upload(src_file, dst, verbose=False)
#                     ##assert self.ssh.file_exists(dst)
            f = io.StringIO(self.pbsjobfile(self.sim.ios))
            f.seek(0)
            self.ssh.upload(f, self.tmp_exepath + "%s.in" % self.simulation_id)
            self.ssh.execute("mkdir -p %s%s" % (self.tmp_exepath, os.path.dirname(self.stdout_filename)))
            remote_log_filename = "%s%s" % (self.tmp_exepath, self.log_filename)
            self.ssh.execute("rm -f %s" % remote_log_filename)

    def _finish_simulation(self, output_files):
        with self.ssh:
            download_failed = []
            try:
                self.ssh.download_files(self.tmp_modelpath, self.modelpath, file_lst=[os.path.relpath(
                    f, self.tmp_modelpath) for f in output_files], callback=self.sim.progress_callback("Copy from host"))
            except Exception:
                #
                #             for src_file in output_files:
                #                 try:
                #                     dst_file = os.path.join(self.modelpath, os.path.relpath(src_file, self.tmp_modelpath))
                #                     os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                #                     self.download(src_file, dst_file, retry=10)
                #                 except Exception as e:
                #                     download_failed.append(dst_file)
                #             if download_failed:
                raise Warning("Failed to download %s from %s" % (",".join(download_failed), self.ssh.host))
            else:
                try:
                    self.ssh.execute('rm -r .hawc2launcher/%s' % self.simulation_id)
                finally:
                    try:
                        self.ssh.execute('rm .hawc2launcher/status_%s' % self.simulation_id)
                    except Exception:
                        raise Warning("Fail to remove temporary files and folders on %s" % self.ssh.host)

    def _simulate(self):
        """starts blocking simulation"""
        self.sim.logFile = LogInfo(log_file.MISSING, 0, "None", "")

        self.pbsjob.submit("%s.in" % self.simulation_id, self.tmp_exepath, self.sim.stdout_filename)
        sleeptime = 1
        while self.is_simulating:
            time.sleep(sleeptime)

        local_out_file = self.exepath + self.stdout_filename
        with self.ssh:
            try:
                self.ssh.download(self.tmp_exepath + self.stdout_filename, local_out_file)
                with open(local_out_file) as fid:
                    _, self.stdout, returncode_str, _ = fid.read().split("---------------------")
                    self.returncode = returncode_str.strip() != "0"
            except Exception:
                self.returncode = 1
                self.stdout = "error: Could not download and read stdout file"
            try:
                self.ssh.download(self.tmp_exepath + self.log_filename, self.exepath + self.log_filename)
            except Exception:
                raise Warning("Logfile not found", self.tmp_modelpath + self.log_filename)
        self.sim.logFile = LogFile.from_htcfile(self.htcFile, self.exepath)

    def update_logFile_status(self):
        def pbsjob_status():
            if self.pbsjob._status in [NOT_SUBMITTED, DONE]:
                return self.pbsjob._status
            if self.pbsjob.jobid in self.resource.is_executing:
                self.pbsjob._status = pbsjob.RUNNING
            elif self.simulation_id in self.resource.finished:
                self.pbsjob._status = DONE
                self.pbsjob.jobid = None
            return self.pbsjob._status

        def set_status():
            if self.simulation_id in self.resource.loglines:
                self.logline = self.resource.loglines[self.simulation_id]
                self.status = self.logline[0]
                self.logFile = LogInfo(*self.logline[1:])

        status = pbsjob_status()
        if status == pbsjob.NOT_SUBMITTED:
            pass
        elif status == pbsjob.DONE:
            if self.is_simulating:
                set_status()
            self.is_simulating = False
        else:
            set_status()

    def start(self):
        """Start non blocking distributed simulation"""
        self.non_blocking_simulation_thread.start()

    def abort(self):
        self.pbsjob.stop()
        self.stop()
        try:
            self.finish_simulation()
        except Exception:
            pass
        self.is_simulating = False
        self.is_done = True
        if self.status != ERROR and self.logFile.status not in [log_file.DONE]:
            self.status = ABORTED

    def stop(self):
        self.is_simulating = False
        self.pbsjob.stop()

    def pbsjobfile(self, ios=False):
        cp_back = ""
        for folder in set([fmt_path(os.path.dirname(os.path.relpath(f))) for f in self.htcFile.output_files() + self.htcFile.turbulence_files()]):
            if folder == '':
                continue
            cp_back += "mkdir -p $PBS_O_WORKDIR/%s/. \n" % folder
            cp_back += "cp -R -f %s/. $PBS_O_WORKDIR/%s/.\n" % (folder, folder)
        rel_htcfilename = fmt_path(os.path.relpath(self.htcFile.filename, self.exepath))
        try:
            steps = self.htcFile.simulation.time_stop[0] / self.htcFile.simulation.newmark.deltat[0]
            walltime = "%02d:00:00" % np.ceil(steps / 500 / 60)
        except Exception:
            walltime = "04:00:00"
        init = """
### Standard Output
#PBS -N h2l_%s
### merge stderr into stdout
#PBS -j oe
#PBS -o %s
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=%s
###PBS -a 201547.53
#PBS -lnodes=1:ppn=1
### Queue name
#PBS -q %s
### Create scratch directory and copy data to it
cd $PBS_O_WORKDIR
pwd""" % (self.simulation_id, self.stdout_filename, walltime, self.resource.queue)
        copy_to = """
cp -R %s /scratch/$USER/$PBS_JOBID
### Execute commands on scratch nodes
cd /scratch/$USER/$PBS_JOBID%s
pwd""" % ((".", "../")[ios], ("", "/input")[ios])
        run = '''
%s
### modelpath: %s
### htc: %s
echo "---------------------"
%s -c "from wetb.hawc2.cluster_simulation import ClusterSimulation;ClusterSimulation('.','%s', ('%s','%s'))"
echo "---------------------"
echo $?
echo "---------------------"''' % (self.resource.init_cmd, self.modelpath, self.htcFile.filename, self.resource.python_cmd, rel_htcfilename, self.resource.wine_cmd, self.hawc2exe)
        copy_back = """
### Copy back from scratch directory
cd /scratch/$USER/$PBS_JOBID%s
%s
echo $PBS_JOBID
cd /scratch/
### rm -r $PBS_JOBID
exit""" % (("", "/input")[ios], cp_back)
        return init + copy_to + run + copy_back
