
import glob
import json
import os
import re
import sys
from threading import Thread
import time

from wetb.hawc2 import log_file
from wetb.hawc2.htc_file import HTCFile, fmt_path
from wetb.hawc2.log_file import LogFile
import tempfile
import stat
import shutil



QUEUED = "queued"  # until start
PREPARING = "Copy to host"  # during prepare simulation
INITIALIZING = "Initializing"  # when starting
SIMULATING = "Simulating"  # when logfile.status=simulating
FINISHING = "Copy from host"  # during prepare simulation
FINISH = "Simulation finish"  # when HAWC2 finish
ERROR = "Error"  # when hawc2 returns error
ABORTED = "Aborted"  # when stopped and logfile.status != Done
CLEANED = "Cleaned"  # after copy back


class Simulation(object):
    """Class for doing hawc2 simulations




    Examples
    --------
    >>> sim = Simulation("<path>/MyModel","htc/MyHtc")

    Blocking inplace simulation
    >>> sim.simulate()

    Non-blocking distributed simulation(everything copied to temporary folder)\n
    Starts two threads:
    - non_blocking_simulation_thread:
        - prepare_simulation() # copy to temporary folder
        - simulate() # simulate
        - finish_simulation # copy results back again
    - updateSimStatusThread:
        - update status every second
    >>> sim.start()
    >>> while sim.status!=CLEANED:
    >>>     sim.show_status()


    The default host is LocalSimulationHost. To simulate on pbs featured cluster
    >>> sim.host = PBSClusterSimulationHost(sim, <hostname>, <username>, <password>)
    >>> sim.start()
    >>> while sim.status!=CLEANED:
    >>>     sim.show_status()
    """

    is_simulating = False
    is_done = False
    status = QUEUED

    def __init__(self, modelpath, htcfile, hawc2exe="HAWC2MB.exe", copy_turbulence=True):
        if isinstance(htcfile, HTCFile):
            htcfilename = htcfile.filename
        else:
            htcfilename = htcfile

        self.modelpath = os.path.abspath(modelpath).replace("\\", "/")
        if self.modelpath[-1] != '/':
            self.modelpath += "/"
        if os.path.isabs(htcfilename):
            htcfilename = os.path.relpath(htcfilename, modelpath)
        if htcfilename.startswith("input/"):
            htcfilename = htcfilename[6:]
        exists = [os.path.isfile(os.path.join(modelpath, htcfilename)),
                  os.path.isfile(os.path.join(modelpath, "input/", htcfilename))]
        if all(exists):
            raise Exception(
                "Both standard and input/output file structure available for %s in %s. Delete one of the options" % (htcfilename, modelpath))
        if not any(exists):
            raise Exception("%s not found in %s" % (htcfilename, modelpath))
        else:
            self.ios = exists[1]  # input/output file structure

        if self.ios:
            self.exepath = self.modelpath + "input/"
        else:
            self.exepath = self.modelpath
        # model_path: top level path containing all resources
        # exepath: parent path for relative paths

        htcfilename = fmt_path(htcfilename)

        self.tmp_modelpath = self.exepath
        self.folder = os.path.dirname(htcfilename)

        self.filename = os.path.basename(htcfilename)
        if isinstance(htcfile, HTCFile):
            self.htcFile = htcfile
        else:
            self.htcFile = HTCFile(os.path.join(self.exepath, htcfilename), self.exepath)
        self.time_stop = self.htcFile.simulation.time_stop[0]
        self.hawc2exe = hawc2exe
        self.copy_turbulence = copy_turbulence
        self.simulation_id = (htcfilename.replace("\\", "/").replace("/", "_")[:50] + "_%d" % id(self))
        if self.simulation_id.startswith("input_"):
            self.simulation_id = self.simulation_id[6:]
        self.stdout_filename = fmt_path(os.path.join(os.path.relpath(self.exepath, self.modelpath),
                                                     (os.path.splitext(htcfilename)[0] + ".out").replace('htc', 'stdout', 1)))
        if self.ios:
            assert self.stdout_filename.startswith("input/")
            self.stdout_filename = self.stdout_filename.replace("input/", "../output/")
        # self.stdout_filename = "stdout/%s.out" % self.simulation_id
        if 'logfile' in self.htcFile.simulation:
            self.log_filename = self.htcFile.simulation.logfile[0]
        else:
            self.log_filename = self.stdout_filename
        if os.path.isabs(self.log_filename):
            self.log_filename = os.path.relpath(self.log_filename, self.modelpath)
        else:
            self.log_filename = os.path.relpath(self.log_filename)
        self.log_filename = fmt_path(self.log_filename)
        self.logFile = LogFile(os.path.join(self.exepath, self.log_filename), self.time_stop)
        self.logFile.clear()
        self.last_status = self.status
        self.errors = []
        from wetb.hawc2.simulation_resources import LocalSimulationHost
        self.host = LocalSimulationHost(self)
        self.stdout = ""
        self.returncode = 0

    def start(self, update_interval=1):
        """Start non blocking distributed simulation"""
        self.is_simulating = True
        if update_interval > 0:
            self.updateSimStatusThread = UpdateSimStatusThread(self)
            self.updateSimStatusThread.interval = update_interval
            self.updateSimStatusThread.start()
        self.non_blocking_simulation_thread = Thread(
            target=lambda: self.simulate_distributed(raise_simulation_errors=False))
        self.non_blocking_simulation_thread.start()

    def wait(self):
        self.non_blocking_simulation_thread.join()
        self.update_status()

    def abort(self, update_status=True):
        self.status = ABORTED
        self.is_simulating = False
        self.is_done = True
        self.host.stop()
        if update_status:
            self.update_status()

#         if self.status != QUEUED:
#             self.host.stop()
#             for _ in range(50):
#                 if self.is_simulating is False:
#                     break
#                 time.sleep(0.1)
#         if self.logFile.status not in [log_file.DONE]:
#             self.status = ABORTED
#         self.is_simulating = False
#         self.is_done = True
#         if update_status:
#             self.update_status()

    def show_status(self):
        # print ("log status:", self.logFile.status)
        if self.logFile.status == log_file.SIMULATING:
            if self.last_status != log_file.SIMULATING:
                print("|" + ("-" * 50) + "|" + ("-" * 49) + "|")
                sys.stdout.write("|")
            sys.stdout.write("." * (self.logFile.pct - getattr(self, 'last_pct', 0)))
            sys.stdout.flush()
            self.last_pct = self.logFile.pct
        elif self.last_status == log_file.SIMULATING:
            sys.stdout.write("." * (100 - self.last_pct) + "|")
            sys.stdout.flush()
            print("\n")
        elif self.logFile.status == log_file.UNKNOWN:
            print(self.status)
        else:
            print(self.logFile.status)
        if self.logFile.status != log_file.SIMULATING:
            if self.logFile.errors:
                print(self.logFile.errors)
        self.last_status = self.logFile.status

    def prepare_simulation(self):
        self.status = PREPARING
        # self.tmp_modelpath = os.path.join(".hawc2launcher/%s/" % self.simulation_id)
        # self.tmp_exepath = os.path.join(self.tmp_modelpath, os.path.relpath(self.exepath, self.modelpath) ) + "/"
        self.set_id(self.simulation_id, str(self.host), self.tmp_modelpath)

        def fmt(src):
            if os.path.isabs(src):
                src = os.path.relpath(os.path.abspath(src), self.exepath)
            else:
                src = os.path.relpath(src)
            assert not src.startswith(
                ".."), "%s referes to a file outside the model path\nAll input files be inside model path" % src
            return src
        if self.ios:
            input_folder_files = []
            for root, _, filenames in os.walk(os.path.join(self.modelpath, "input/")):
                for filename in filenames:
                    input_folder_files.append(os.path.join(root, filename))

            input_patterns = [fmt(src) for src in input_folder_files + ([], self.htcFile.turbulence_files())
                              [self.copy_turbulence] + self.additional_files().get('input', [])]
        else:
            input_patterns = [fmt(src) for src in self.htcFile.input_files(
            ) + ([], self.htcFile.turbulence_files())[self.copy_turbulence] + self.additional_files().get('input', [])]
        input_files = set([f for pattern in input_patterns for f in glob.glob(
            os.path.join(self.exepath, pattern)) if os.path.isfile(f) and ".hawc2launcher" not in f])

        self.host._prepare_simulation(input_files)


#        return [fmt(src) for src in self.htcFile.input_files() + self.htcFile.turbulence_files() + self.additional_files().get('input', [])]
#
#        for src in self._input_sources():
#            for src_file in glob.glob(os.path.join(self.modelpath, src)):
#
#
#        self.host._prepare_simulation()

    def simulate(self):
        # starts blocking simulation

        self.is_simulating = True
        self.errors = []
        self.status = INITIALIZING
        self.logFile.clear()
        self.host._simulate()
        self.returncode, self.stdout = self.host.returncode, self.host.stdout
        if self.host.returncode or 'error' in self.host.stdout.lower():
            if self.status == ABORTED:
                return
            if "error" in self.host.stdout.lower():
                self.errors = (list(set([l for l in self.host.stdout.split(
                    "\n") if 'error' in l.lower() and not 'rms error' in l])))
            self.status = ERROR
        if 'HAWC2MB version:' not in self.host.stdout and 'Build information for HAWC2MB' not in self.host.stdout:
            self.errors.append(self.host.stdout)
            self.status = ERROR

        self.logFile.update_status()
        self.errors.extend(list(set(self.logFile.errors)))
        self.update_status()
        self.is_simulating = False
        if self.host.returncode or self.errors:
            raise Exception("Simulation error:\nReturn code: %d\n%s" % (self.host.returncode, "\n".join(self.errors)))
        elif self.logFile.status != log_file.DONE or self.logFile.errors:
            raise Warning("Simulation succeded with errors:\nLog status:%s\nErrors:\n%s" %
                          (self.logFile.status, "\n".join(self.logFile.errors)))
        else:
            self.status = FINISH

    def finish_simulation(self):
        if self.status == ABORTED:
            return
        if self.status != ERROR:
            self.status = FINISHING

        def fmt(dst):
            if os.path.isabs(dst):
                dst = os.path.relpath(os.path.abspath(dst), self.exepath)
            else:
                dst = os.path.relpath(dst)
            dst = fmt_path(dst)
            assert not os.path.relpath(os.path.join(self.exepath, dst), self.modelpath).startswith(
                ".."), "%s referes to a file outside the model path\nAll input files be inside model path" % dst
            return dst
        turb_files = [f for f in self.htcFile.turbulence_files()
                      if self.copy_turbulence and not os.path.isfile(os.path.join(self.exepath, f))]
        if self.ios:
            output_patterns = ["../output/*", "../output/"] + turb_files + \
                [os.path.join(self.exepath, self.stdout_filename)]
        else:
            output_patterns = self.htcFile.output_files() + turb_files + \
                [os.path.join(self.exepath, self.stdout_filename)]
        output_files = set(self.host.glob([fmt_path(os.path.join(self.tmp_exepath, fmt(p)))
                                           for p in output_patterns], recursive=self.ios))
        if not os.path.isdir(os.path.dirname(self.exepath + self.stdout_filename)):
            os.makedirs(os.path.dirname(self.exepath + self.stdout_filename))

        try:
            self.host._finish_simulation(output_files)
            if self.status != ERROR:
                self.status = CLEANED
        except Warning as e:
            self.errors.append(str(e))
            if self.status != ERROR:
                self.status = CLEANED
        except Exception as e:
            self.errors.append(str(e))
            raise

        finally:
            self.set_id(self.filename)
            self.logFile.reset()
            self.htcFile.reset()

    def update_status(self, *args, **kwargs):
        self.host.update_logFile_status()
        if self.status in [INITIALIZING, SIMULATING]:
            if self.logFile.status == log_file.SIMULATING:
                self.status = SIMULATING
            if self.logFile.status == log_file.DONE and self.is_simulating is False:
                self.status = FINISH

    def __str__(self):
        return "Simulation(%s)" % self.filename

    def additional_files(self):
        additional_files_file = os.path.join(self.modelpath, 'additional_files.txt')
        additional_files = {}
        if os.path.isfile(additional_files_file):
            with open(additional_files_file, encoding='utf-8') as fid:
                additional_files = json.loads(fid.read().replace("\\", "/"))
        return additional_files

    def add_additional_input_file(self, file):
        additional_files = self.additional_files()
        additional_files['input'] = list(set(additional_files.get('input', []) + [file]))
        additional_files_file = os.path.join(self.modelpath, 'additional_files.txt')
        with open(additional_files_file, 'w', encoding='utf-8') as fid:
            json.dump(additional_files, fid)

    def simulate_distributed(self, raise_simulation_errors=True):
        try:
            with tempfile.TemporaryDirectory(prefix="h2launcher_%s_" % os.path.basename(self.filename)) as tmpdirname:
                self.tmp_modelpath = tmpdirname + "/"
                self.tmp_exepath = os.path.join(self.tmp_modelpath, os.path.relpath(self.exepath, self.modelpath)) + "/"
                self.prepare_simulation()
                try:
                    self.simulate()
                except Warning as e:
                    print("simulation failed", str(self))
                    print("Trying to finish")
                    raise
                finally:
                    try:
                        if self.status != ABORTED:
                            self.finish_simulation()
                    except Exception:
                        print("finish_simulation failed", str(self))
                        raise
#                     finally:
#                         def remove_readonly(fn, path, excinfo):
#                             try:
#                                 os.chmod(path, stat.S_IWRITE)
#                                 fn(path)
#                             except Exception as exc:
#                                 print("Skipped:", path, "because:\n", exc)
#
#                         shutil.rmtree(tmpdirname, onerror=remove_readonly)

        except Exception as e:
            self.status = ERROR
            self.errors.append(str(e))
            if raise_simulation_errors:
                raise e
        finally:
            self.is_done = True

    def fix_errors(self):
        def confirm_add_additional_file(folder, file):
            if os.path.isfile(os.path.join(self.modelpath, folder, file)):
                filename = fmt_path(os.path.join(folder, file))
                if self.get_confirmation(
                        "File missing", "'%s' seems to be missing in the temporary working directory. \n\nDo you want to add it to additional_files.txt" % filename):
                    self.add_additional_input_file(filename)
                    self.show_message(
                        "'%s' is now added to additional_files.txt.\n\nPlease restart the simulation" % filename)
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
        """override in subclass"""
        return True

    def show_message(self, msg, title="Information"):
        print(msg)

    def set_id(self, *args, **kwargs):
        pass

    def progress_callback(self, *args, **kwargs):
        pass


class UpdateSimStatusThread(Thread):
    def __init__(self, simulation, interval=1):
        Thread.__init__(self)
        self.simulation = simulation
        self.interval = interval

    def start(self):
        Thread.start(self)

    def run(self):
        while self.simulation.is_done is False:
            self.simulation.update_status()
            time.sleep(0.5)
            t = time.time()
            while self.simulation.is_simulating and time.time() < t + self.interval:
                time.sleep(1)
