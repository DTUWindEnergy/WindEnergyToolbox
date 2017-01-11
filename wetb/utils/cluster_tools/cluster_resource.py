'''
Created on 04/04/2016

@author: MMPE
'''
import glob
import multiprocessing
import os
import re
import threading

from wetb.utils.cluster_tools import pbswrap
from wetb.utils.cluster_tools.ssh_client import SSHClient, SharedSSHClient


def unix_path(path, cwd=None, fail_on_missing=False):
    """Convert case insensitive filename into unix case sensitive filename

    If no matching file or folder is found an error is raised

    Parameters
    ---------
    x : str
        Case insensitive filename or folder

    Returns
    -------
    Filename or folder name

    """
    if cwd:
        path = os.path.join(cwd, path)
    path = os.path.abspath(path)
    r = glob.glob(re.sub(r'([^:/\\])(?=[/\\]|$)', r'[\1]', path))
    if r:
        path = r[0]
    elif fail_on_missing:
        raise FileExistsError("File or folder matching '%s' not found"%path)
    if cwd:
        path = os.path.relpath(path, cwd)
    if os.path.isdir(path):
        path+="/"
    return path.replace("\\","/")

#     filename = os.path.realpath(filename.replace("\\", "/")).replace("\\", "/")
#     ufn, rest = os.path.splitdrive(filename)
#     ufn += "/"
#     for f in rest[1:].split("/"):
#         f_lst = [f_ for f_ in os.listdir(ufn) if f_.lower() == f.lower()]
#         if len(f_lst) > 1:
#             f_lst = [f_ for f_ in f_lst if f_ == f]
#         elif len(f_lst) == 0:
#             raise IOError("'%s' not found in '%s'" % (f, ufn))
#         else: # one match found
#             ufn = os.path.join(ufn, f_lst[0])
#     return ufn.replace("\\", "/")




class Resource(object):

    def __init__(self, min_cpu, min_free):
        self.min_cpu = min_cpu
        self.min_free = min_free
        self.acquired = 0
        self.lock = threading.Lock()

    def ok2submit(self):
        """Always ok to have min_cpu cpus and ok to have more if there are min_free free cpus"""
        try:
            total, free, user = self.check_resources()
        except:
            return False
        if user < self.min_cpu:
            return True
        elif free > self.min_free:
            return True
        else:
            return False

    def acquire(self):
        with self.lock:
            self.acquired += 1

    def release(self):
        with self.lock:
            self.acquired -= 1


    def update_status(self):
        try:
            self.no_cpu, self.cpu_free, self.no_current_process = self.check_resources()
        except Exception:
            pass


class SSHPBSClusterResource(Resource, SSHClient):
    finished = []
    loglines = {}
    is_executing = []
    
    def __init__(self, host, username, password, port, min_cpu, min_free, init_cmd, wine_cmd, python_cmd):
        Resource.__init__(self, min_cpu, min_free)
        self.init_cmd = init_cmd
        self.wine_cmd = wine_cmd
        self.python_cmd = python_cmd
        self.shared_ssh = SharedSSHClient(host, username, password, port)
        SSHClient.__init__(self, host, username, password, port=port)
        self.lock = threading.Lock()


    def new_ssh_connection(self):
        return SSHClient(self.host, self.username, self.password, self.port)

    def check_resources(self):
        with self.lock:
            try:
                with self:
                    _, output, _ = self.execute('pbsnodes -l all')
                    pbsnodes, nodes = pbswrap.parse_pbsnode_lall(output.split("\n"))

                    _, output, _ = self.execute('qstat -n1')
                    users, host, nodesload = pbswrap.parse_qstat_n1(output.split("\n"), self.host)


                # if the user does not have any jobs, this will not exist
                try:
                    cpu_user = users[self.username]['cpus']
                    cpu_user += users[self.username]['Q']
                except KeyError:
                    cpu_user = 0
                cpu_user = max(cpu_user, self.acquired)
                cpu_free, nodeSum = pbswrap.count_cpus(users, host, pbsnodes)

                return nodeSum['used_cpu'] + cpu_free, cpu_free, cpu_user
            except Exception as e:
                raise EnvironmentError(str(e))


    def jobids(self, jobname_prefix):
            _, output, _ = self.execute('qstat -u %s' % self.username)
            return [l.split()[0].split(".")[0] for l in output.split("\n")[5:] if l.strip() != "" and l.split()[3].startswith("h2l")]

    def stop_pbsjobs(self, jobids):
        if not hasattr(jobids, "len"):
            jobids = list(jobids)
        self.execute("qdel %s" % (" ".join(jobids)))
        
        
   


class LocalResource(Resource):
    def __init__(self, cpu_limit):

        Resource.__init__(self, cpu_limit, multiprocessing.cpu_count())
        #self.process_name = process_name
        self.host = 'Localhost'

    def check_resources(self):
        import psutil
        def name(i):
            try:
                return psutil.Process(i).name()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                return ""

        no_cpu = multiprocessing.cpu_count()
        cpu_free = (1 - psutil.cpu_percent(.5) / 100) * no_cpu
        #no_current_process = len([i for i in psutil.pids() if name(i) == self.process_name.lower()])
        #used = max(self.acquired, no_cpu - cpu_free, no_current_process)
        used = self.acquired
        return no_cpu, cpu_free, used
