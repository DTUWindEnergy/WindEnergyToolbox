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
        raise FileExistsError("File or folder matching '%s' not found" % path)
    if cwd:
        path = os.path.relpath(path, cwd)
    if os.path.isdir(path):
        path += "/"
    return path.replace("\\", "/")


def unix_path_old(filename):
    filename = os.path.realpath(str(filename)).replace("\\", "/")
    ufn, rest = os.path.splitdrive(filename)
    ufn += "/"
    for f in rest[1:].split("/"):
        f_lst = [f_ for f_ in os.listdir(ufn) if f_.lower() == f.lower()]
        if len(f_lst) > 1:
            f_lst = [f_ for f_ in f_lst if f_ == f]
        elif len(f_lst) == 0:
            raise IOError("'%s' not found in '%s'" % (f, ufn))
        else:  # one match found
            ufn = os.path.join(ufn, f_lst[0])
    return ufn.replace("\\", "/")


class Resource(object):

    def __init__(self, min_cpu, min_free):
        self.min_cpu = min_cpu
        self.min_free = min_free
        self.cpu_free = 0
        self.acquired = 0
        self.no_cpu = "?"
        self.used_by_user = 0
        self.resource_lock = threading.Lock()

    def ok2submit(self):
        """Always ok to have min_cpu cpus and ok to have more if there are min_free free cpus"""
        try:
            #print ("ok2submit")
            total, free, user = self.check_resources()
            user = max(user, self.acquired)
            user_max = max((self.min_cpu - user), self.cpu_free - self.min_free)
            #print ("ok2submit", total, free, user, user_max)
            return user_max
        except:
            return False
#         if user < self.min_cpu:
#             return True
#         elif free > self.min_free:
#             return True
#         else:
#             return False

    def acquire(self):
        with self.resource_lock:
            self.acquired += 1

    def release(self):
        with self.resource_lock:
            self.acquired -= 1

    def update_resource_status(self):
        try:
            self.no_cpu, self.cpu_free, self.used_by_user = self.check_resources()
        except Exception:
            pass


class SSHPBSClusterResource(Resource):
    finished = []
    loglines = {}
    is_executing = []

    def __init__(self, sshclient, min_cpu, min_free):
        Resource.__init__(self, min_cpu, min_free)
        self.ssh = sshclient
        self.resource_lock = threading.Lock()

    def glob(self, filepattern, cwd="", recursive=False):
        return self.ssh.glob(filepattern, cwd, recursive)

    @property
    def host(self):
        return self.ssh.host

    @property
    def username(self):
        return self.ssh.username

    def new_ssh_connection(self):
        from wetb.utils.cluster_tools.ssh_client import SSHClient
        return SSHClient(self.host, self.ssh.username, self.ssh.password, self.ssh.port)
        # return self.ssh

    def check_resources(self):
        with self.resource_lock:
            try:
                with self.ssh:
                    _, output, _ = self.ssh.execute('pbsnodes -l all')
                    pbsnodes, nodes = pbswrap.parse_pbsnode_lall(output.split("\n"))

                    _, output, _ = self.ssh.execute('qstat -n1')
                    users, host, nodesload = pbswrap.parse_qstat_n1(output.split("\n"), self.ssh.host)

                # if the user does not have any jobs, this will not exist
                try:
                    cpu_user = users[self.ssh.username]['cpus']
                    cpu_user += users[self.ssh.username]['Q']
                except KeyError:
                    cpu_user = 0
                cpu_user = max(cpu_user, self.acquired)
                cpu_free, nodeSum = pbswrap.count_cpus(users, host, pbsnodes)

                return nodeSum['used_cpu'] + cpu_free, cpu_free, cpu_user
            except Exception as e:
                raise EnvironmentError(str(e))

    def jobids(self, jobname_prefix):
        _, output, _ = self.ssh.execute('qstat -u %s' % self.username)
        return [l.split()[0].split(".")[0] for l in output.split("\n")[5:] if l.strip() != "" and l.split()[3].startswith("h2l")]

    def stop_pbsjobs(self, jobids):
        if not hasattr(jobids, "len"):
            jobids = list(jobids)
        self.ssh.execute("qdel %s" % (" ".join(jobids)))

    def setup_wine(self):
        self.ssh.execute("""rm -f ./config-wine-hawc2.sh &&
wget https://gitlab.windenergy.dtu.dk/toolbox/pbsutils/raw/master/config-wine-hawc2.sh &&
chmod 777 config-wine-hawc2.sh &&
./config-wine-hawc2.sh""")


class LocalResource(Resource):
    def __init__(self, cpu_limit):

        Resource.__init__(self, cpu_limit, multiprocessing.cpu_count())
        #self.process_name = process_name
        self.host = 'Localhost'

    def check_resources(self):
        import psutil
        no_cpu = multiprocessing.cpu_count()
        cpu_free = (1 - psutil.cpu_percent(.1) / 100) * no_cpu
        used = self.acquired
        return no_cpu, cpu_free, used
