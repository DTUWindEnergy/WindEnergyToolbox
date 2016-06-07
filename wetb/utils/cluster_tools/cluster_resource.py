'''
Created on 04/04/2016

@author: MMPE
'''
import multiprocessing
import threading

import psutil

from wetb.utils.cluster_tools import pbswrap
from wetb.utils.cluster_tools.ssh_client import SSHClient, SharedSSHClient
from _collections import deque
import time


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



class SSHPBSClusterResource(Resource, SSHClient):
    def __init__(self, host, username, password, port, min_cpu, min_free):
        Resource.__init__(self, min_cpu, min_free)
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
                    users, host, nodesload = pbswrap.parse_qstat_n1(output.split("\n"))


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
                raise EnvironmentError("check resources failed")

    def jobids(self, jobname_prefix):
            _, output, _ = self.execute('qstat -u %s' % self.username)
            return [l.split()[0].split(".")[0] for l in output.split("\n")[5:] if l.strip() != "" and l.split()[3].startswith("h2l")]

    def stop_pbsjobs(self, jobids):
        if not hasattr(jobids, "len"):
            jobids = list(jobids)
        self.execute("qdel %s" % (" ".join(jobids)))





class LocalResource(Resource):
    def __init__(self, process_name):
        N = max(1, multiprocessing.cpu_count() / 2)
        Resource.__init__(self, N, multiprocessing.cpu_count())
        self.process_name = process_name
        self.host = 'Localhost'

    def check_resources(self):
        def name(i):
            try:
                return psutil.Process(i).name()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                return ""

        no_cpu = multiprocessing.cpu_count()
        cpu_free = (1 - psutil.cpu_percent(.5) / 100) * no_cpu
        no_current_process = len([i for i in psutil.pids() if name(i).lower().startswith(self.process_name.lower())])
        return no_cpu, cpu_free, self.acquired
