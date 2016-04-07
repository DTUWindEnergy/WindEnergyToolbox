'''
Created on 04/04/2016

@author: MMPE
'''
from wetb.utils.cluster_tools.ssh_client import SSHClient
from wetb.utils.cluster_tools import pbswrap
import multiprocessing
import psutil

class Resource(object):

    def __init__(self, min_cpu, min_free):
        self.min_cpu = min_cpu
        self.min_free = min_free

    def ok2submit(self):
        """Always ok to have min_cpu cpus and ok to have more if there are min_free free cpus"""
        total, free, user = self.check_resources()

        if user < self.min_cpu:
            return True
        elif free > self.min_free:
            return True
        else:
            return False




class SSHPBSClusterResource(Resource, SSHClient):
    def __init__(self, host, username, password, port, min_cpu, min_free):
        Resource.__init__(self, min_cpu, min_free)
        SSHClient.__init__(self, host, username, password, port=port)

    def new_ssh_connection(self):
        return SSHClient(self.host, self.username, self.password, self.port)

    def check_resources(self):
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
        cpu_free, nodeSum = pbswrap.count_cpus(users, host, pbsnodes)

        return nodeSum['used_cpu'] + cpu_free, cpu_free, cpu_user



class LocalResource(Resource):
    def __init__(self, process_name):
        N = max(1, multiprocessing.cpu_count() / 4)
        Resource.__init__(self, N, N)
        self.process_name = process_name
        self.host = 'Localhost'

    def check_resources(self):
        def name(i):
            try:
                return psutil.Process(i).name
            except psutil._error.AccessDenied:
                return ""

        no_cpu = multiprocessing.cpu_count()
        cpu_free = (1 - psutil.cpu_percent(.5) / 100) * no_cpu
        no_current_process = len([i for i in psutil.get_pid_list() if name(i).lower().startswith(self.process_name.lower())])
        return no_cpu, cpu_free, no_current_process
