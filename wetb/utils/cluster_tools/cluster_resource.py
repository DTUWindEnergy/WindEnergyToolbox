'''
Created on 04/04/2016

@author: MMPE
'''
from wetb.utils.cluster_tools.ssh_client import SSHClient
from wetb.utils.cluster_tools import pbswrap
import multiprocessing
import psutil

class Resource(object):
    pass

class LocalResource(Resource):
    def __init__(self, process_name):
        self.process_name = process_name
        self.no_users = 1
        self.host = 'Localhost'

    def check_resources(self):
        def name(i):
            try:
                return psutil.Process(i).name
            except psutil._error.AccessDenied:
                pass


        no_cpu = multiprocessing.cpu_count()
        cpu_free = (1 - psutil.cpu_percent(.5) / 100) * no_cpu
        no_current_process = len([i for i in psutil.get_pid_list() if name(i) == self.process_name])
        return no_cpu, cpu_free, no_current_process

    def ok2submit(self):

        total, free, user = self.check_resources()
        minimum_cpus = total * 1 / self.no_users
        if user < minimum_cpus and free > 2:
            return True
        else:
            return False



class PBSClusterResource(Resource, SSHClient):
    def __init__(self, host, username, password, port=22):
        SSHClient.__init__(self, host, username, password, port=port)
        self.no_users = 20

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

    def ok2submit(self):

        total, free, user = self.check_resources()
        minimum_cpus = total * 1 / self.no_users
        if user < minimum_cpus:
            return True
        elif free > minimum_cpus * 4:
            return True
        else:
            return False
        pass

