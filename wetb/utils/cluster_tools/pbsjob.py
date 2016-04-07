'''
Created on 04/12/2015

@author: mmpe
'''

#import x
import time
from wetb.utils.cluster_tools.ssh_client import SSHClient
import os
import paramiko
import subprocess



NOT_SUBMITTED = "Job not submitted"
PENDING = "Pending"
RUNNING = "Running"
DONE = "Done"
class PBSJob(object):
    _status = NOT_SUBMITTED
    nodeid = None
    def __init__(self, host, username, password):
        self.client = SSHClient(host, username, password, port=22)


    def execute(self, cmd, cwd="./"):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=cwd)
        stdout, stderr = proc.communicate()
        errorcode = proc.returncode
        return errorcode, stdout.decode(), stderr.decode()

    def submit(self, job, cwd, pbs_out_file):
        self.cwd = cwd
        self.pbs_out_file = os.path.relpath(cwd + pbs_out_file).replace("\\", "/")
        self.nodeid = None
        try:
            os.remove (self.pbs_out_file)
        except FileNotFoundError:
            pass
        _, out, _ = self.execute("qsub %s" % job, cwd)
        self.jobid = out.split(".")[0]
        self._status = PENDING

    @property
    def status(self):
        if self._status in [NOT_SUBMITTED, DONE]:
            return self._status

        if self.nodeid is None:
            self.nodeid = self.get_nodeid()
            if self.nodeid is not None:
                self._status = RUNNING

        if self.in_queue() and self.nodeid is None:
            self._status = PENDING
        elif os.path.isfile(self.pbs_out_file):
            self._status = DONE
        return self._status

    def get_nodeid(self):
            errorcode, out, err = self.execute("qstat -f %s | grep exec_host" % self.jobid)
            if errorcode == 0:
                return out.strip().replace("exec_host = ", "").split(".")[0]
            elif errorcode == 1 and out == "":
                return None
            elif errorcode == 153 and 'qstat: Unknown Job Id' in err:
                return None
            else:
                raise Exception(str(errorcode) + out + err)

    def stop(self):
        try:
            self.execute("qdel %s" % self.jobid)
        except Warning as e:
            if 'qdel: Unknown Job Id' in str(e):
                return
            raise e


    def in_queue(self):
        errorcode, out, err = self.execute("qstat %s" % self.jobid)
        if errorcode == 0:
            return True
        elif 'qstat: Unknown Job Id' in str(err):
            return False
        else:
            raise Exception(str(errorcode) + out + err)

