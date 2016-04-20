'''
Created on 04/12/2015

@author: mmpe
'''
import os
from wetb.utils.cluster_tools.ssh_client import SSHClient

NOT_SUBMITTED = "Job not submitted"
PENDING = "Pending"
RUNNING = "Running"
DONE = "Done"


class SSHPBSJob(SSHClient):
    _status = NOT_SUBMITTED
    nodeid = None
    jobid = None


    def __init__(self, host, username, password, port=22):
        SSHClient.__init__(self, host, username, password, port=port)

    def submit(self, job, cwd, pbs_out_file):
        self.cwd = cwd
        self.pbs_out_file = os.path.relpath(cwd + pbs_out_file).replace("\\", "/")
        self.nodeid = None
        #self.execute()

        cmds = ['rm -f %s' % self.pbs_out_file]
        if cwd != "":
            cmds.append("cd %s" % cwd)
        cmds.append("qsub %s" % job)
        _, out, _ = self.execute(";".join(cmds))
        self.jobid = out.split(".")[0]
        self._status = PENDING

    @property
    def status(self):
        if self._status in [NOT_SUBMITTED, DONE]:
            return self._status
        with self:
            if self.is_executing():
                self._status = RUNNING
            elif self.file_exists(self.pbs_out_file):
                self._status = DONE
                self.jobid = None
        return self._status

    def get_nodeid(self):
        try:
            _, out, _ = self.execute("qstat -f %s | grep exec_host" % self.jobid)
            return out.strip().replace("exec_host = ", "").split(".")[0]
        except Warning as e:
            if 'qstat: Unknown Job Id' in str(e):
                return None
            #raise e

    def stop(self):
        if self.jobid:
            try:
                self.execute("qdel %s" % self.jobid)
            except Warning as e:
                if 'qdel: Unknown Job Id' in str(e):
                    return
                raise e


    def is_executing(self):
        try:
            self.execute("qstat %s" % self.jobid)
            return True
        except Warning as e:
            if 'qstat: Unknown Job Id' in str(e):
                return False
            raise e