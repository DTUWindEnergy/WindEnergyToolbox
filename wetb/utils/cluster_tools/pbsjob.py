'''
Created on 04/12/2015

@author: mmpe
'''
import os
import io
from wetb.utils.cluster_tools.pbsfile import PBSFile
from wetb.utils.cluster_tools.os_path import relpath

NOT_SUBMITTED = "Job not submitted"
PENDING = "Pending"
RUNNING = "Running"
DONE = "Done"


def pjoin(*args):
    return os.path.join(*args).replace('\\', '/')


class SSHPBSJob(object):
    _status = NOT_SUBMITTED
    nodeid = None
    jobid = None

    def __init__(self, sshClient):
        self.ssh = sshClient

    def submit(self, pbsfile, cwd=None, pbs_out_file=None):
        self.cwd = cwd
        self.nodeid = None
        if isinstance(pbsfile, PBSFile):
            f = io.StringIO(str(pbsfile))
            f.seek(0)
            pbs_filename = pjoin(cwd, pbsfile.filename)
            self.ssh.upload(f, pbs_filename)
            self.pbs_out_file = pjoin(cwd, pbsfile.stdout_filename)
            cwd = pbsfile.workdir
            pbsfile = pbsfile.filename
        else:
            self.pbs_out_file = os.path.relpath(cwd + pbs_out_file).replace("\\", "/")
        cmds = ['rm -f %s' % self.pbs_out_file]
        if cwd != "":
            cmds.append("cd %s" % cwd)
        cmds.append("qsub %s" % pbsfile)
        _, out, _ = self.ssh.execute(";".join(cmds))
        self.jobid = out.split(".")[0]
        self._status = PENDING

    @property
    def status(self):
        if self._status in [NOT_SUBMITTED, DONE]:
            return self._status
        with self.ssh:
            if self.is_executing():
                self._status = RUNNING
            elif self.ssh.file_exists(self.pbs_out_file):
                self._status = DONE
                self.jobid = None
        return self._status

    def get_nodeid(self):
        try:
            _, out, _ = self.ssh.execute("qstat -f %s | grep exec_host" % self.jobid)
            return out.strip().replace("exec_host = ", "").split(".")[0]
        except Warning as e:
            if 'qstat: Unknown Job Id' in str(e):
                return None
            #raise e

    def stop(self):
        if self.jobid:
            try:
                self.ssh.execute("qdel %s" % self.jobid)
            except Warning as e:
                if 'qdel: Unknown Job Id' in str(e):
                    return
                raise e

    def is_executing(self):
        try:
            self.ssh.execute("qstat %s" % self.jobid)
            return True
        except Warning as e:
            if 'qstat: Unknown Job Id' in str(e):
                return False
            raise e
