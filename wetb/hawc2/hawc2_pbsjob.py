

from wetb.hawc2.htc_file import HTCFile
from wetb.utils.cluster_tools.pbsjob import PBSJob
from wetb.hawc2.log_file import LogInterpreter
import os
import time
from wetb.utils.cluster_tools import pbsjob
class HAWC2PBSJob(PBSJob):

    def __init__(self, host, username, password):
        PBSJob.__init__(self, host, username, password)


    def submit(self, job, cwd, pbs_out_file):
        with open (cwd + job) as fid:
            htcfilename = [l for l in fid if l.startswith('wine')][0].rsplit(" ", 1)[1].strip()
        print (htcfilename)
        htcfile = HTCFile(cwd + htcfilename)

        logfilename = htcfile.simulation.logfile[0]
        self.loginterpreter = LogInterpreter(htcfile.simulation.time_stop[0])
        print (self.loginterpreter.filename)
        PBSJob.submit(self, job, cwd, pbs_out_file)

    def status_monitor(self, update=5):
        i = 0
        self.loglinenumber = 0
        while self.in_queue():
            i += 1
            print (i, self.status, self.get_nodeid())
            if self.status is pbsjob.RUNNING:
                #self.test()
                scratch_log_filename = "/scratch/%s/%s.g-000.risoe.dk/%s" % (self.client.username, self.jobid, self.loginterpreter.filename)
                try:
                    n, out, err = self.client.execute('tail --lines=+%d %s' % (self.loglinenumber, scratch_log_filename))
                    self.loginterpreter.update_status(out)
                    print (self.loginterpreter.status, self.loginterpreter.pct, self.loginterpreter.remaining_time, self.loginterpreter.lastline)
                    with open("status" + self.jobid, 'w') as fid:
                        fid.write(";".join([self.loginterpreter.status, str(self.loginterpreter.pct), str(self.loginterpreter.remaining_time), self.loginterpreter.lastline]))
                    #print (out)
                    self.loglinenumber += out.count ("\n")
                    #print (err)

                except Warning as e:
                    if not "tail: cannot open" in str(e):
                        print (str(e))

            time.sleep(update)
        print (i, self.status, self.get_nodeid())

    def test(self):
        self.log_filename = "logfiles/short.log"
        scratch_log_filename = "/scratch/%s/%s.g-000.risoe.dk/%s" % (self.client.username, self.jobid, self.log_filename)
        print (scratch_log_filename)
        try:
            n, out, err = self.client.execute('tail --lines=+%d %s' % (self.loglinenumber, scratch_log_filename))
            print (n)
            print (out)
            self.loglinenumber += out.count ("\n")
            print (err)

        except Warning as e:
            print (str(e))
