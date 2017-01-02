from wetb.hawc2.simulation import Simulation
import os
import sys
from threading import  Thread
class ClusterSimulation(Simulation):
    def __init__(self, modelpath, htcfilename, hawc2exe="HAWC2MB.exe"):
        Simulation.__init__(self, modelpath, htcfilename, hawc2exe=hawc2exe)

        self.simulation_id = [f for f in os.listdir('.') if f.endswith('.in')][0][:-3]
        self.host.simulationThread.low_priority = False
        self.non_blocking_simulation_thread = Thread(target=self.simulate)
        self.start(1)
        self.wait()
        self.is_done = True
        print (self.host.simulationThread.res[1])  # print hawc2 output to stdout
        sys.exit(self.host.simulationThread.res[0])

    def update_status(self, *args, **kwargs):
        Simulation.update_status(self, *args, **kwargs)
        with open("/home/%s/.hawc2launcher/status_%s" % (os.environ['USER'], self.simulation_id), 'w') as fid:
            fid.write (";".join([self.simulation_id, self.status] + [str(getattr(self.logFile, v)) for v in ['status', 'pct', 'remaining_time', 'lastline']]) + "\n")

    def show_status(self):
        pass
