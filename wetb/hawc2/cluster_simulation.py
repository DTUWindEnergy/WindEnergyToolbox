from wetb.hawc2.simulation import Simulation, SimulationThread
import os
import sys
from threading import  Thread
class ClusterSimulation(Simulation):
    def __init__(self, modelpath, htcfilename, hawc2exe="HAWC2MB.exe"):
        Simulation.__init__(self, modelpath, htcfilename, hawc2exe=hawc2exe)
        self.simulation_id = [f for f in os.listdir('.') if f.endswith('.in')][0][:-3]
        self.simulationThread = SimulationThread(self, False)
        self.thread = Thread(target=self.simulate)
        self.start(1)
        self.wait()
        print (self.simulationThread.res[1])  # print hawc2 output to stdout
        sys.exit(self.simulationThread.res[0])

    def update_status(self, *args, **kwargs):
        Simulation.update_status(self, *args, **kwargs)
        with open("/home/mmpe/.hawc2launcher/status_%s" % self.simulation_id, 'w') as fid:
            fid.write (";".join([self.status] + [str(getattr(self.logFile, v)) for v in ['status', 'pct', 'remaining_time', 'lastline']]) + "\n")

    def show_status(self):
        pass
