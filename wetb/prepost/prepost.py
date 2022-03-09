# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:47:32 2015

@author: dave
"""
import os
import copy

class PBSScript(object):
    """
    Generate a PBS script that includes commands such as copying model files
    to the node and copying back the results
    """

    template = """
### Standard Output
#PBS -N [jobname]
#PBS -o [path_pbs_o]
### Standard Error
#PBS -e [path_pbs_e]
#PBS -W umask=[umask]
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=[walltime]
#PBS -lnodes=[lnodes]:ppn=[ppn]
### Queue name
#PBS -q [queue]

### #PBS -a [start_time]
### #PBS -W depend=afterany:[job_id]

### Browse to current working dir
echo ""
cd $PBS_O_WORKDIR
echo "current working dir:"
pwd
echo ""

### ===========================================================================
echo "------------------------------------------------------------------------"
echo "PRELUDE"
echo "------------------------------------------------------------------------"

[prelude]

echo ""
echo "------------------------------------------------------------------------"
echo "EXECUTION"
echo "------------------------------------------------------------------------"

[execution]
### wait for jobs to finish
wait

echo ""
echo "------------------------------------------------------------------------"
echo "CODA"
echo "------------------------------------------------------------------------"

[coda]

echo ""
### ===========================================================================
exit
"""

    def __init__(self):

        # PBS configuration
        self.jobname = 'no_name_job'
        # relative paths with respect to PBS working directory
        self.path_pbs_o = 'pbs_out/dummy.out'
        self.path_pbs_e = 'pbs_out/dummy.err'
        self.path_pbs_i = 'pbs_in/dummy.pbs'
        # absolute path of the PBS working directory
        self.pbsworkdir = './'
        self.umask = '003'
        self.walltime = '00:59:59'
        self.queue = 'workq'
        self.lnodes = '1'
        self.ppn = '1'

        # regarding the job
        # source2node = [ [/abs/src/base/a/b/c, a/b/c.mod] ]
        self.source2node = [] # copy from the source to the node
        # node2source = [ [a/b/d, /abs/src/base/a/b/d.mod] ]
        self.node2source = [] # what to copy back from the node
        self.ziparchives = []
        self.prelude = ''
        self.execution = ''
        self.coda = ''
        self.scratchdir = '/scratch/$USER/$PBS_JOBID/'

    def check_dirs(self):
        """Create the directories of std out, std error and pbs file if they
        do not exist"""
        dnames = set([os.path.dirname(self.path_pbs_o),
                      os.path.dirname(self.path_pbs_e),
                      os.path.dirname(self.path_pbs_i)])
        for dname in dnames:
            if not os.path.exists(os.path.join(self.pbsworkdir, dname)):
                os.makedirs(os.path.join(self.pbsworkdir, dname))

    def create(self, **kwargs):
        """
        path_pbs_e, path_pbs_o, and path_pbs are relative with respect to
        the working dir

        Parameters
        ----------

        template : str, default=PBSSCript.template

        """

        pbs = kwargs.get('template', copy.copy(self.template))
        jobname = kwargs.get('jobname', self.jobname)
        path_pbs_o = kwargs.get('path_pbs_o', self.path_pbs_o)
        path_pbs_e = kwargs.get('path_pbs_e', self.path_pbs_e)
        path_pbs_i = kwargs.get('path_pbs_i', self.path_pbs_i)
        pbsworkdir = kwargs.get('pbsworkdir', self.pbsworkdir)
        umask = kwargs.get('umask', self.umask)
        walltime = kwargs.get('walltime', self.walltime)
        queue = kwargs.get('queue', self.queue)
        lnodes = kwargs.get('lnodes', self.lnodes)
        ppn = kwargs.get('ppn', self.ppn)
#        source2node = kwargs.get('source2node', self.source2node)
#        node2source = kwargs.get('node2source', self.node2source)
#        ziparchives = kwargs.get('ziparchives', self.ziparchives)
        prelude = kwargs.get('prelude', self.prelude)
        execution = kwargs.get('execution', self.execution)
        coda = kwargs.get('coda', self.coda)
        check_dirs = kwargs.get('check_dirs', False)

        if not os.path.isabs(path_pbs_o):
            path_pbs_o = './' + path_pbs_o
        if not os.path.isabs(path_pbs_e):
            path_pbs_e = './' + path_pbs_e

        pbs = pbs.replace('[jobname]', jobname)
        pbs = pbs.replace('[path_pbs_o]', path_pbs_o)
        pbs = pbs.replace('[path_pbs_e]', path_pbs_e)
        pbs = pbs.replace('[umask]', umask)
        pbs = pbs.replace('[walltime]', walltime)
        pbs = pbs.replace('[queue]', queue)
        pbs = pbs.replace('[lnodes]', lnodes)
        pbs = pbs.replace('[ppn]', ppn)

        pbs = pbs.replace('[prelude]', prelude)
        pbs = pbs.replace('[execution]', execution)
        pbs = pbs.replace('[coda]', coda)

        if check_dirs:
            self.check_dirs()

        # write the pbs_script
        with open(os.path.join(pbsworkdir, path_pbs_i), 'w') as f:
            f.write(pbs)


if __name__ == '__main__':
    pass
