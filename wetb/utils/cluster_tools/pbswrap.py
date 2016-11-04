#!/usr/bin/python
#-----------------------------------------------------------------------------
# Author: David Verelst - david.verelst@gmail.com
# Version: 0.10 - 15/03/2010
# Version: 0.11 - 13/04/2010 : adding nodes down, offline to the total
#                                 overview of available nodes
# version: 0.20 - 17/11/2011 : major rewrite, support added for gorm
# version: 0.30 - 19/12/2012 : refactoring, added the node overview
# version: 0.40 - 26/03/2014 : removed Thyra, added support for Jess
#-----------------------------------------------------------------------------

import os
import socket

def print_dashboard(users, host, pbsnodes):

    # print nicely
    # the header
    #    ---------------------------------------------
    #         User  Running   Queued  Waiting    Other
    #    ---------------------------------------------
    #         jber        3        0        0        0

    print
    print ('-' * 54)
    print ('cpus'.rjust(18) + 'nodes'.rjust(9))
    print ('User'.rjust(9) + 'Running'.rjust(9) + 'Running'.rjust(9) \
        + 'Queued'.rjust(9) + 'Waiting'.rjust(9) + 'Other'.rjust(9))
    # nodeSum: overview (summation of all jobs) nodes per user:
    # nodeSum = [running, queued, waiting, other, cpus]
    nodeSum = [0, 0, 0, 0, 0]
    print ('-' * 54)
    # print all values in the table: the nodes used per user
    #userlist = users['users'].keys()
    #userlist.sort()
    for uid in sorted(users):

        # or get the unique nodes the user is on
        try:
            R = len(users[uid]['nodes'])
        except KeyError:
            # means it is not running yet but queued, waiting or otherwise
            R = users[uid]['R']
        Q = users[uid]['Q']
        W = users[uid]['W']
        O = users[uid]['E'] + users[uid]['H'] + users[uid]['T'] \
           + users[uid]['S'] + users[uid]['O'] + users[uid]['C']

        cpus = users[uid]['cpus']
        print (uid.rjust(9) + str(cpus).rjust(9) + str(R).rjust(9) \
            + str(Q).rjust(9) + str(W).rjust(9) + str(O).rjust(9))
        nodeSum[0] += R
        nodeSum[1] += Q
        nodeSum[2] += W
        nodeSum[3] += O
        nodeSum[4] += cpus

    nr_nodes = pbsnodes['nr_nodes']
    down = pbsnodes['down']
    others = pbsnodes['others']
    total_cpu = host['cpu_per_node'] * nr_nodes

    # the summed up for each node status (queued, running,...)
    print ('-' * 54)
    print ('total'.rjust(9) + str(nodeSum[4]).rjust(9) + str(nodeSum[0]).rjust(9) \
        + str(nodeSum[1]).rjust(9) + str(nodeSum[2]).rjust(9)\
        + str(nodeSum[3]).rjust(9))
    print ('-' * 54)
    print ('free'.rjust(9) + str(total_cpu - nodeSum[4]).rjust(9) \
        + str(nr_nodes - nodeSum[0] - others - down).rjust(9))
    print ('down'.rjust(9) + str(down).rjust(18))
    print ('-' * 54)
    print


def print_node_loading(users, host, nodes, nodesload):
    """
    Give an overview of how each node is loaded
    """

    if len(host) < 1:
        print ('It is very quit, nobody is working on the cluster.')
        return

    hostname = host['name']
    cpunode = host['cpu_per_node']

    print
    # print a header
    if hostname == 'gorm':
        print ('-' * 79)
        header = '|'.join([str(k).center(5) for k in range(1, 13, 1)]) + '|'
        print ('id'.center(5), header)
        print ('-' * 79)
    elif hostname == 'jess':
        print ('-' * 126)
        header = '|'.join([str(k).center(5) for k in range(1, 21, 1)]) + '|'
        print ('id'.center(5), header)
        print ('-' * 126)

    # print who is using the nodes
    for node in sorted(nodes):
        status = nodes[node]
        # now we have a list of user on this node
        try:
            users = sorted(nodesload[node])
            for kk in range(len(users), cpunode):
                users.append('')
            # limit uid names to 5 characters
            printlist = '|'.join([k[:5].center(5) for k in users]) + '|'
        # if it doesn't exist in the nodesload, just print the status
        except KeyError:
            printlist = status.center(5)

        print (node, printlist)

    # print a header
    if hostname == 'gorm':
        print ('-' * 79)
        print ('id'.center(5), header)
        print ('-' * 79)
    elif hostname == 'jess':
        print ('-' * 126)
        print ('id'.center(5), header)
        print ('-' * 126)
    #print


def parse_pbsnode_lall(output):
    """Parse output of pbsnodes -l all
    """
    # read the qstat output
    frees, exclusives, others, down = 0, 0, 0, 0
    nr_nodes = 0

    nodes = {}

    for k in output:
        if len(k) > 2:
            line = k.split()
            status = line[1]
            node = line[0].split('.')[0]

            if node.startswith('v-'):
                #host = 'valde'
                # uglye monkey patch: ignore any valde nodes
                continue

            #elif node.startswith('g-'):
                #host = 'gorm'
            #elif node.startswith('th-'):
                #host = 'thyra'

            if status == 'free':
                frees += 1
                nr_nodes += 1
            elif status == 'job-exclusive':
                exclusives += 1
                nr_nodes += 1
            elif status == 'down,offline':
                down += 1
            elif status == 'offline':
                down += 1
            elif status == 'down':
                down += 1
            else:
                others += 1

            nodes[node] = status

    #check = frees + exclusives + down + others

    pbsnodes = {'frees' : frees, 'nr_nodes' : nr_nodes, 'others' : others,
               'exclusives' : exclusives, 'down' : down}

    return pbsnodes, nodes


def parse_qstat_n1(output, hostname=None):
    """
    Parse the output of qstat -n1
    """

    # for global node usage, keep track of how many processes are running on
    # each of the nodes
    nodesload = {}
    # store it all in dictionaries
    host = {}
    users = {}
    # get the hostname
    if hostname is None:
        hostname = socket.gethostname()
    if 'jess' in hostname:
        host['name'] = 'jess'
        host['cpu_per_node'] = 20
    else:
        host['name'] = 'gorm'
        host['cpu_per_node'] = 12

    # take the available nodes in nr_nodes: it excludes the ones
    # who are down
    #queue['_total_cpu_'] = cpu_node*nr_nodes

    for line in output[5:]:
        if len(line.strip()) == 0:
            continue
        items = line.split()
        queue = items[2]

        # uglye monkey patch: ignore any valde nodes
        if queue == 'valdeq':
            continue

        jobid = items[0]
        # valid line starts with the jobid, which is an int
        jobid = jobid.split('.')[0]
        userid = items[1]
        # nr nodes used by the job
        job_nodes = int(items[5])
        # status of the job
        job_status = items[9]
        # is the user already in the queue dict?
        try:
            users[userid]['jobs'].append(jobid)
            users[userid][job_status] += job_nodes
        # if the user wasn't added before, create the sub dictionaries
        except KeyError:
            # initialise the users dictionary and job list
            users[userid] = dict()
            users[userid]['C'] = 0
            users[userid]['E'] = 0
            users[userid]['H'] = 0
            users[userid]['Q'] = 0
            users[userid]['R'] = 0
            users[userid]['T'] = 0
            users[userid]['W'] = 0
            users[userid]['S'] = 0
            users[userid]['O'] = 0
            users[userid]['cpus'] = 0
            users[userid]['jobs'] = []
            users[userid]['nodes'] = set()
            # set the values
            users[userid]['jobs'].append(jobid)
            users[userid][job_status] += job_nodes

        if job_status == 'R':
            # each occurance of the node name is seprated by a + and
            # indicates a process running on a CPU of that node
            nodes = items[11].split('+')
            # TODO: take into account cpu number for jess: j-304/5
            # on jess, the cpu number of the node is indicated, ignore for now
            # FIXME: host name can be jess or j-
            if host['name'].startswith('jess'):
                for i, node in enumerate(nodes):
                    nodes[i] = node.split('/')[0]
            # keep track of the number of processes the user running
            users[userid]['cpus'] += len(nodes)
            # for the number of used nodes, keep track of the unique nodes used
            users[userid]['nodes'].update(set(nodes))
            # remember all the nodes the user is on in a dictionary
            for node in nodes:
                try:
                    nodesload[node].append(userid)
                except KeyError:
                    nodesload[node] = [userid]

    return users, host, nodesload

# FIXME: counts diffferent compared to launch.py....
def count_cpus(users, host, pbsnodes):
    """
    See how many cpu's are actually free
    """
    nodeSum = {'R':0, 'Q':0, 'W':0, 'O':0, 'used_cpu':0, 'H':0}

    for uid in users:

        # or get the unique nodes the user is on
        try:
            nodeSum['R'] = len(users[uid]['nodes'])
        except KeyError:
            # means it is not running yet but queued, waiting or otherwise
            nodeSum['R'] = users[uid]['R']

        nodeSum['Q'] += users[uid]['Q']
        nodeSum['W'] += users[uid]['W']
        nodeSum['H'] += users[uid]['H']
        nodeSum['used_cpu'] += users[uid]['cpus']
        nodeSum['O'] += users[uid]['E'] + users[uid]['T'] + users[uid]['S'] \
                      + users[uid]['O'] + users[uid]['C']

    # free cpus
    down_cpu = host['cpu_per_node'] * pbsnodes['down']
    total_cpu = host['cpu_per_node'] * pbsnodes['nr_nodes']
    cpu_free = total_cpu - down_cpu - nodeSum['used_cpu']

    return cpu_free, nodeSum


PBS_TEMP = """
### Standard Output
#PBS -N [jobname]
#PBS -o ./[pbs_out_file]
### Standard Error
#PBS -e ./[pbs_err_file]
#PBS -W umask=003
### Maximum wallclock time format HOURS:MINUTES:SECONDS
#PBS -l walltime=[walltime]
#PBS -lnodes=[lnodes]:ppn=[ppn]
### Queue name
#PBS -q [queue]
### Browse to current working dir
cd $PBS_O_WORKDIR
pwd
### ===========================================================================
### set environment variables, activate python virtual environment, execute
[commands]

### ===========================================================================
### wait for jobs to finish
wait
exit
"""
# TODO: this is very similar compared to what happens in qsub-wrap
def create_input(walltime='00:59:59', queue='xpresq', pbs_in='pbs_in/', ppn=1,
                 pbs_out='pbs_out/', jobname=None, commands=None, lnodes=1):
    """
    Create a PBS script for a command. Optionally, define a python environment.
    """

    pbs_err_file = os.path.join(pbs_out, jobname + '.err')
    pbs_out_file = os.path.join(pbs_out, jobname + '.out')
    pbs_in_file = os.path.join(pbs_in, jobname + '.pbswrap')

    pbs_script = PBS_TEMP
    pbs_script = pbs_script.replace('[jobname]', jobname)
    pbs_script = pbs_script.replace('[pbs_out_file]', pbs_out_file)
    pbs_script = pbs_script.replace('[pbs_err_file]', pbs_err_file)
    pbs_script = pbs_script.replace('[walltime]', walltime)
    pbs_script = pbs_script.replace('[lnodes]', str(lnodes))
    pbs_script = pbs_script.replace('[ppn]', str(ppn))
    pbs_script = pbs_script.replace('[queue]', queue)
    pbs_script = pbs_script.replace('[commands]', commands)

    print ('following commands will be executed on the cluster:')
    print ('%s' % (commands))

    # make sure a pbs_in and pbs_out directory exists
    if not os.path.exists(pbs_in):
        os.makedirs(pbs_in)
    if not os.path.exists(pbs_out):
        os.makedirs(pbs_out)

    # write the pbs_script
    FILE = open(pbs_in_file, 'w')
    FILE.write(pbs_script)
    FILE.close()

    return pbs_in_file


if __name__ == '__main__':

    pass
