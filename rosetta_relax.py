#!/usr/bin/env python3

from __future__ import print_function
import os
import subprocess
import datetime

d1 = datetime.datetime.now()

'''
This code is used to relax 3D structure of wild-type proteins
'''
use_multiprocessing = True
if use_multiprocessing:
    import multiprocessing
    max_cpus = 40 # Number of cpus core you can use

def wild_relax(start_struct, pdb_chain):
    rosetta_relax_script_path = os.path.expanduser('./relax.mpi.linuxgccrelease')
    output_directory = os.path.expanduser('./relaxed_structure')
    input_pdb_path = os.path.expanduser('./S2648')
    start_struct_path = os.path.join(input_pdb_path, start_struct)
   
    # rosetta_cmd
    wild_relax_script_arg = [
        os.path.abspath(rosetta_relax_script_path),
        '-in:file:s', os.path.abspath(start_struct_path),
        '-in:file:fullatom',
        '-relax:constrain_relax_to_start_coords',
        '-out:no_nstruct_label', '-relax:ramp_constraints false',
        '-default_max_cycles 200',
        '-out:file:scorefile', os.path.join(pdb_chain + '_relaxed.sc'),
        '-out:suffix', '_relaxed',
    ]

    log_path = os.path.join(output_directory, 'rosetta.out')

    print( 'Running Rosetta with args:' )
    print( ' '.join(wild_relax_script_arg) )
    print( 'Output logged to:', os.path.abspath(log_path) )
    print()

    outfile = open(log_path, 'w')
    process = subprocess.Popen(wild_relax_script_arg , stdout=outfile, stderr=subprocess.STDOUT, close_fds = True, cwd = output_directory)
    returncode = process.wait()
    outfile.close()

if __name__=='__main__':

    case = []
    input_pdb_path = os.path.expanduser('./S2648') # input pdb folder

    for start_struct in os.listdir(input_pdb_path):
        pdb_chain = os.path.splitext(start_struct)[0] #Get the file name without extension
        case.append((start_struct, pdb_chain))

    # Apply multiprocessing package to run multiple rosetta script simultaneously
    if use_multiprocessing:
        pool = multiprocessing.Pool( processes = min(max_cpus, multiprocessing.cpu_count()) )

    for args in case:
        if use_multiprocessing:
            pool.apply_async( wild_relax, args = args )
        else:
            wild_relax(*args)

    if use_multiprocessing:
        pool.close()
        pool.join()

    d2=datetime.datetime.now()
    print(d2-d1)

