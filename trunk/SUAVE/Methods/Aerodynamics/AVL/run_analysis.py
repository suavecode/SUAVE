# Tim Momose, October 2014

import os
from SUAVE.Methods.Aerodynamics.AVL.read_results import read_results
from SUAVE.Methods.Aerodynamics.AVL.purge_files  import purge_files


def run_analysis(avl_object):

    call_avl(avl_object)
    results = read_results(avl_object)

    return results


def call_avl(avl_object):

    import sys
    import time
    import subprocess
    import SUAVE.Plugins.VyPy.tools.redirect as redirect

    log_file = avl_object.settings.filenames.log_filename
    err_file = avl_object.settings.filenames.err_filename
    if isinstance(log_file,str):
        purge_files(log_file)
    if isinstance(err_file,str):
        purge_files(err_file)
    avl_call = avl_object.settings.filenames.avl_bin_name
    geometry = avl_object.settings.filenames.features
    in_deck  = avl_object.current_status.deck_file

    with redirect.output(log_file,err_file):

        ctime = time.ctime() # Current date and time stamp
        sys.stdout.write("Log File of System stdout from AVL Run \n{}\n\n".format(ctime))
        sys.stderr.write("Log File of System stderr from AVL Run \n{}\n\n".format(ctime))

        with open(in_deck,'r') as commands:
            avl_run = subprocess.Popen([avl_call,geometry],stdout=sys.stdout,stderr=sys.stderr,stdin=subprocess.PIPE)
            for line in commands:
                avl_run.stdin.write(line)
        avl_run.wait()

        exit_status = avl_run.returncode
        ctime = time.ctime()
        sys.stdout.write("\nProcess finished: {0}\nExit status: {1}\n".format(ctime,exit_status))
        sys.stderr.write("\nProcess finished: {0}\nExit status: {1}\n".format(ctime,exit_status))        

    return exit_status





#=====================#
#     OLD METHODS     #
#=====================#


def build_avl_command(geometry_path,deck_path,avl_bin_path):
    """ builds a command to run an avl analysis on the specified geometry,
    	according to the commands in the input deck. 
    	filenames are referenced to AVL_Callable.settings.filenames.run_folder
    """

    command_skeleton = '{0} {1}<{2}' # {avl_path} {geometry}<{input_deck}
    command = command_skeleton.format(avl_bin_path,geometry_path,deck_path)

    return command


def run_command(command):

    import sys
    import time
    import SUAVE.Plugins.VyPy.tools.redirect as redirect

    with redirect.output('avl_log.txt','stderr.txt'):
        ctime = time.ctime() # Current date and time stamp
        sys.stdout.write("Log File of System stdout from AVL Run \n{}\n\n".format(ctime))
        sys.stderr.write("Log File of System stderr from AVL Run \n{}\n\n".format(ctime))
        exit_status = os.system(command)
        ctime = time.ctime()
        sys.stdout.write("\nProcess finished: {0}\nExit status: {1}\n".format(ctime,exit_status))
        sys.stderr.write("\nProcess finished: {0}\nExit status: {1}\n".format(ctime,exit_status))		

    return exit_status