# folder_setup.py
# Created:  Trent L. June 2015
# Modified: 

'''
# About

This script creates a set of folders and clones SUAVE repo's.


# How to use this script

Open a command line terminal, and change directories to
the folder in which you want to start a new SUAVE Project.
Windows users, download msysgit (msysgit.github.io) and
type "git bash" in the start menu to get a command line.

This is the command:
$ python folder_setup.py

If you would like to use an ssh key to clone, use this command:
$ python folder_setup.py ssh

Windows users, setup a puttykey file to use this feature.

If you have access to the additional developer repo's:
$ python folder_setup.py ssh all
'''

# ----------------------------------------------------------------------
#   Import
# ----------------------------------------------------------------------

import sys, os, shutil

# ----------------------------------------------------------------------
#   Setup
# ----------------------------------------------------------------------

urls = {
    'source'       : 'suavecode/SUAVE.git',
    'tutorials'    : 'suavecode/Tutorials.git',
    'experimental' : 'suavecode/Experimental.git',
    'website'      : 'suavecode/suavecode.github.io.git',
}

argv = sys.argv
ssh = 'ssh'    in argv
dev = 'all'    in argv

def git_clone(url):
    if ssh:
        os.system('git clone git@github.com:' + url)
    else:
        os.system('git clone https://github.com/' + url)
    return


# ----------------------------------------------------------------------
#   Clone 
# ----------------------------------------------------------------------

# Project    
os.mkdir('SUAVE_Project')
os.chdir('SUAVE_Project')

# Source
git_clone(urls['source'])
os.rename('SUAVE','Source')

# Tutorials
git_clone(urls['tutorials'])

# Workspace
os.mkdir('Workspace')

if dev:
    
    # Experimental 
    git_clone(urls['experimental'])
    
    # Website
    git_clone(urls['website'])
    os.rename('suavecode.github.io','Website')

# move this file
shutil.move('../folder_setup.py','.')


    
