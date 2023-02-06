# folder_setup.py
# Created:  Trent L. June 2015
# Modified: 

'''
# About

This script creates a set of folders and clones MARC repo's.


# How to use this script

Open a command line terminal, and change directories to
the folder in which you want to start a new MARC Project.
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
    'source'       : 'mitleads/MARC.git', 
    'website'      : 'https://www.matthewaclarke.com/marc',
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
os.mkdir('MARC_Project')
os.chdir('MARC_Project')

# Source
git_clone(urls['source'])
os.rename('MARC','Source') 

# Workspace
os.mkdir('Workspace')

# move this file
shutil.move('../folder_setup.py','.')


    
