# release_export.py
# Created:  Trent L. June 2015
# Modified: 

'''
# About
Exports a SUAVE Release to Releases/Release_*

# How to Use

Open a command line in the SUAVE_Project folder.
Run this command -

$ python release_export.py <branch> <version>

Example:
$ python release_export.py master 0.2.0
'''

# ----------------------------------------------------------------------
#   Import
# ----------------------------------------------------------------------

import sys, os, shutil

# ----------------------------------------------------------------------
#   Setup
# ----------------------------------------------------------------------

def git_export(branch,folder):
    os.system('git checkout %s' % branch)
    os.system('git archive %s | tar -x -C %s' % (branch,folder))

if len(sys.argv) > 1:
    branch  = sys.argv[1]
    version = sys.argv[2]
else:
    branch = 'develop'
    version = 'testing'
    
folder = 'Releases/Release_'+version

# ----------------------------------------------------------------------
#   Export
# ----------------------------------------------------------------------

os.makedirs(folder)
os.makedirs(folder+'/Source')
os.makedirs(folder+'/Tutorials')
os.makedirs(folder+'/Workspace')

os.chdir('Source')
git_export(branch,'../'+folder+'/Source')
os.chdir('..')

os.chdir('Tutorials')
git_export(branch,'../'+folder+'/Tutorials')    
os.chdir('..')
    
