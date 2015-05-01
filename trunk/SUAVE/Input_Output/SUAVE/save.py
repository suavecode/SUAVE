# Input_Output.SUAVE.save.py
#
# Created By:   Trent Jan 2015

""" Save a native SUAVE file """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Plugins.VyPy.data import save as vypy_save

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def save(data,filename):
    """ save data to file """
    
    vypy_save(data,filename,file_format='pickle')
    