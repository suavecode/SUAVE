# Input_Output.SUAVE.load.py
#
# Created By:   Trent Jan 2015

""" Load a native SUAVE file """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Plugins.VyPy.data import load as vypy_load


# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def load(filename):
    """ load data from file """
    
    data = vypy_load(filename,file_format='pickle')
    
    return data