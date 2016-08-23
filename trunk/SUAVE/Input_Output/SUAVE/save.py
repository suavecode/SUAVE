#save.py
#
# Created By:   Trent Jan 2015
# Updated: Carlos Ilario, Feb 2016

""" Save a native SUAVE file """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core.Input_Output import save_data

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def save(data,filename):
    """ save data to file """
    
    save_data(data,filename,file_format='pickle')
    