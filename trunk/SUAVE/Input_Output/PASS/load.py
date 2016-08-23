# load.py
#
# Created By:       M. Colonno  4/15/13
# Updated:          Carlos Ilario, Feb 2016

""" SUAVE Methods for IO """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

#import numpy
import re, copy
import xml.sax.handler
import os.path
from warnings import warn

# from SUAVE.Methods import Airfoils
from SUAVE.Core  import Data, Container
from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def load(vehicle,input_file,mass_props_file=""):

    """ ImportFromPASS: Import PASS model

        Inputs:     vehcile             Vehicle class instance (required) 
        Inputs:     pass_file           PASS model file (required)  
        Outputs:    err                 Error status 
    """

    import xml.etree.ElementTree as et
    err = False

    # check for input file (required)
    if os.path.exists(input_file):
        if os.path.isfile(input_file):
            print "Importing " + input_file + " ..."
        else:
            print "Error: model file " + input_file + " exists but appears to be a directory."
            err = True; return err
    else:
        print "Error: model file " + input_file + " was not found; please check inputs."
        err = True; return err

    # open PASS file, import data, close
    data = et.parse(input_file)
    # file = open(input_file)
    # data = xml2obj(file.read())
    # file.close()

    # print some data
    # print data

    return data