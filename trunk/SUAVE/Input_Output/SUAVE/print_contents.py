#print_contents.py
#
# Created By:   Luke Kulik, Aug 2016
# Updated:

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data
import numpy as np

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------


def print_contents(dict, indent=0):
    """ Recursively print contents of a data dictionary (based on http://stackoverflow.com/a/3229493) """

    for key in dict.keys():
        print '\t' * indent + str(key)
        v = dict[key]
        if isinstance(v, Data):
            print_contents(v, indent + 1)
        elif isinstance(v, (np.ndarray, np.generic)):
            continue
        else:
            print '\t' * (indent + 1) + str(v)

    return
