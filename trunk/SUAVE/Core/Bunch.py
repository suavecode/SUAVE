# Bunch.py
#
# Created:  Aug 2015, T. Lukacyzk
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from collections import OrderedDict

# ----------------------------------------------------------------------
#   Bunch 
# ----------------------------------------------------------------------

class Bunch(OrderedDict):
    """ A dictionary that provides attribute-style access.
        This implementation does not extend __getattribute__ to maintain
        performance.
        
    """
        