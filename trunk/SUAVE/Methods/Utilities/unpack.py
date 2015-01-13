""" Utilities.py: Mathematical tools and numerical integration methods for ODEs """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Data
from SUAVE.Attributes.Results import Segment

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def unpack(x,problem):

    # grab tf and trim x
    tf = x[-1]; x = x[0:-1]

    # get state data
    indices = range((problem.Npoints-1)*problem.Nstate)
    z = np.reshape(x[indices],(problem.Npoints-1,problem.Nstate),order="F") 

    # get control data if applicable
    if problem.Ncontrol > 0:

        # get control vector
        indices = indices[-1] + range(problem.Npoints*problem.Ncontrol)
        u = np.reshape(x[indices],(problem.Npoints,problem.Ncontrol),order="F") 

        # get throttle
        indices = indices[-1] + range(problem.Npoints)
        eta = x[indices]

        return z, u, eta, tf

    else:
        return z, tf