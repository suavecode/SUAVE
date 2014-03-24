""" compute_alpha.py: compute angle of attack  """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ---------------------------------------------------------------------
def compute_alpha(results):

    for i in range(len(results.Segments)):
        results.Segments[i].alpha = results.Segments[i].gamma - results.Segments[i].psi

    return