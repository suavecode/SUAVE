#write_sizing_inputs.py
# Created: Jun 2016, M. Vegh


# ----------------------------------------------------------------------
#  Imports
# ---------------

import numpy as np


# ----------------------------------------------------------------------
#  write_sizing_residuals
# ----------------------------------------------------------------------


def write_sizing_residuals(sizing_loop, y_save, opt_inputs, residuals):

    file=open(sizing_loop.residual_filename, 'ab')
    print 'opt_inputs = ', opt_inputs
    print 'y_save.tolist()+opt_inputs.tolist() = ', y_save.tolist()+opt_inputs.tolist()
    file.write(str(y_save.tolist()+opt_inputs.tolist()))
    file.write(' ')
    file.write(str(residuals.tolist()))
    file.write('\n') 
    file.close()
                
    return