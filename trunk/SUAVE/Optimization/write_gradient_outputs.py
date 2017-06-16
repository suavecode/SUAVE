# write_optimization_outputs.py
#
# Created:  May 2016, M. Vegh
# Modified:


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from helper_functions import get_values, scale_obj_values, scale_const_values

# ----------------------------------------------------------------------
#  write_optimization_outputs
# ----------------------------------------------------------------------



def write_gradient_outputs(x, grad_obj, jac_con, filename):
 
    #unpack gradient problem values
    file = open(filename, 'ab')
    file.write('inputs = ')
    file.write(str(x.tolist()))
    file.write(', objective = ')
    file.write(str(grad_obj.tolist()))
    file.write(', constraints = ')
    file.write(str(jac_con.tolist()))
    file.write('\n') 
    file.close()    

    
    return