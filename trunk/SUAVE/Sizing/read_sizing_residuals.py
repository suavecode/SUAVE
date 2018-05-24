#read_sizing_inputs.py
# Created: Jun 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ---------------

import numpy as np
from read_sizing_inputs import format_input_data

# ----------------------------------------------------------------------
#  read_sizing_residuals
# ----------------------------------------------------------------------


def read_sizing_residuals(sizing_loop, opt_inputs):
    try:
        file_in        = open(sizing_loop.residual_filename)
        read_success   = 1
    
    except IOError:
        print 'no data to read, use default values'
        read_success   = 0
        
        
    #read data from previous iterations
    if  read_success==1:
        data=file_in.readlines()
        file_in.close()
        data=format_input_data(data) #format data so we can work with it
        file_in.close()
        
        if len(data)>0:
            default_y    = sizing_loop.default_y
            data_inputs  = data[:, 0:len(default_y)+len(opt_inputs)]  #values from optimization problem
            data_outputs = data[:,len(default_y)+ len(opt_inputs):len(opt_inputs)+len(default_y)*2]  #sizing loop residual values
        else:
            print 'empty sizing variable file, use default inputs'
            data_inputs  = 0
            data_outputs = 0
            read_success = 0
        
    
    else:

        data_inputs  = 0
        data_outputs = 0

    return data_inputs, data_outputs, read_success
    
