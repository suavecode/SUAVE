#read_sizing_inputs.py
# Created: Jun 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ---------------

import numpy as np


# ----------------------------------------------------------------------
#  read_sizing_inputs
# ----------------------------------------------------------------------


def read_sizing_inputs(sizing_loop, opt_inputs):
    try:
        file_in        = open(sizing_loop.output_filename)
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
            data_inputs = data[:, 0:len(opt_inputs)]  #values from optimization problem
            data_outputs= data[:,len(opt_inputs):len(opt_inputs)+len(sizing_loop.default_y)]  #variables we iterate on in sizing loop
        else:
            print 'empty sizing variable file, use default inputs'
            data_inputs  = 0
            data_outputs = 0
            read_success = 0
        
    
    else:
  
        data_inputs = 0
        data_outputs = 0
        
    return data_inputs, data_outputs, read_success
    
def format_input_data(data):

    data_out=[]
    for line in data:
   
        line=line.replace('[','')
        line=line.replace(']','')
        line=line.replace(',','')
    
        numbers = line.split(' ')
        numbers_out=[]
        for number in numbers:
            if number != ' ' or number != '\t':
                numbers_out.append(float(number))
        
        data_out.append(numbers_out)
    data_out=np.array(data_out)  #change into numpy array to work with later

    return data_out
    