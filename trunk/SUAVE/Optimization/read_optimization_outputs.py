# read_optimization_outputs.py
#
# Created:  May 206, M. Vegh
# Modified:



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

def format_input_data(data):

    data_out=[]
  
    for line in data:
       
        line = line.replace('[','')
        line = line.replace(']','')
        line = line.replace(',','')
        line = line.replace('iteration = ' , '')
        line = line.replace(' objective = ' , '')
        line = line.replace('inputs = ' , '')
        line = line.replace('constraints = ' , '')
        numbers = line.split(' ')
        
        numbers_out=[]
        for number in numbers:
            if number != ' ' or number != '\t':
                
                numbers_out.append(float(number))
            
        data_out.append(numbers_out)
    data_out=np.array(data_out)  #change into numpy array to work with later
    
    return data_out
    
    
def read_optimization_outputs(filename, base_inputs, constraint_inputs):
    #need vector of initial inputs to determine where to separate 
    #inputs from constraints in text file
    file_in = open(filename)
    data = file_in.readlines()
    file_in.close()
    data = format_input_data(data)
    
    #unpack data
    iterations    = data[:,0]
    obj_values    = data[:,1]
    inp_end_idx   = len(base_inputs)+2
    const_end_idx = len(constraint_inputs)+inp_end_idx
    inputs        = data[:,2:inp_end_idx]
    constraints   = data[:,inp_end_idx:const_end_idx] #cannot use [-1] because it takes second to last value in list
    return iterations, obj_values, inputs, constraints
    