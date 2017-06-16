# read_optimization_outputs.py
#
# Created:  May 2016, M. Vegh
# Modified:



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------
#  read_optimization_outputs_inputs
# ----------------------------------------------------------------------



def format_input_data(data):

    data_out=[]
  
    for line in data:
       
        line = line.replace('[','')
        line = line.replace(']','')
        line = line.replace(',','')
        #line = line.replace('inputs = ' , '')
        #line = line.split(' objective = ') #split into three arrays
        #line = line.split(' constraints = ' , '')
        #line = line.replace(' constraints =' , '')
        
        
      
        
        line = line.replace('inputs = ' , '')
        line = line.replace(' objective =' , '')
        line = line.replace(' constraints =' , '')
        
        numbers = line.split(' ')
        
        numbers_out=[]
       
        for number in numbers:
            if number != ' ' or number != '\t':
                numbers_out.append(float(number))
            
        data_out.append(numbers_out)
    data_out=np.array(data_out)  #change into numpy array to work with later
    
    return data_out
    
    
def read_gradient_outputs(filename, base_inputs, constraint_inputs):
    #need vector of initial inputs to determine where to separate 
    #inputs from constraints in text file
    try:
        file_in        = open(filename)
        read_success   = 1
    
    except IOError:
        read_success   = 0
    if read_success:
        file_in = open(filename)

        
        data = file_in.readlines()
        file_in.close()
        data = format_input_data(data)
        if len(data)>0:
            #unpack data
            input_vals    = data[:,0:len(base_inputs)]
            obj_end_idx = len(base_inputs)+len(base_inputs)
            obj_values    = data[:,len(base_inputs):obj_end_idx ]
            
            const_end_idx = len(constraint_inputs)*len(base_inputs)+obj_end_idx
            #inputs        = data[:,2:inp_end_idx]
            constraints   = data[:,obj_end_idx:const_end_idx] #cannot use [-1] because it takes second to last value in list
            #now reshape constraints so they are right shape (nconstraint x ninput
            out_cons = []
            con_counter = 0
     
        
            for k in range(len(constraints[:,0])):
                out_cons_2 = [] #based on inputs
                for j in range(len(constraint_inputs)):
                    con = []
                    for i in range(len(input_vals[0,:])):
                        con.append(float(constraints[k,con_counter]))
                        con_counter += 1
            
                    out_cons_2.append(con)
                if j==len(constraint_inputs)-1:
                    con_counter = 0            
                    out_cons.append(out_cons_2)
        else:
            input_vals = 0
            obj_values = 0
            out_cons   = 0
            read_success = 0
    else:
        input_vals = 0
        obj_values = 0
        out_cons   = 0
    
    return input_vals, obj_values , out_cons, read_success
    