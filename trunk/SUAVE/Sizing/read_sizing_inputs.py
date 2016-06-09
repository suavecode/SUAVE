#read_sizing_inputs.py
# Created: Jun 2016, M. Vegh


import numpy as np

def read_sizing_inputs(sizing_loop, opt_inputs):
    file_in        = open(sizing_loop.output_filename)
        
    #read data from previous iterations
    data=file_in.readlines()
    
    if np.length(data)>0:
        file_in.close()
        data=format_input_data(data) #format data so we can work with it
        
        
        data_inputs = data[:, 0:len(opt_inputs)]  #values from optimization problem
        data_outputs= data[:,len(opt_inputs):len(opt_inputs)+len(self.default_y)]  #variables we iterate on in sizing loop
        read_success =1
    else:
        print 'no data to read'
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
    