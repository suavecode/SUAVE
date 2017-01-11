# helper_functions.py
# 
# Created:  May 2015, E. Botero
# Modified: Feb 2015, M. Vegh

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

#from SUAVE.Core import Data, Units
import numpy as np

# ----------------------------------------------------------------------        
#   Set_values
# ----------------------------------------------------------------------    

def set_values(dictionary,input_dictionary,converted_values,aliases):
    
    provided_names = input_dictionary[:,0]
        
    # Correspond aliases to inputs
    pointer = []
    for ii in xrange(0,len(provided_names)):
        for jj in xrange(0,len(aliases)):
            if provided_names[ii] == aliases[jj][0]:
                pointer.append(aliases[jj][1])

    for ii in xrange(0,len(pointer)):
        pointers = pointer[ii][:]
        if isinstance(pointers,str):
            length = 0
            if '*' in pointers:
                newstrings = find_a_star(dictionary,pointer[ii])
                for jj in xrange(0,len(newstrings)):
                    localstrs = newstrings[jj].split('.')
                    correctname = '.'.join(localstrs[0:])
                    dictionary.deep_set(correctname,converted_values[ii])
                    
            elif '*' not in pointers:
                localstrs = pointers.split('.')
                correctname = '.'.join(localstrs[0:])
                dictionary.deep_set(correctname,converted_values[ii])              
        else:
            for zz in xrange(1,len(pointers)+1):
                if '*' in pointers[zz-1]:
                    newstrings = find_a_star(dictionary,pointer[ii][zz-1])
                    for jj in xrange(0,len(newstrings)):
                        localstrs = newstrings[jj].split('.')
                        correctname = '.'.join(localstrs[0:])
                        dictionary.deep_set(correctname,converted_values[ii])
                        
                elif '*' not in pointers[zz-1]:
                    localstrs = pointers[zz-1].split('.')
                    correctname = '.'.join(localstrs[0:])
                    dictionary.deep_set(correctname,converted_values[ii])            
            
    return dictionary
        

def find_a_star(dictionary,string):
    splitstring = string.split('.')
    for ii in xrange(0,len(splitstring)):
        if '*' in splitstring[ii]:
            if ii==0:
                newkeys = dictionary.keys()
            elif ii !=0:
                strtoeval = 'dictionary.'+'.'.join(splitstring[0:ii])+'.keys()'
                newkeys = eval(strtoeval)
            lastindex   = ii
            
    newstrings = []
    for ii in xrange(0,len(newkeys)):
        newstrings.append('.'.join(splitstring[0:lastindex])+'.'+newkeys[ii]+'.'+'.'.join(splitstring[lastindex+1:]))
        
    return newstrings

def scale_input_values(inputs,x):
    
    provided_scale = inputs[:,3]
    inputs[:,1] =  x*provided_scale
    
    return inputs

def convert_values(inputs): 
    
    provided_values  = inputs[:,1] 
    
    # Most important 2 lines of these functions
    provided_units   = inputs[:,-1]*1.0
    inputs[:,-1] = provided_units
    
    converted_values = provided_values*provided_units
    
    return converted_values


# ----------------------------------------------------------------------        
#   Get
# ----------------------------------------------------------------------  

def get_values(dictionary,outputs,aliases):
    
    npoutputs   = np.array(outputs)
    output_names = npoutputs[:,0]
        
    # Correspond aliases to outputs
    pointer = []
    for ii in xrange(0,len(output_names)):
        for jj in xrange(0,len(aliases)):
            if output_names[ii] == aliases[jj][0]:
                pointer.append(aliases[jj][1])    
                
    values = np.zeros(len(outputs))
    for ii in xrange(0,len(outputs)):
        splitstring = pointer[ii].split('.')
        values[ii]  = eval('dictionary.'+'.'.join(splitstring[0:]))
    
    return values

def scale_obj_values(inputs,x):
    
    provided_scale = inputs[:,1]
    provided_units   = inputs[:,-1]*1.0
    inputs[:,-1] = provided_units
    
    scaled =  x/(provided_scale*provided_units)
    
    return scaled

def scale_const_values(inputs,x):
    
    provided_scale = np.array(inputs[:,3],dtype = float)
    scaled =  x/provided_scale
    
    return scaled

def scale_const_bnds(inputs):
    
    provided_bounds = np.array(inputs[:,2],dtype = float)
    
    # Most important 2 lines of these functions
    provided_units   = inputs[:,-1]*1.0
    inputs[:,-1] = provided_units
    
    converted_values = provided_bounds*provided_units
    
    return converted_values

def unscale_const_values(inputs,x):
    
    provided_units   = inputs[:,-1]*1.0
    provided_scale = np.array(inputs[:,3],dtype = float)
    scaled =  x*provided_scale/provided_units
    
    return scaled
