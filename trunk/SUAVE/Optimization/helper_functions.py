## @ingroup Optimization
# helper_functions.py
# 
# Created:  May 2015, E. Botero
# Modified: Feb 2015, M. Vegh
#           May 2021, E. Botero 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import numpy as np
from SUAVE.Core import Data
import jax.numpy as jnp

# ----------------------------------------------------------------------        
#   Set_values
# ----------------------------------------------------------------------    

## @ingroup Optimization
def set_values(dictionary,input_dictionary,converted_values,aliases):
    """ This method regresses through a dictionary to set the required values.
        dictionary is the base class that will be modified, input_dictionary is
        the set of inputs to be used, converted_values are values to be set in the
        base dictionary, and finally the aliases which are where in the dictionary
        the names link to

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    dictionary       [Data()]
    input_dictionary [Data()]
    converted_values [Data()]
    aliases          [list of str]

    Outputs:
    None

    Properties Used:
    N/A
    """      
    
    provided_names = list(input_dictionary.keys())
        
    # Correspond aliases to inputs
    pointer = []
    for name in provided_names:
        pointer.append(aliases[name])

    for ii in range(0,len(pointer)):
        pointers = pointer[ii][:]
        if isinstance(pointers,str):
            if '*' in pointers:
                newstrings = find_a_star(dictionary,pointer[ii])
                for jj in range(0,len(newstrings)):
                    localstrs = newstrings[jj].split('.')
                    correctname = '.'.join(localstrs[0:])
                    dictionary.deep_set(correctname,converted_values[ii])
                    
            elif '*' not in pointers:
                localstrs = pointers.split('.')
                correctname = '.'.join(localstrs[0:])
                dictionary.deep_set(correctname,converted_values[ii])              
        else:
            for zz in range(1,len(pointers)+1):
                if '*' in pointers[zz-1]:
                    newstrings = find_a_star(dictionary,pointer[ii][zz-1])
                    for jj in range(0,len(newstrings)):
                        localstrs = newstrings[jj].split('.')
                        correctname = '.'.join(localstrs[0:])
                        dictionary.deep_set(correctname,converted_values[ii])
                        
                elif '*' not in pointers[zz-1]:
                    localstrs = pointers[zz-1].split('.')
                    correctname = '.'.join(localstrs[0:])
                    dictionary.deep_set(correctname,converted_values[ii])            
            
    return dictionary
        
## @ingroup Optimization
def find_a_star(dictionary,string):
    """ Searches through a dictionary looking for an *

    Assumptions:
    There may or may not be an asterisk

    Source:
    N/A

    Inputs:
    dictionary       [Data()]
    input_dictionary [Data()]
    converted_values [Data()]
    aliases          [list of str]

    Outputs:
    newstrings       [list of str]

    Properties Used:
    N/A
    """
    splitstring = string.split('.')
    for ii in range(0,len(splitstring)):
        if '*' in splitstring[ii]:
            if ii==0:
                newkeys = dictionary.keys()
            elif ii !=0:
                strtoeval = 'dictionary.'+'.'.join(splitstring[0:ii])+'.keys()'
                newkeys = list(eval(strtoeval))
            lastindex   = ii
            
    newstrings = []
    for ii in range(0,len(newkeys)):
        newstrings.append('.'.join(splitstring[0:lastindex])+'.'+newkeys[ii]+'.'+'.'.join(splitstring[lastindex+1:]))
        
    return newstrings

## @ingroup Optimization
def scale_input_values(inputs,x):
    """ Scales the values according to the a provided scale

    Assumptions:
    

    Source:
    N/A

    Inputs:
    x                [array]         
    inputs           [list]

    Outputs:
    inputs           [list]

    Properties Used:
    N/A
    """    
    full_inputs     = pack_array(inputs)
    
    provided_scale  = full_inputs[3::5]
    adjusted_inputs = x*provided_scale
    
    full_inputs = full_inputs.at[0::5].set(adjusted_inputs)
    
    inputs      = unpack_array(inputs,full_inputs)
    
    return inputs

def limit_input_values(inputs):
    """ Ensures that the inputs are between the bounds

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    x                [array]         
    inputs           [list]

    Outputs:
    inputs           [list]

    Properties Used:
    N/A
    """      
    
    provided_values = inputs[:,1]
    lower_bounds    = inputs[:,2]
    upper_bounds    = inputs[:,3]
    
    # Fix if the input is too low
    provided_values[provided_values<lower_bounds] = lower_bounds[provided_values<lower_bounds] 
    
    # Fix if the input is too high
    provided_values[provided_values>upper_bounds] = upper_bounds[provided_values>upper_bounds]

    
    return inputs
    

## @ingroup Optimization
def convert_values(inputs): 
    """ Converts an inputs from an optimization into the right units

    Assumptions:
    Always multiply the units by 1!

    Source:
    N/A

    Inputs:
    inputs           [list]

    Outputs:
    converted_values [list of str]

    Properties Used:
    N/A
    """    
    inputs_packed = pack_array(inputs)
    
    provided_values  = inputs_packed[::5]
    
    provided_units   = inputs_packed[3::5]*1.0
    
    converted_values = provided_values*provided_units
    
    return converted_values


# ----------------------------------------------------------------------        
#   Get
# ----------------------------------------------------------------------  

## @ingroup Optimization
def get_values(dictionary,outputs,aliases):
    """ Retrieves values saved in a dictionary 

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    dictionary       [Data()]
    outputs          [Data()]
    aliases          [list of str]

    Outputs:
    values           [float]

    Properties Used:
    N/A
    """     
    
    output_names = list(outputs.keys())
        
    # Correspond aliases to outputs
    pointer = []
    for name in output_names:
        pointer.append(aliases[name])
                
    values = jnp.zeros(len(outputs))
    for ii in range(0,len(outputs)):
        splitstring = pointer[ii].split('.')
        values = values.at[ii].set(eval('dictionary.'+'.'.join(splitstring[0:])).flatten()[0])
    
    return values

## @ingroup Optimization
def get_jacobian_values(dictionary,inputs,outputs,aliases):
    """ Retrieves values saved in a dictionary 

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    dictionary       [Data()]
    outputs          [Data()]
    aliases          [list of str]

    Outputs:
    values           [float]

    Properties Used:
    N/A
    """     
    
    output_names = list(outputs.keys())
    input_names  = list(inputs.keys())
        
    # Correspond aliases to outputs
    pointer_outputs = []
    for name in output_names:
        pointer_outputs.append(aliases[name])
                
    pointer_inputs = []
    for name in input_names:
        pointer_inputs.append(aliases[name])
            
    values = jnp.zeros((len(outputs),len(inputs)))
    for ii in range(0,len(outputs)):
        for jj in range(0,len(inputs)):
            splitstring = pointer_outputs[ii].split('.')+pointer_inputs[jj].split('.')
            try:
                values = values.at[ii,jj].set(eval('dictionary.'+'.'.join(splitstring[0:])))
            except:
                new_string    = 'dictionary.'+'.'.join(splitstring[0:])
                split         = new_string.split('[')
                split[-1]     = split[-1][:-1]
                index         = int(split[-1])
                flatarray     = eval(split[0]).flatten()
                values        = values.at[ii,jj].set(flatarray[index])
    
    return values

## @ingroup Optimization
def scale_obj_values(inputs,x):
    """ Rescales an objective based on Nexus inputs scale

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    inputs          [Data()]
    x               [float]

    Outputs:
    scaled          [float]

    Properties Used:
    N/A
    """     
    
    provided_scale = pack_array(inputs)[0::2]
    provided_units = pack_array(inputs)[1::2]
    
    scaled =  x/(provided_scale*provided_units)
    
    return scaled

## @ingroup Optimization
def scale_const_values(inputs,x):
    """ Rescales constraint values based on Nexus inputs scale

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    inputs          [Data()]
    x               [array]

    Outputs:
    scaled          [array]

    Properties Used:
    N/A
    """        
    
    provided_scale = pack_array(inputs)[2::4]
    scaled =  x/provided_scale
    
    return scaled

## @ingroup Optimization
def scale_const_bnds(inputs):
    """ Rescales constraint bounds based on Nexus inputs scale

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    inputs           [Data()]

    Outputs:
    converted_values [array]

    Properties Used:
    N/A
    """     
    
    provided_bounds = pack_array(inputs)[1::4]
    provided_units  = pack_array(inputs)[3::4]
    
    converted_values = provided_bounds*provided_units
    
    return converted_values

## @ingroup Optimization
def unscale_const_values(inputs,x):
    """ Rescales values based on Nexus inputs scale

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    inputs           [Data()]
    x                [array]

    Outputs:
    scaled           [array]

    Properties Used:
    N/A
    """     
    
    provided_units   = inputs[:,-1]*1.0
    provided_scale = np.array(inputs[:,-2],dtype = float)
    scaled =  x*provided_scale/provided_units
    
    return scaled

def pack_array(dictionary):
    """"""
    
    list_items = list(dictionary.values())
    array      = jnp.array(list_items).flatten()

    return array
    
    
def unpack_array(dictionary, array):
    """"""
    index = 0
    
    for key in list(dictionary.keys()):
        existing_array  = dictionary[key]
        length          = jnp.size(existing_array)
        dictionary[key] = array[index:(index+length)]
        index           = index + length



    return dictionary
    
    