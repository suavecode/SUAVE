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
    
    provided_names = input_dictionary[:,0]
        
    # Correspond aliases to inputs
    pointer = []
    for ii in range(0,len(provided_names)):
        for jj in range(0,len(aliases)):
            if provided_names[ii] == aliases[jj][0]:
                pointer.append(aliases[jj][1])

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
    
    provided_scale = inputs[:,-2]
    inputs[:,1] =  x*provided_scale
    
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
    
    provided_values  = inputs[:,1] 
    
    # Most important 2 lines of these functions
    provided_units   = inputs[:,-1]*1.0
    inputs[:,-1]     = provided_units
    
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
    
    npoutputs   = np.array(outputs)
    output_names = npoutputs[:,0]
        
    # Correspond aliases to outputs
    pointer = []
    for ii in range(0,len(output_names)):
        for jj in range(0,len(aliases)):
            if output_names[ii] == aliases[jj][0]:
                pointer.append(aliases[jj][1])    
                
    values = np.zeros(len(outputs))
    for ii in range(0,len(outputs)):
        splitstring = pointer[ii].split('.')
        values[ii]  = eval('dictionary.'+'.'.join(splitstring[0:]))
    
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
    
    provided_scale = inputs[:,1]
    provided_units = inputs[:,-1]*1.0
    inputs[:,-1]   = provided_units
    
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
    
    provided_scale = np.array(inputs[:,3],dtype = float)
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
    
    provided_bounds = np.array(inputs[:,2],dtype = float)
    
    # Most important 2 lines of these functions
    provided_units  = inputs[:,-1]*1.0
    inputs[:,-1]    = provided_units
    
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