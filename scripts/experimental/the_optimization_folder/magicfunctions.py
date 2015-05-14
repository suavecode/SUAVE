# magicfunctions.py
# 
# Created:  May 2015, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
import Vehicles
import Optimize
from SUAVE.Core import Units, Data


# ----------------------------------------------------------------------        
#   Test functions
# ----------------------------------------------------------------------  

def main():
    
    # Make a test dictionary of data
    vehicle = Vehicles.setup()
    
    # Input a test alias structure
    problem = Optimize.setup()
    aliases = problem.aliases
    
    # Setup the input variables
    inputs = problem.inputs
    
    # Test setting a dictionary
    
    # Print the dictionary

    # Setup the output variables
    
    # Test getting a dictionary
    
    
    
    return


# ----------------------------------------------------------------------        
#   Set
# ----------------------------------------------------------------------    

def set_vals(dictionary,inputs,aliases):
    
    # Make a vector of provided aliases
    # Make a vector of values that correspond to provided aliases
    for ii in xrange(0,len(inputs)):
        provided_aliases[ii] = inputs[ii][0]
        provided_values[ii]  = inputs[ii][1]
    
    # Correspond the aliases to the dictionary pointers
       # This requires searching for alias names and saving the pointers to a vec
    for ii in xrange(0,len(inputs)):
        for jj in xrange(0,len(inputs)):
            if provided_aliases[ii] == aliases[jj][0]:
                pointer[ii] = aliases[jj][1]
    
    # Update the values in the dictionary from the pointers
       # Pick a pointer
    all_pointers = pointer.expand()
    for ii in xrange(0,len(all_pointers)):
        
        pass
           
       # Open a dictionary
       # Need to search the keys for the following value 
        # This will need to be a recursive search
        



def deep_set(self,keys,val):
    
    if isinstance(keys,str):
        keys = keys.split('.')
    
    data = self
     
    if len(keys) > 1:
        for k in keys[:-2]:
            data = data[k]
    
    data[ keys[-1] ] = value
    
# ----------------------------------------------------------------------        
#   Get
# ----------------------------------------------------------------------        

def deep_get(self,keys):
    
    if isinstance(keys,str):
        keys = keys.split('.')
    
    data = self
     
    if len(keys) > 1:
        for k in keys[:-1]:
            data = data[k]
    
    value = data[ keys[-1] ]
    
    return value

if __name__ == '__main__': 
    main()    