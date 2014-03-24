
#from SUAVE import DataBase

from copy     import deepcopy
from warnings import warn, simplefilter

# can override the behavior of a warning class
simplefilter('default',Warning)
# available behaviors:
#   error   - turn the warning into an exception
#   ignore  - discard the warning.
#   always  - always emit a warning.
#   default - print the warning the first time it is generated from each location.
#   module  - print the warning the first time it is generated from each module.
#   once    - print the warning the first time it is generated.

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    # setup some inputs
    input1 = {} # Data
    input1['key1'] = 0
    input1['key2'] = 3
    
    input2 = {} # Data
    input2['key1'] = 10
    
    # run example function
    output = example_method(input1,input2)
    
    # will trigger a warning
    trigger_warning()
    
    # will not re-trigger a warning
    trigger_warning()
    
    # will trigger an error
    try:
        trigger_error()
        
    except Exception:
        print "caught a MyException, but continuing"
        
        
    print 'done!'
    

#: def main()    
    
    
# ----------------------------------------------------------------------
#  Example Method
# ----------------------------------------------------------------------

def example_method(input1,input2):
    """ output = example_method(Vehicle,Misson,input1,input2)
    
        Description:
            Short description of function
        
        Assumptions:
            list of assumptions
        
        Inputs:
            input1 - description
            input1.key1 - specific key needed
            input1.key2 - specific key needed
            
            input2 - description
            input2.key3 - specific key needed
        
        Outputs:
            output - dictionary of data
            output.key1 - description
            output.key2 - description
        
        Modifies:
            input1.key3 - description
            (these are the database items modified in place)
        
    """
    
    # unpack inputs
    local1 = input1['key1']
    local2 = input1['key2']
    local3 = input2['key1']
    
    # make a local copy of an input 
    # (can modify this without propogating through pointer)
    local_input1 = deepcopy(input1)
    
    # method body 
    value1 = local1 * local2
    value2 = value1 + local3
    value3 = 'hello world!'
    
    # update data
    input1['key3'] = value2
    
    # pack outputs
    output = {} # Data()
    output['key1'] = value1
    output['key2'] = value3
    
    # return
    return output
    
#: def example_function()


# ----------------------------------------------------------------------
#  Trigger Warning
# ----------------------------------------------------------------------
    
def trigger_warning():
    
    value1 = 20
    
    if value1 > 10:
        warn('this will print once',Warning)
        # syntax: warn( message_string, warning_class )
        # other kinds of warning classes available:
        #   Warning
        #   UserWarning
        #   DepreciationWarning
        #   SyntaxWarning
        #   RuntimeWarning
        #   FutureWarning
        #   ImportWarning
        # and any abstraction of these (see MyWarning below)
        # make sure to import warn (see code top)
        
    elif value1 < 10:
        warn('a subclassed warning',MyWarning)
        
#: def trigger_warning()


# ----------------------------------------------------------------------
#  Trigger Error
# ----------------------------------------------------------------------
        
def trigger_error():
    
    value1 = -100
    
    if value1 > 20:
        raise Exception , 'this will stop python!'
        # syntax: raise exception_class , message_string
        # other kinds of exception classes available:
        #   NotImplementedError
        #   TypeError
        #   ValueError
        #   WindowsError
        # and any abstraction of these (see MyException below)    
        
    elif value1 < 20:
        raise MyException , 'a subclassed exception!'
        
#: def trigger_error()


# ----------------------------------------------------------------------
#  Subclassed Warnings and Exceptions
# ----------------------------------------------------------------------

class MyWarning(Warning):
    pass

class MyException(Exception):
    pass



# ----------------------------------------------------------------------
#  Run Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    main()