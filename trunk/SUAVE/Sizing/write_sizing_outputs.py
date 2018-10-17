## @ingroup Sizing
#write_sizing_inputs.py
# Created : Jun 2016, M. Vegh
# Modified: May 2017, M. Vegh

# ----------------------------------------------------------------------
#  write_sizing_outputs
# ----------------------------------------------------------------------

## @ingroup Sizing
def write_sizing_outputs(sizing_loop, y_save, opt_inputs):
    """
    This function writes out the optimization input variables and the 
    solved sizing inputs at that point
    
    Inputs:
    sizing_loop.
        output_filename
    
    y_save
    opt_inputs

    Outputs:
    None
    
    """
    file=open(sizing_loop.output_filename, 'a')
    if len(opt_inputs) == 1:
        #weird python formatting issue when writing a 1 entry array
        file.write('[')
        file.write(str(opt_inputs[0]))
        file.write(']')
    else:
        file.write(str(opt_inputs))
    file.write(' ')
    file.write(str(y_save.tolist()))
    file.write('\n') 
    file.close()
                
    return