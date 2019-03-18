## @ingroup Sizing
#write_sizing_residuals.py
# Created : May 2018, M. Vegh

# ----------------------------------------------------------------------
#  write_sizing_residuals
# ----------------------------------------------------------------------

#write_sizing_residuals.py
def write_sizing_residuals(sizing_loop, y_save, opt_inputs, residuals):
    """
    This function writes out the residual values at each sizing iteration
    
    Inputs:
    sizing_loop.
        residual_filename
    
    
    y_save     [array]
    opt_inputs [array]
    residuals  [array]
    
    Outputs:
    None
    
    """
    
    file=open(sizing_loop.residual_filename, 'a')
    file.write(str(y_save.tolist()+opt_inputs.tolist()))
    file.write(' ')
    file.write(str(residuals.tolist()))
    file.write('\n') 
    file.close()
                
    return