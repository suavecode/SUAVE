## @ingroup Sizing
#write_sizing_residuals.py
# Created : May 2018, M. Vegh



# ----------------------------------------------------------------------
#  write_sizing_residuals
# ----------------------------------------------------------------------


def write_sizing_residuals(sizing_loop, y_save, opt_inputs, residuals):

    file=open(sizing_loop.residual_filename, 'ab')
    file.write(str(y_save.tolist()+opt_inputs.tolist()))
    file.write(' ')
    file.write(str(residuals.tolist()))
    file.write('\n') 
    file.close()
                
    return