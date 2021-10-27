## @ingroup Input_Output-VTK
# write_azimuthal_cell_values.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#  


def write_azimuthal_cell_values(f, n_cells, n_a):
    """
    Writes the node locations (x,y,z) for each node around the azimuth of 
    a component.
    
    Inputs:
       f            File to which data is being written   [Unitless]
       n_cells      Total number of cells in component    [Unitless]
       n_a          Total number of azimuthal nodes       [Unitless]
    
    Outputs:                                   
       N/A

    Properties Used:
       N/A 
    
    Assumptions:
       N/A
    
    Source:  
       None
    """
    rlap=0
    for i in range(n_cells): # loop over all nodes in blade
        if i==(n_a-1+n_a*rlap):
            # last airfoil face connects back to first node
            b    = i-(n_a-1)
            c    = i+1
            rlap = rlap+1
        else:
            b = i+1
            c = i+n_a+1
            
        a        = i
        d        = i+n_a
        new_cell = "\n4 "+str(a)+" "+str(b)+" "+str(c)+" "+str(d)
        
        f.write(new_cell)
    return
