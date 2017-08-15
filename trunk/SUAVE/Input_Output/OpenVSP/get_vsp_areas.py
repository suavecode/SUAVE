## @ingroup Input_Output-OpenVSP
# get_vsp_areas.py
# 
# Created:  --- 2016, T. MacDonald
# Modified: Jan 2017, T. MacDonald

try:
    import vsp_g as vsp
except ImportError:
    pass # This allows SUAVE to build without OpenVSP
import numpy as np

## @ingroup Input_Output-OpenVSP
def get_vsp_areas(tag):
    """This calls OpenVSP to compute the wetted areas of a previously written vehicle.
    
    Assumptions:
    Vehicle must be open in OpenVSP (via recently used vsp_write)

    Source:
    N/A

    Inputs:
    None

    Outputs:
    wetted_areas   [m^2] - Dictionary with wetted areas for each component, with component tags as the keys.

    Properties Used:
    N/A
    """        
    
    half_mesh = False # Note that this does not affect the Gmsh/SU2 meshing process
    # it only affects how much area of a component is included in the output
    try:
        file_type = vsp.COMP_GEOM_CSV_TYPE
    except NameError:
        print 'VSP import failed'
        return -1

    vsp.ComputeCompGeom(vsp.SET_ALL, half_mesh, file_type)
    
    f = open('Unnamed_CompGeom.csv')
    
    wetted_areas = dict()
    
    # Extract wetted areas for each component
    for ii, line in enumerate(f):
        if ii == 0:
            pass
        elif line == '\n':
            break
        else:
            vals = line.split(',')
            item_tag = vals[0][:-1]
            item_w_area = float(vals[2])
            if item_tag in wetted_areas:
                # The tag 'Total' will always conflict if used, since this is a default VSP output
                raise ValueError('Multiple components have identical tags. Wetted areas cannot be assigned.')
            wetted_areas[item_tag] = item_w_area
    
    return wetted_areas