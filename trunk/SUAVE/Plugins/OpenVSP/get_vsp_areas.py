import vsp_g as vsp
import numpy as np

def get_vsp_areas(tag):
    
    half_mesh = False # Note that this does not affect the Gmsh/SU2 meshing process
    # it only affects how much area of a component is included in the output
    file_type = vsp.COMP_GEOM_CSV_TYPE

    vsp.ComputeCompGeom(vsp.SET_ALL, half_mesh, file_type)
    
    f = open('Unnamed_CompGeom.csv')
    
    wetted_areas = dict()
    
    # Extract wetted areas for each component
    for ii, line in enumerate(f):
        if ii == 0:
            pass
        else:
            vals = line.split(',')
            item_tag = vals[0][:-1]
            item_w_area = float(vals[2])
            if item_tag in wetted_areas:
                # The tag 'Total' will always conflict if used, since this is a default VSP output
                raise ValueError('Multiple components have identical tags. Wetted areas cannot be assigned.')
            wetted_areas[item_tag] = item_w_area
    
    return wetted_areas