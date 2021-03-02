## @ingroup Input_Output-OpenVSP
# get_vsp_measurements.py
# 
# Created:  --- 2016, T. MacDonald
# Modified: Aug 2017, T. MacDonald
#           Mar 2018, T. MacDonald
#           Jan 2020, T. MacDonald
#           Feb 2021, T. MacDonald

try:
    import vsp as vsp
except ImportError:
    pass # This allows SUAVE to build without OpenVSP
import numpy as np

## @ingroup Input_Output-OpenVSP
def get_vsp_measurements(filename = 'Unnamed_CompGeom.csv', measurement_type = 'wetted_area'):
    """This calls OpenVSP to compute the wetted areas or volumes of a previously written vehicle.
    
    Assumptions:
    Vehicle must be open in OpenVSP (via recently used vsp_write)
    All components have different tags. Repeated tags are added together under the
    assumption that this represents multiple engines or similar. Values computed from
    repeated tags in this way may need to be divided by the number of entities later 
    before assignment. This is because some analyses may multiply an assigned area
    by number of engines, for example.

    Source:
    N/A

    Inputs:
    None

    Outputs:
    measurement   [m^2 or m^3] - Dictionary with values for each component, 
      with component tags as the keys.

    Properties Used:
    N/A
    """        
    
    if measurement_type == 'wetted_area':
        output_ind = 2
    elif measurement_type == 'wetted_volume':
        output_ind = 4
    else:
        raise NotImplementedError
    
    half_mesh = False # Note that this does not affect the Gmsh/SU2 meshing process
    # it only affects how much area of a component is included in the output
    try:
        file_type = vsp.COMP_GEOM_CSV_TYPE
    except NameError:
        print('VSP import failed')
        return -1

    vsp.SetComputationFileName(file_type, filename)
    vsp.ComputeCompGeom(vsp.SET_ALL, half_mesh, file_type)
    
    f = open(filename)
    
    measurements = dict()
    
    # Extract wetted areas for each component
    for ii, line in enumerate(f):
        if ii == 0:
            pass
        elif line == '\n':
            break
        else:
            vals = line.split(',')
            item_tag = vals[0][:]
            item_w_area = float(vals[output_ind])
            if item_tag in measurements:
                item_w_area = measurements[item_tag] + item_w_area
            measurements[item_tag] = item_w_area
            
    f.close()
    
    return measurements