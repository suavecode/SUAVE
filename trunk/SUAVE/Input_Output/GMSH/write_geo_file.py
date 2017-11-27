## @ingroup Input_Output-GMSH
# write_geo_file.py
#
# Created:  Oct 2016, T. MacDonald
# Modified: Jan 2017, T. MacDonald

from SUAVE.Core import Data
import numpy as np

## @ingroup Input_Output-GMSH
def write_geo_file(tag):
    """This reads an .stl file output from OpenVSP using the SUAVE interface and builds
    a .geo file that can be read to GMSH to produce a volume mesh.

    Assumptions:
    The OpenVSP .stl has a farfield surface

    Source:
    N/A

    Inputs:
    tag        <string>  This corresponds to a configuration from SUAVE

    Outputs:
    <tag>.geo

    Properties Used:
    N/A
    """      
    
    # Create .geo file
    # This is essentially a list of commands used to build a volume for meshing
    
    vehicle_nums, farfield_num, symmetry_num = read_keys(tag)
    
    filename = tag + '.geo'
    f = open(filename, mode='w')
    f.write('Merge "' + tag + '.stl";\n')
    
    num_of_vehicle_nums = len(vehicle_nums) # Check how many surfaces are in the vehicle
    vehicle_key_str = '' # Create a string of the numbers indicating the surfaces of the vehicle
    for ii,num in enumerate(vehicle_nums):
        if ii == 0:
            vehicle_key_str = vehicle_key_str + str(int(num))
        else:
            vehicle_key_str = vehicle_key_str + ', ' + str(int(num))
            
    
    mesh_count = num_of_vehicle_nums + 1 # Default is number of vehicle surfaces plus far field
    f.write('Physical Surface("VEHICLE") = {' + vehicle_key_str + '};\n')
    if symmetry_num != -1: # If a symmetry plane has been found
        f.write('Physical Surface("SYMPLANE") = {' + str(int(symmetry_num)) + '};\n')
        mesh_count += 1
    f.write('Physical Surface("FARFIELD") = {' + str(int(farfield_num)) + '};\n')
    
    
    surface_num_str = ''
    
    f.write('Surface Loop(' + str(mesh_count+1) + ') = {' + str(int(farfield_num)) + '};\n')
    surface_num_str = surface_num_str + str(int(mesh_count+1))
    total_count = mesh_count + 1
    if symmetry_num != -1:
        f.write('Surface Loop(' + str(mesh_count+2) + ') = {' + str(int(symmetry_num)) + '};\n')
        surface_num_str = surface_num_str + ', ' + str(mesh_count+2)
        total_count += 1
        
    for num in vehicle_nums:
        f.write('Surface Loop(' + str(total_count + 1) + ') = {' + str(int(num)) + '};\n')
        surface_num_str = surface_num_str + ', ' + str(total_count + 1)
        total_count += 1
    
    f.write('Volume(' + str(total_count) + ') = {' + surface_num_str + '};\n')
    f.close
    
## @ingroup Input_Output-GMSH
def read_keys(tag):
    """This reads the corresponding OpenVSP .key file to check which surfaces
    exist in the .stl file for the vehicle.
    
    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    tag          <string>  This corresponds to a configuration from SUAVE

    Outputs:
    vehicle_nums [-] the numbers corresponding to surfaces on the vehicle
    farfield_num [-] the number corresponding to the farfield, -1 if not available
    symmetry_num [-] the number corresponding to the symmetry plane, -1 if not available

    Properties Used:
    N/A
    """     
    # Read OpenVSP key file to determine which surfaces exist
    
    filename = tag + '.key'
    f = open(filename)
    vehicle_nums = list()
    
    # Set defaults
    farfield_num = -1
    symmetry_num = -1
    
    # Name to identify the far field and the symmetry plane
    farfield_val = 'FarField'
    symmetry_val = 'SymPlane'    
    
    # Get the surface number of each surface
    for ii, line in enumerate(f):
        if ii == 0:
            pass
        else:
            vals = line.split()
            # vals[0] is surface mesh number
            # vals[1] is name of surface mesh
            if vals[1] == farfield_val:
                farfield_num = float(vals[0])
            elif vals[1] == symmetry_val:
                symmetry_num = float(vals[0])
            else:
                vehicle_nums.append(float(vals[0]))

    f.close()
    
    return vehicle_nums, farfield_num, symmetry_num       


if __name__ == '__main__':
    
    tag = 'build_geo_test'
    build_geo(tag)