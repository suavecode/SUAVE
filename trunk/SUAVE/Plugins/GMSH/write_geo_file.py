from SUAVE.Core import Data
import numpy as np

def write_geo_file(tag):
    
    vehicle_nums, farfield_num, symmetry_num = read_keys(tag)
    filename = tag + '.geo'
    f = open(filename, mode='w')
    f.write('Merge "' + tag + '.stl";\n')
    
    num_of_vehicle_nums = len(vehicle_nums)
    vehicle_key_str = '' # create a string of the numbers indicating the surface of the vehicle
    for ii,num in enumerate(vehicle_nums):
        if ii == 0:
            vehicle_key_str = vehicle_key_str + str(int(num))
        else:
            vehicle_key_str = vehicle_key_str + ', ' + str(int(num))
            
    mesh_count = num_of_vehicle_nums + 1
    f.write('Physical Surface("VEHICLE") = {' + vehicle_key_str + '};\n')
    if symmetry_num != -1:
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
    
def read_keys(tag):
    
    filename = tag + '.key'
    f = open(filename)
    vehicle_nums = list()
    
    farfield_num = -1
    symmetry_num = -1
    farfield_val = 'FarField'
    symmetry_val = 'SymPlane'    
    
    for ii, line in enumerate(f):
        if ii == 0:
            pass
        else:
            vals = line.split()
            # vals[0] = surface mesh number
            # vals[1] = name of surface mesh
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
    #tag = 'build_geo_test_half'
    build_geo(tag)