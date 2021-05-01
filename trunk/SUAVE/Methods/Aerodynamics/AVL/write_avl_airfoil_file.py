## @ingroup Methods-Aerodynamics-AVL
# write_avl_airfoil_file.py
# 
# Created:  Apr 2020, M. Clarke 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import os
from SUAVE.Methods.Aerodynamics.AVL.purge_files       import purge_files
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry 

## @ingroup Analyses-AVL
def write_avl_airfoil_file(suave_airfoil_filename):
    """ This function writes the standard airfoil file format from Airfoil tools 
    to avl file format

    Assumptions:
        None
        
    Source: 
   
    Inputs:
        filename
   
    Outputs:
        airfoil.dat
   
    Properties Used:
        N/A
    """       
    

    # unpack avl_inputs
    avl_airfoil_filename =  suave_airfoil_filename.split(".")[-2].split("/")[-1] + '.dat'
    
    # purge file 
    purge_files([avl_airfoil_filename]) 
    
    # read airfoil file header 
    origin  = os.getcwd()
    os_path = os.path.split(origin)[0]
    f_path  = os_path + '/' + suave_airfoil_filename
    f = open(f_path)  
    data_block = f.readlines() 
    f.close()   
    airfoil_name = data_block[0].strip()
    
    # import airfoil coordinates 
    airfoil_geometry_data = import_airfoil_geometry([f_path])
    dim = len(airfoil_geometry_data.x_coordinates[0])
              
    # write file  
    with open(avl_airfoil_filename,'w') as afile:
            afile.write(airfoil_name + "\n")  
            for i in range(dim - 1):
                if i == int(dim/2):
                    pass  
                elif airfoil_geometry_data.y_coordinates[0][i] < 0.0:
                    case_text = '\t' + format(airfoil_geometry_data.x_coordinates[0][i], '.7f')+ "   " + format(airfoil_geometry_data.y_coordinates[0][i], '.7f') + "\n" 
                    afile.write(case_text)
                else:   
                    case_text = '\t' + format(airfoil_geometry_data.x_coordinates[0][i], '.7f')+ "    " + format(airfoil_geometry_data.y_coordinates[0][i], '.7f') + "\n" 
                    afile.write(case_text)
    afile.close()
    return avl_airfoil_filename 
 
