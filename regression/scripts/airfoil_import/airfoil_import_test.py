import SUAVE
from SUAVE.Core import Units, Data 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars \
     import import_airfoil_polars

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
    
    airfoil_polar_names    = ['airfoil_polar_1.txt','airfoil_polar_2.txt']    
    airfoil_polar_data =  import_airfoil_polars(airfoil_polar_names) 

    airfoil_geometry_names = ['airfoil_geometry_1.txt','airfoil_geometry_2.txt']    
    airfoil_geometry_data = import_airfoil_geometry (airfoil_geometry_names)
    
    return  

if __name__ == '__main__': 
    main()    
