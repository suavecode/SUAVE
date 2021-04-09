## @ingroup Methods-Aerodynamics-AVL
# write_mass_file.py
# 
# Created: Aug 2019, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Methods.Aerodynamics.AVL.purge_files       import purge_files
from SUAVE.Methods.Aerodynamics.AVL.Data.Settings     import Settings
import numpy as np

## @ingroup Analyses-AVL
def write_mass_file(avl_object,run_conditions):
    """This function writes the translated aircraft mass into text file read 
    by AVL when it is called
    """    
    
    # unpack inputs
    mass_file       = avl_object.settings.filenames.mass_file 
    aircraft        = avl_object.geometry
    
    # Open the mass file after purging if it already exists
    purge_files([mass_file]) 
    mass_file_script  = open(mass_file,'w')

    with open(mass_file,'w') as mass_file_script:        
        """This function writes the header using the template required for the AVL executable to read
        """   
        # mass file template
        base_text = \
'''
#-------------------------------------------------
#  {0}
#
#  Dimensional unit and parameter data.
#  Mass & Inertia breakdown.
#-------------------------------------------------

#  Names and scalings for units to be used for trim and eigenmode calculations.
#  The Lunit and Munit values scale the mass, xyz, and inertia table data below.
#  Lunit value will also scale all lengths and areas in the AVL input file.
Lunit = 1.0 m
Munit = 1.0 kg
Tunit = 1.0 s

#------------------------- 
#  Gravity and density to be used as default values in trim setup.
#  Must be in the units given above.
g   = {1}
rho = {2}

#-------------------------
#  Mass & Inertia breakdown.
#  x y z  is location of item's own CG.
#  Ixx... are item's inertias about item's own CG.
#
#  x,y,z system here must be exactly the same one used in the AVL input file
#     (same orientation, same origin location, same length units)
#
#  mass     x     y     z    Ixx    Iyy    Izz   Component Name
#   
    {3}  {4}  {5}  {6}  {7}  {8}  {9} ! {0}
'''

        # Unpack inputs
        name    = avl_object.geometry._base.tag
        density = run_conditions.freestream.density 
        gravity = run_conditions.freestream.gravity
        
        if aircraft.mass_properties.mass == 0:
            mass = aircraft.mass_properties.max_takeoff
        elif aircraft.mass_properties.max_takeoff == 0:
            mass = aircraft.mass_properties.mass
        else:
            raise AttributeError("Specify Vehicle Mass")
         
        x       = aircraft.mass_properties.center_of_gravity[0][0]
        y       = aircraft.mass_properties.center_of_gravity[0][1]
        z       = aircraft.mass_properties.center_of_gravity[0][2]
        Ixx     = aircraft.mass_properties.moments_of_inertia.tensor[0][0]
        Iyy     = aircraft.mass_properties.moments_of_inertia.tensor[1][1]
        Izz     = aircraft.mass_properties.moments_of_inertia.tensor[2][2]
        
        # Insert inputs into the template
        text = base_text.format(name, gravity , density,mass, x,y,z,Ixx,Iyy,Izz)
        mass_file_script.write(text)  
            
    return