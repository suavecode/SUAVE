## @ingroup Input_Output-OpenVSP
# write_vsp_propeller.py
# 
# Created: Feb 2021, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from SUAVE.Core import Units, Data
from SUAVE.Methods.Aerodynamics.AVL.purge_files import purge_files 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
import numpy as np
import shutil 

## @ingroup Input_Output-OpenVSP
def write_vsp_propeller(vsp_bem_filename,propeller):
    """   This functions write a .bem file for OpenVSP

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        OpenVSP .bem filename
        SUAVE Propeller Data Structure 

    Outputs:
        OpenVSP .bem file 

    Properties Used:
        N/A
    """    
    
    # unpack inputs 
    # Open the vsp_bem file after purging if it already exists
    purge_files([vsp_bem_filename]) 
    vsp_bem = open(vsp_bem_filename,'w')

    with open(vsp_bem_filename,'w') as vsp_bem:
        make_header_text(vsp_bem, propeller)
         
        make_section_text(vsp_bem,propeller)
        
        make_airfoil_text(vsp_bem,propeller)  
            
    return

## @ingroup Input_Output-OpenVSP
def make_header_text(vsp_bem,prop):  
    """This function writes the header of the OpenVSP .bem file

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        vsp_bem - OpenVSP .bem file 
        prop    - SUAVE propeller data structure 
        
    Outputs:
        NA                

    Properties Used:
        N/A
    """      
    header_base = \
'''...{0}... 
Num_Sections: {1}
Num_Blade: {2}
Diameter: {3}
Beta 3/4 (deg): {4}
Feather (deg): 0.00000000
Pre_Cone (deg): 0.00000000
Center: {5}, {6}, {7}
Normal: {8}, {9}, {10}

''' 
    # Unpack inputs 
    name     = prop.tag
    N        = len(prop.radius_distribution)
    B        = prop.number_of_blades
    D        = prop.tip_radius*2
    beta     = np.round(prop.twist_distribution/Units.degrees,5)
    X        = prop.origin[0][0]
    Y        = prop.origin[0][1]    
    Z        = prop.origin[0][2]    
    Xn       = np.round(np.cos(np.pi- prop.thrust_angle ),5)
    Yn       = 0.0000
    Zn       = np.round(np.sin(np.pi- prop.thrust_angle ),5)
    
    beta_3_4 = beta[round(N*0.75)] 
    
    # Insert inputs into the template
    header_text = header_base.format(name,N,B,D,beta_3_4,X,Y,Z,Xn,Yn,Zn) 
    vsp_bem.write(header_text)    
    
    return   

## @ingroup Input_Output-OpenVSP
def make_section_text(vsp_bem,prop):
    """This function writes the sectional information of the propeller 

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        vsp_bem - OpenVSP .bem file 
        prop    - SUAVE propeller data structure 
        
    Outputs:
        NA                                                

    Properties Used:
        N/A
    """  
    header = \
    '''Radius/R, Chord/R, Twist (deg), Rake/R, Skew/R, Sweep, t/c, CLi, Axial, Tangential\n''' 
    
    N          = len(prop.radius_distribution)
    r_R        = np.zeros(N)
    c_R        = np.zeros(N)
    beta_deg   = np.zeros(N)
    Rake_R     = np.zeros(N)
    Skew_R     = np.zeros(N)
    Sweep      = np.zeros(N)
    t_c        = np.zeros(N)
    CLi        = np.zeros(N)
    Axial      = np.zeros(N)
    Tangential = np.zeros(N)
     
    r_R        = prop.radius_distribution/prop.tip_radius
    c_R        = prop.chord_distribution/prop.tip_radius
    beta_deg   = prop.twist_distribution/Units.degrees 
    Rake_R     = np.zeros(N)
    Skew_R     = np.zeros(N)
    Sweep      = np.arctan(prop.mid_chord_alignment/prop.radius_distribution)
    t_c        = prop.thickness_to_chord
    CLi        = np.ones(N)*prop.design_Cl
    Axial      = np.zeros(N)
    Tangential = np.zeros(N)
    
    # Write propeller station imformation
    vsp_bem.write(header)       
    for i in range(N):
        section_text = format(r_R[i], '.7f')+ ", " + format(c_R[i], '.7f')+ ", " + format(beta_deg[i], '.7f')+ ", " +\
            format( Rake_R[i], '.7f')+ ", " + format(Skew_R[i], '.7f')+ ", " + format(Sweep[i], '.7f')+ ", " +\
            format(t_c[i], '.7f')+ ", " + format(CLi[i], '.7f') + ", "+ format(Axial[i], '.7f') + ", " +\
            format(Tangential[i], '.7f') + "\n"  
        vsp_bem.write(section_text)      

    return   

## @ingroup Input_Output-OpenVSP
def make_airfoil_text(vsp_bem,prop):   
    """This function writes the airfoil geometry into the vsp file

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        vsp_bem - OpenVSP .bem file 
        prop    - SUAVE propeller data structure 
        
    Outputs:
        NA                

    Properties Used:
        N/A
    """ 
  
    N             = len(prop.radius_distribution)  
    airfoil_data  = import_airfoil_geometry(prop.airfoil_geometry)
    a_sec         = prop.airfoil_polar_stations
    for i in range(N):
        airfoil_station_header = '\nSection ' + str(i) + ' X, Y\n'  
        vsp_bem.write(airfoil_station_header)   
        
        airfoil_x     = airfoil_data.x_coordinates[int(a_sec[i])] 
        airfoil_y     = airfoil_data.y_coordinates[int(a_sec[i])] 
         
        for j in range(len(airfoil_x)): 
            section_text = format(airfoil_x[j], '.7f')+ ", " + format(airfoil_y[j], '.7f') + "\n"  
            vsp_bem.write(section_text)      
    return 


 
