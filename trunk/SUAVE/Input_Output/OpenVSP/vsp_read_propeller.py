## @ingroup Input_Output-OpenVSP
# vsp_read_propeller.py
#
# Created:  Sep 2018, W. Maier
# Modified: Apr 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Data, Units
import numpy as np

# ----------------------------------------------------------------------
#  Reading BEM files
# ---------------------------------------------------------------------- 
## @ingroup Input_Output-OpenVSP
def vsp_read_propeller(filename):
    """   This functions reads a .bem file from OpenVSP and saves it in the SUAVE propeller format

    Assumptions:
        None
        
    Source:
        None

    Inputs:
        OpenVSP .bem filename

    Outputs:
        SUAVE Propeller Data Structure     

    Properties Used:
        N/A
    """  
    # open newly written result files and read in aerodynamic properties 
    with open(filename,'r') as vsp_prop_file:  
        vsp_bem_lines   = vsp_prop_file.readlines()
         
        tag                  = vsp_bem_lines[0][0:20].strip('.')
        n_stations           = int(vsp_bem_lines[1][13:16].strip())  
        num_blades           = int(vsp_bem_lines[2][10:14].strip())  
        diameter             = float(vsp_bem_lines[3][9:21].strip()) 
        three_quarter_twist  = float(vsp_bem_lines[4][15:28].strip())   
        feather              = float(vsp_bem_lines[5][14:28].strip()) 
        precone              = float(vsp_bem_lines[6][15:28].strip()) 
        center               = list(vsp_bem_lines[7][7:44].strip().split(',')) 
        normal               = list(vsp_bem_lines[7][7:44].strip().split(','))  
          
        header     = 11 
        Radius_R   = np.zeros(n_stations)
        Chord_R    = np.zeros(n_stations)
        Twist_deg  = np.zeros(n_stations)
        Rake_R     = np.zeros(n_stations)
        Skew_R     = np.zeros(n_stations)
        Sweep      = np.zeros(n_stations)
        t_c        = np.zeros(n_stations)
        CLi        = np.zeros(n_stations)
        Axial      = np.zeros(n_stations)
        Tangential = np.zeros(n_stations) 
        
        for i in range(n_stations):
            station       = list(vsp_bem_lines[header + i][0:120].strip().split(','))   
            Radius_R[i]   = float(station[0])
            Chord_R[i]    = float(station[1])
            Twist_deg[i]  = float(station[2])
            Rake_R[i]     = float(station[3])
            Skew_R[i]     = float(station[4])
            Sweep[i]      = float(station[5])
            t_c[i]        = float(station[6])
            CLi[i]        = float(station[7])
            Axial[i]      = float(station[8])
            Tangential[i] = float(station[9]) 
    
    # non dimensional radius cannot be 1.0 for bemt
    Radius_R[-1] = 0.99
    
    # unpack 
    prop = SUAVE.Components.Energy.Converters.Propeller()  
    prop.inputs                     = Data()
    prop.number_of_blades           = num_blades 
    prop.tag                        = tag 
    prop.tip_radius                 = diameter/2 
    prop.hub_radius                 = prop.tip_radius*Radius_R[0]  
    prop.design_Cl                  = np.mean(CLi)      
    prop.radius_distribution        = Radius_R*prop.tip_radius
    prop.chord_distribution         = Chord_R*prop.tip_radius
    prop.twist_distribution         = Twist_deg*Units.degrees 
    prop.mid_chord_alignment        = np.tan(Sweep*Units.degrees)*prop.radius_distribution    
    prop.thickness_to_chord         = t_c 
    prop.max_thickness_distribution = t_c*prop.chord_distribution
    prop.origin                     = [[float(center[0]) ,float(center[1]),float(center[2]) ]]
    prop.thrust_angle               = np.tan(float(normal[2])/-float(normal[0])) 
    prop.Cl_distribution            = CLi  
    
    return prop