## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# VLM.py
# 
# Created:  Dec 2013, SUAVE Team
# Modified: Apr 2017, T. MacDonald
#           Oct 2017, E. Botero
#           Jun 2018, M. Clarke


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Aerodynamics.XFOIL.compute_airfoil_polars import read_wing_airfoil
import matplotlib.pyplot as plt
import time
# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def VLM(conditions,configuration,geometry):
    ti = time.time()
    """Uses the vortex lattice method to compute the lift, induced drag and moment coefficients  

    Assumptions:
    None

    Source:
    1. Aerodynamics for Engineers, Sixth Edition by John Bertin & Russel Cummings 
    
    2. An Introduction to Theoretical and Computational Aerodynamics by Jack Moran
    
    3. Yahyaoui, M. "Generalized Vortex Lattice Method for Predicting Characteristics of Wings
    with Flap and Aileron Deflection" , World Academy of Science, Engineering and Technology 
    International Journal of Mechanical, Aerospace, Industrial and Mechatronics Engineering 
    Vol:8 No:10, 2014
    

    Inputs:
    geometry.
       wing.
         spans.projected                       [m]
         chords.root                           [m]
         chords.tip                            [m]
         sweeps.quarter_chord                  [radians]
         taper                                 [Unitless]
         twists.root                           [radians]
         twists.tip                            [radians]
         symmetric                             [Boolean]
         aspect_ratio                          [Unitless]
         areas.reference                       [m^2]
         vertical                              [Boolean]
         origin                                [m]
       configuration.number_panels_spanwise    [Unitless]
       configuration.number_panels_chordwise   [Unitless]
       conditions.aerodynamics.angle_of_attack [radians]

    Outputs:
    CL                                      [Unitless]
    Cl                                      [Unitless]
    CDi                                     [Unitless]
    Cdi                                     [Unitless]

    Properties Used:
    N/A
    """ 
    
    # ---------------------------------------------------------------------------------------
    # STEP 1: 
    # ---------------------------------------------------------------------------------------
    
    # unpack settings
    n_sw   = configuration.number_panels_spanwise   # per wing, if segments are defined, per segment
    n_cw   = configuration.number_panels_chordwise  # per wing, if segments are defined, per segment
    
    # define point about which moment coefficient is computed 
    x_mac  = geometry.wings['main_wing'].aerodynamic_center[0] + geometry.wings['main_wing'].origin[0]
    x_cg   = geometry.mass_properties.center_of_gravity[0] 
    if x_cg == None:
        x_m = x_mac 
    else:
        x_m = x_cg
    
    aoa = conditions.aerodynamics.angle_of_attack   # angle of attack    
    
    # define empty vectors for coordinates of panes, control points and bound vortices
    XAH = np.empty(shape=[0,1])
    YAH = np.empty(shape=[0,1])
    ZAH = np.empty(shape=[0,1])
    XBH = np.empty(shape=[0,1])
    YBH = np.empty(shape=[0,1])
    ZBH = np.empty(shape=[0,1])
    XCH = np.empty(shape=[0,1])
    YCH = np.empty(shape=[0,1])
    ZCH = np.empty(shape=[0,1]) 
    
    XA1 = np.empty(shape=[0,1])
    YA1 = np.empty(shape=[0,1])  
    ZA1 = np.empty(shape=[0,1])
    XA2 = np.empty(shape=[0,1])
    YA2 = np.empty(shape=[0,1])    
    ZA2 = np.empty(shape=[0,1])
    
    XB1 = np.empty(shape=[0,1])
    YB1 = np.empty(shape=[0,1])  
    ZB1 = np.empty(shape=[0,1])
    XB2 = np.empty(shape=[0,1])
    YB2 = np.empty(shape=[0,1])    
    ZB2 = np.empty(shape=[0,1]) 

    XAC = np.empty(shape=[0,1])
    YAC = np.empty(shape=[0,1])
    ZAC = np.empty(shape=[0,1]) 
    XBC = np.empty(shape=[0,1])
    YBC = np.empty(shape=[0,1])
    ZBC = np.empty(shape=[0,1]) 
    
    XC = np.empty(shape=[0,1])
    YC = np.empty(shape=[0,1])
    ZC = np.empty(shape=[0,1]) 
    
    CS = np.empty(shape=[0,1]) 
    n_w = 0 # number of wings 
    n_cp = 0 # number of bound vortices
    wing_areas = [] # wing areas 
    
    for wing in geometry.wings.values():
        # ---------------------------------------------------------------------------------------
        # STEP 2: Unpack aircraft wing geometry 
        # ---------------------------------------------------------------------------------------
        span        = wing.spans.projected
        root_chord  = wing.chords.root
        tip_chord   = wing.chords.tip
        sweep_qc    = wing.sweeps.quarter_chord
        sweep_le    = wing.sweeps.leading_edge
        taper       = wing.taper
        twist_rc    = wing.twists.root
        twist_tc    = wing.twists.tip
        dihedral    = wing.dihedral
        sym_para    = wing.symmetric
        Sref        = wing.areas.reference
        vertical_wing = wing.vertical
        wing_origin = wing.origin

        # determine of vehicle has symmetry 
        if sym_para is True :
            span = span/2
            
        delta_y  = span/n_sw    
        
        n_w += 1
        cs_w = np.zeros(n_sw)
        
        ya  = np.zeros(n_sw*n_cw) 
        yb  = np.zeros(n_sw*n_cw)
        
        xah = np.zeros(n_sw*n_cw)
        yah = np.zeros(n_sw*n_cw)
        zah = np.zeros(n_sw*n_cw)
        xbh = np.zeros(n_sw*n_cw)
        ybh = np.zeros(n_sw*n_cw)
        zbh = np.zeros(n_sw*n_cw)   
        
        xch = np.zeros(n_sw*n_cw)
        ych = np.zeros(n_sw*n_cw)
        zch = np.zeros(n_sw*n_cw)
          
        xa1 = np.zeros(n_sw*n_cw)
        ya1 = np.zeros(n_sw*n_cw)
        za1 = np.zeros(n_sw*n_cw)
        xa2 = np.zeros(n_sw*n_cw)
        ya2 = np.zeros(n_sw*n_cw)
        za2 = np.zeros(n_sw*n_cw)
        
        xb1 = np.zeros(n_sw*n_cw)
        yb1 = np.zeros(n_sw*n_cw)
        zb1 = np.zeros(n_sw*n_cw)
        xb2 = np.zeros(n_sw*n_cw) 
        yb2 = np.zeros(n_sw*n_cw) 
        zb2 = np.zeros(n_sw*n_cw)
                       
        xac = np.zeros(n_sw*n_cw)
        yac = np.zeros(n_sw*n_cw)
        zac = np.zeros(n_sw*n_cw)
        
        xbc = np.zeros(n_sw*n_cw)
        ybc = np.zeros(n_sw*n_cw)
        zbc = np.zeros(n_sw*n_cw)
        
        xc  = np.zeros(n_sw*n_cw) 
        yc  = np.zeros(n_sw*n_cw) 
        zc  = np.zeros(n_sw*n_cw)
        
        section_twist= np.zeros(n_sw) 
        # ---------------------------------------------------------------------------------------
        # STEP 3: Determine if wing segments are defined  
        # ---------------------------------------------------------------------------------------
        n_segments           = len(wing.Segments.keys())
        if n_segments>0:
            # ---------------------------------------------------------------------------------------
            # STEP 4A: Discretizing the wing sections into panels
            # ---------------------------------------------------------------------------------------
            i             = np.arange(0,n_sw)
            j             = np.arange(0,n_sw+1)
            y_coordinates = (j)*delta_y  
            
            segment_chord          = np.zeros(n_segments)
            segment_twist          = np.zeros(n_segments)
            segment_sweep          = np.zeros(n_segments)
            segment_span           = np.zeros(n_segments)
            segment_area           = np.zeros(n_segments)
            segment_dihedral       = np.zeros(n_segments)
            segment_x_coord        = [] 
            segment_camber         = []
            segment_chord_x_offset = np.zeros(n_segments)
            segment_chord_z_offset = np.zeros(n_segments)
            section_stations       = np.zeros(n_segments)
            
            # ---------------------------------------------------------------------------------------
            # STEP 5A: Obtain sweep, chord, dihedral and twist at the beginning/end of each segment.
            #          If applicable, append airfoil section data and flap/aileron deflection angles.
            # ---------------------------------------------------------------------------------------
            segment_sweeps = []
            for i_seg in range(n_segments):   
                segment_chord[i_seg]    = wing.Segments[i_seg].root_chord_percent*root_chord
                segment_twist[i_seg]    = wing.Segments[i_seg].twist
                section_stations[i_seg] = wing.Segments[i_seg].percent_span_location*span  
                segment_dihedral[i_seg] = wing.Segments[i_seg].dihedral_outboard                    
        
                # change to leading edge sweep, if quarter chord sweep givent, convert to leading edge sweep 
                if (i_seg == n_segments-1):
                    segment_sweep[i_seg] = 0                                  
                else: 
                    if wing.Segments[i_seg].sweeps.leading_edge > 0:
                        segment_sweep[i_seg] = wing.Segments[i_seg].sweeps.leading_edge
                    else:                                                                 
                        sweep_quarter_chord  = wing.Segments[i_seg].sweeps.quarter_chord
                        cf       = 0.25                          
                        seg_root_chord       = root_chord*wing.Segments[i_seg].root_chord_percent
                        seg_tip_chord        = root_chord*wing.Segments[i_seg+1].root_chord_percent
                        seg_span             = span*(wing.Segments[i_seg+1].percent_span_location - wing.Segments[i_seg].percent_span_location )
                        segment_sweep[i_seg] = np.arctan(((seg_root_chord*cf) + (np.tan(sweep_quarter_chord)*seg_span - cf*seg_tip_chord)) /seg_span)  
        
                if i_seg == 0:
                    segment_span[i_seg]           = 0.0
                    segment_chord_x_offset[i_seg] = 0.0  
                    segment_chord_z_offset[i_seg] = 0.0
                else:
                    segment_span[i_seg]           = wing.Segments[i_seg].percent_span_location*span - wing.Segments[i_seg-1].percent_span_location*span
                    segment_chord_x_offset[i_seg] = segment_chord_x_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_sweep[i_seg-1])
                    segment_chord_z_offset[i_seg] = segment_chord_z_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_dihedral[i_seg-1])
                
                # Get airfoil section data  
                if wing.Segments[i_seg].Airfoil: 
                    airfoil_data = read_wing_airfoil(wing.Segments[i_seg].Airfoil.airfoil.coordinate_file )    
                    segment_camber.append(airfoil_data.camber_coordinates)
                    segment_x_coord.append(airfoil_data.x_lower_surface) 
                else:
                    segment_camber.append(np.zeros(30))              
                    segment_x_coord.append(np.linspace(0,1,30)) 
               
                # ** TO DO ** Get flap/aileron locations and deflection
                
            wing_areas.append(np.sum(segment_area[:]))
            if sym_para is True :
                wing_areas.append(np.sum(segment_area[:]))
                
            #Shift spanwise vortices onto section breaks  
            for i_seg in range(n_segments):
                idx =  (np.abs(y_coordinates-section_stations[i_seg])).argmin()
                y_coordinates[idx] = section_stations[i_seg]                
            
            # ---------------------------------------------------------------------------------------
            # STEP 6A: Define coordinates of panels horseshoe vortices and control points 
            # ---------------------------------------------------------------------------------------
            del_y = y_coordinates[i+1] - y_coordinates[i]
        
            # define coordinates of horseshoe vortices and control points
            i_seg = 0
            idx   = 0
            y_a   = np.atleast_2d(y_coordinates[i]) 
            y_b   = np.atleast_2d(y_coordinates[i+1])
            
            #new_segment = False 
            for idx_y in range(n_sw):                    
                for idx_x in range(n_cw):
                    eta_a = (y_a[0][idx_y] - section_stations[i_seg])  
                    eta_b = (y_b[0][idx_y] - section_stations[i_seg]) 
                    eta   = (y_b[0][idx_y] - del_y[idx_y]/2 - section_stations[i_seg]) 
                    
                    segment_chord_ratio = (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1]
                    segment_twist_ratio = (segment_twist[i_seg+1] - segment_twist[i_seg])/segment_span[i_seg+1]
                    
                    wing_chord_section_a  = segment_chord[i_seg] + (eta_a*segment_chord_ratio) 
                    wing_chord_section_b  = segment_chord[i_seg] + (eta_b*segment_chord_ratio)
                    wing_chord_section    = segment_chord[i_seg] + (eta*segment_chord_ratio)
                    
                    delta_x_a = wing_chord_section_a/n_cw   
                    delta_x_b = wing_chord_section_b/n_cw   
                    delta_x = wing_chord_section/n_cw                                  
                    
                    xi_a1 = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x                  # x coordinate of top left corner of panel
                    xi_ah = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x + delta_x_a*0.25 # x coordinate of left corner of panel
                    xi_a2 = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x + delta_x_a      # x coordinate of bottom left corner of bound vortex 
                    xi_ac = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x + delta_x_a*0.75 # x coordinate of bottom left corner of control point vortex  
                    xi_b1 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x                  # x coordinate of top right corner of panel      
                    xi_bh = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
                    xi_b2 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel
                    xi_bc = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
                    xi_c  = segment_chord_x_offset[i_seg] + eta *np.tan(segment_sweep[i_seg])  + delta_x  *idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
                    xi_ch = segment_chord_x_offset[i_seg] + eta *np.tan(segment_sweep[i_seg])  + delta_x  *idx_x + delta_x*0.25   # x coordinate center of bound vortex of each panel 
                    

                    # camber
                    section_camber_a  = segment_camber[i_seg]*wing_chord_section_a  
                    section_camber_b  = segment_camber[i_seg]*wing_chord_section_b  
                    section_camber_c    = segment_camber[i_seg]*wing_chord_section
                    
                    section_x_coord_a = segment_x_coord[i_seg]*wing_chord_section_a
                    section_x_coord_b = segment_x_coord[i_seg]*wing_chord_section_b
                    section_x_coord   = segment_x_coord[i_seg]*wing_chord_section
                    
                    z_c_a1 = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_camber_a) 
                    z_c_ah = np.interp((idx_x    *delta_x_a + delta_x_a*0.25) ,section_x_coord_a,section_camber_a)
                    z_c_a2 = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_camber_a) 
                    z_c_ac = np.interp((idx_x    *delta_x_a + delta_x_a*0.75) ,section_x_coord_a,section_camber_a) 
                    z_c_b1 = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_camber_b)   
                    z_c_bh = np.interp((idx_x    *delta_x_b + delta_x_b*0.25) ,section_x_coord_b,section_camber_b) 
                    z_c_b2 = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_camber_b) 
                    z_c_bc = np.interp((idx_x    *delta_x_b + delta_x_b*0.75) ,section_x_coord_b,section_camber_b) 
                    z_c    = np.interp((idx_x    *delta_x   + delta_x  *0.75) ,section_x_coord,section_camber_c) 
                    z_c_ch = np.interp((idx_x    *delta_x   + delta_x  *0.25) ,section_x_coord,section_camber_c) 
                    
                    zeta_a1 = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a1  # z coordinate of top left corner of panel
                    zeta_ah = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_ah  # z coordinate of left corner of bound vortex  
                    zeta_a2 = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a2  # z coordinate of bottom left corner of panel
                    zeta_ac = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_ac  # z coordinate of bottom left corner of panel of control point
                    zeta_bc = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_bc  # z coordinate of top right corner of panel of control point                          
                    zeta_b1 = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b1  # z coordinate of top right corner of panel  
                    zeta_bh = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_bh  # z coordinate of right corner of bound vortex        
                    zeta_b2 = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b2  # z coordinate of bottom right corner of panel                 
                    zeta    = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])    + z_c     # z coordinate three-quarter chord control point for each panel
                    zeta_ch = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])    + z_c_ch  # z coordinate center of bound vortex on each panel
                    
                    # adjustment of panels for twist  
                    xi_LE_a = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg])               # x location of leading edge left corner of wing
                    xi_LE_b = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg])               # x location of leading edge right of wing
                    xi_LE   = segment_chord_x_offset[i_seg] + eta*np.tan(segment_sweep[i_seg])                 # x location of leading edge center of wing
                    
                    zeta_LE_a = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])          # z location of leading edge left corner of wing
                    zeta_LE_b = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])          # z location of leading edge right of wing
                    zeta_LE   = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])            # z location of leading edge center of wing
                    
                    # determine section twist
                    section_twist_a = segment_twist[i_seg] + (eta_a * segment_twist_ratio)                     # twist at left side of panel
                    section_twist_b = segment_twist[i_seg] + (eta_b * segment_twist_ratio)                     # twist at right side of panel
                    section_twist   = segment_twist[i_seg] + (eta* segment_twist_ratio)                        # twist at center local chord 
                    
                    xi_prime_a1  = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1-zeta_LE_a)   # x coordinate transformation of top left corner
                    xi_prime_ah  = xi_LE_a + np.cos(section_twist_a)*(xi_ah-xi_LE_a) + np.sin(section_twist_a)*(zeta_ah-zeta_LE_a)   # x coordinate transformation of bottom left corner
                    xi_prime_a2  = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner
                    xi_prime_ac  = xi_LE_a + np.cos(section_twist_a)*(xi_ac-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner of control point
                    xi_prime_bc  = xi_LE_b + np.cos(section_twist_b)*(xi_bc-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner of control point                         
                    xi_prime_b1  = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner 
                    xi_prime_bh  = xi_LE_b + np.cos(section_twist_b)*(xi_bh-xi_LE_b) + np.sin(section_twist_b)*(zeta_bh-zeta_LE_b)   # x coordinate transformation of top right corner 
                    xi_prime_b2  = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2-zeta_LE_b)   # x coordinate transformation of botton right corner 
                    xi_prime     = xi_LE   + np.cos(section_twist)  *(xi_c-xi_LE)    + np.sin(section_twist)*(zeta-zeta_LE)          # x coordinate transformation of control point
                    xi_prime_ch  = xi_LE   + np.cos(section_twist)  *(xi_ch-xi_LE)   + np.sin(section_twist)*(zeta_ch-zeta_LE)       # x coordinate transformation of center of horeshoe vortex 
                    
                    zeta_prime_a1  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1-zeta_LE_a) # z coordinate transformation of top left corner
                    zeta_prime_ah  = zeta_LE_a - np.sin(section_twist_a)*(xi_ah-xi_LE_a) + np.cos(section_twist_a)*(zeta_ah-zeta_LE_a) # z coordinate transformation of bottom left corner
                    zeta_prime_a2  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2-zeta_LE_a) # z coordinate transformation of bottom left corner
                    zeta_prime_ac  = zeta_LE_a - np.sin(section_twist_a)*(xi_ac-xi_LE_a) + np.cos(section_twist_a)*(zeta_ac-zeta_LE_a) # z coordinate transformation of bottom left corner
                    zeta_prime_bc  = zeta_LE_b - np.sin(section_twist_b)*(xi_bc-xi_LE_b) + np.cos(section_twist_b)*(zeta_bc-zeta_LE_b) # z coordinate transformation of top right corner                         
                    zeta_prime_b1  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1-zeta_LE_b) # z coordinate transformation of top right corner 
                    zeta_prime_bh  = zeta_LE_b - np.sin(section_twist_b)*(xi_bh-xi_LE_b) + np.cos(section_twist_b)*(zeta_bh-zeta_LE_b) # z coordinate transformation of top right corner 
                    zeta_prime_b2  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2-zeta_LE_b) # z coordinate transformation of botton right corner 
                    zeta_prime     = zeta_LE   - np.sin(section_twist)*(xi_c-xi_LE) + np.cos(-section_twist)*(zeta-zeta_LE)            # z coordinate transformation of control point
                    zeta_prime_ch   = zeta_LE   - np.sin(section_twist)*(xi_ch-xi_LE) + np.cos(-section_twist)*(zeta_ch-zeta_LE)            # z coordinate transformation of center of horseshoe
                                           
                    # ** TO DO ** Get flap/aileron locations and deflection
                    
                    # store coordinates of panels, horseshoeces vortices and control points relative to wing root 
                    if vertical_wing:
                        xa1[idx] = xi_prime_a1 
                        za1[idx] = y_a[0][idx_y]
                        ya1[idx] = zeta_prime_a1
                        xa2[idx] = xi_prime_a2
                        za2[idx] = y_a[0][idx_y]
                        ya2[idx] = zeta_prime_a2
                        
                        xb1[idx] = xi_prime_b1 
                        zb1[idx] = y_b[0][idx_y]
                        yb1[idx] = zeta_prime_b1
                        xb2[idx] = xi_prime_b2 
                        zb2[idx] = y_b[0][idx_y]                        
                        yb2[idx] = zeta_prime_b2 
                        
                        xah[idx] = xi_prime_ah
                        zah[idx] = y_a[0][idx_y]
                        yah[idx] = zeta_prime_ah                    
                        xbh[idx] = xi_prime_bh 
                        zbh[idx] = y_b[0][idx_y]
                        ybh[idx] = zeta_prime_bh
                        
                        xch[idx] = xi_prime_ch
                        zch[idx] = y_b[0][idx_y] - del_y[idx_y]/2                    
                        ych[idx] = zeta_prime_ch
                        
                        xc[idx]  = xi_prime 
                        zc[idx]  = y_b[0][idx_y] - del_y[idx_y]/2 
                        yc[idx]  = zeta_prime 
                        
                        xac[idx] = xi_prime_ac 
                        zac[idx] = y_a[0][idx_y]
                        yac[idx] = zeta_prime_ac
                        xbc[idx] = xi_prime_bc
                        zbc[idx] = y_b[0][idx_y]                            
                        ybc[idx] = zeta_prime_bc                        
                        
                    else:
                        xa1[idx] = xi_prime_a1 
                        ya1[idx] = y_a[0][idx_y]
                        za1[idx] = zeta_prime_a1
                        xa2[idx] = xi_prime_a2
                        ya2[idx] = y_a[0][idx_y]
                        za2[idx] = zeta_prime_a2
                        
                        xb1[idx] = xi_prime_b1 
                        yb1[idx] = y_b[0][idx_y]
                        zb1[idx] = zeta_prime_b1
                        yb2[idx] = y_b[0][idx_y]
                        xb2[idx] = xi_prime_b2 
                        zb2[idx] = zeta_prime_b2 
                        
                        xah[idx] = xi_prime_ah
                        yah[idx] = y_a[0][idx_y]
                        zah[idx] = zeta_prime_ah                    
                        xbh[idx] = xi_prime_bh 
                        ybh[idx] = y_b[0][idx_y]
                        zbh[idx] = zeta_prime_bh
                        
                        xch[idx] = xi_prime_ch
                        ych[idx] = y_b[0][idx_y] - del_y[idx_y]/2                    
                        zch[idx] = zeta_prime_ch
                        
                        xc[idx]  = xi_prime 
                        yc[idx]  = y_b[0][idx_y] - del_y[idx_y]/2 
                        zc[idx]  = zeta_prime 
                        
                        xac[idx] = xi_prime_ac 
                        yac[idx] = y_a[0][idx_y]
                        zac[idx] = zeta_prime_ac
                        xbc[idx] = xi_prime_bc
                        ybc[idx] = y_b[0][idx_y]                            
                        zbc[idx] = zeta_prime_bc                         
                    idx += 1
                    
                cs_w[idx_y] = wing_chord_section       
            
                if y_coordinates[idx_y] == wing.Segments[i_seg+1].percent_span_location*span: 
                    i_seg += 1                    
                    new_segment = True
                if y_coordinates[idx_y+1] == span:
                    continue                                      
                                                    
        else:   # no segments defined on wing  
            # ---------------------------------------------------------------------------------------
            # STEP 6B: Define coordinates of panels horseshoe vortices and control points 
            # ---------------------------------------------------------------------------------------
            
            if sweep_le != 0:
                sweep = sweep_le
            else:                                                                
                cf    = 0.25                          
                sweep = np.arctan(((root_chord*cf) + (np.tan(sweep_qc)*span - cf*tip_chord)) /span)  
     
            i    = np.arange(0,n_sw)             
            wing_chord_ratio = (tip_chord-root_chord)/span
            wing_twist_ratio = (twist_tc-twist_rc)/span                    
            wing_areas.append(0.5*(root_chord+tip_chord)*span) 
            if sym_para is True :
                wing_areas.append(0.5*(root_chord+tip_chord)*span)   
            
            # Get airfoil section data  
            if wing.Airfoil: 
                airfoil_data = read_wing_airfoil(wing.Airfoil.airfoil.coordinate_file)    
                wing_camber  = airfoil_data.camber_coordinates
                wing_x_coord = airfoil_data.x_lower_surface
            else:
                wing_camber  = np.zeros(30) # dimension of Selig airfoil data file
                wing_x_coord = np.linspace(0,1,30)
            
            idx = 0 
            y_a = np.atleast_2d((i)*delta_y) 
            y_b = np.atleast_2d((i+1)*delta_y) 
            for idx_y in range(n_sw):                    
                for idx_x in range(n_cw):  
                    eta_a = (y_a[0][idx_y])  
                    eta_b = (y_b[0][idx_y]) 
                    eta   = (y_b[0][idx_y] - delta_y/2) 
                     
                    wing_chord_section_a  = root_chord + (eta_a*wing_chord_ratio) 
                    wing_chord_section_b  = root_chord + (eta_b*wing_chord_ratio)
                    wing_chord_section    = root_chord + (eta*wing_chord_ratio)
                    
                    delta_x_a = wing_chord_section_a/n_cw   
                    delta_x_b = wing_chord_section_b/n_cw   
                    delta_x   = wing_chord_section/n_cw                                  
                    
                    xi_a1 = eta_a*np.tan(sweep) + delta_x_a*idx_x                  # x coordinate of top left corner of panel
                    xi_ah = eta_a*np.tan(sweep) + delta_x_a*idx_x + delta_x_a*0.25 # x coordinate of left corner of panel
                    xi_a2 = eta_a*np.tan(sweep) + delta_x_a*idx_x + delta_x_a      # x coordinate of bottom left corner of bound vortex 
                    xi_ac = eta_a*np.tan(sweep) + delta_x_a*idx_x + delta_x_a*0.75 # x coordinate of bottom left corner of control point vortex  
                    xi_b1 = eta_b*np.tan(sweep) + delta_x_b*idx_x                  # x coordinate of top right corner of panel      
                    xi_bh = eta_b*np.tan(sweep) + delta_x_b*idx_x + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
                    xi_b2 = eta_b*np.tan(sweep) + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel
                    xi_bc = eta_b*np.tan(sweep) + delta_x_b*idx_x + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
                    xi_c  =  eta *np.tan(sweep)  + delta_x  *idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
                    xi_ch =  eta *np.tan(sweep)  + delta_x  *idx_x + delta_x*0.25   # x coordinate center of bound vortex of each panel 
                                      
                    section_camber_a  = wing_camber*wing_chord_section_a
                    section_camber_b  = wing_camber*wing_chord_section_b  
                    section_camber_c  = wing_camber*wing_chord_section
                                           
                    section_x_coord_a = wing_x_coord*wing_chord_section_a
                    section_x_coord_b = wing_x_coord*wing_chord_section_b
                    section_x_coord   = wing_x_coord*wing_chord_section
                
                    z_c_a1 = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_camber_a) 
                    z_c_ah = np.interp((idx_x    *delta_x_a + delta_x_a*0.25) ,section_x_coord_a,section_camber_a)
                    z_c_a2 = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_camber_a) 
                    z_c_ac = np.interp((idx_x    *delta_x_a + delta_x_a*0.75) ,section_x_coord_a,section_camber_a) 
                    z_c_b1 = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_camber_b)   
                    z_c_bh = np.interp((idx_x    *delta_x_b + delta_x_b*0.25) ,section_x_coord_b,section_camber_b) 
                    z_c_b2 = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_camber_b) 
                    z_c_bc = np.interp((idx_x    *delta_x_b + delta_x_b*0.75) ,section_x_coord_b,section_camber_b) 
                    z_c    = np.interp((idx_x    *delta_x   + delta_x  *0.75) ,section_x_coord  ,section_camber_c) 
                    z_c_ch = np.interp((idx_x    *delta_x   + delta_x  *0.25) ,section_x_coord  ,section_camber_c) 
                    
                    zeta_a1 = eta_a*np.tan(dihedral)  + z_c_a1  # z coordinate of top left corner of panel
                    zeta_ah = eta_a*np.tan(dihedral)  + z_c_ah  # z coordinate of left corner of bound vortex  
                    zeta_a2 = eta_a*np.tan(dihedral)  + z_c_a2  # z coordinate of bottom left corner of panel
                    zeta_ac = eta_a*np.tan(dihedral)  + z_c_ac  # z coordinate of bottom left corner of panel of control point
                    zeta_bc = eta_b*np.tan(dihedral)  + z_c_bc  # z coordinate of top right corner of panel of control point                          
                    zeta_b1 = eta_b*np.tan(dihedral)  + z_c_b1  # z coordinate of top right corner of panel  
                    zeta_bh = eta_b*np.tan(dihedral)  + z_c_bh  # z coordinate of right corner of bound vortex        
                    zeta_b2 = eta_b*np.tan(dihedral)  + z_c_b2  # z coordinate of bottom right corner of panel                 
                    zeta    =   eta*np.tan(dihedral)    + z_c     # z coordinate three-quarter chord control point for each panel
                    zeta_ch =   eta*np.tan(dihedral)    + z_c_ch  # z coordinate center of bound vortex on each panel
                    
                                            
                    # adjustment of panels for twist  
                    xi_LE_a = eta_a*np.tan(sweep)               # x location of leading edge left corner of wing
                    xi_LE_b = eta_b*np.tan(sweep)               # x location of leading edge right of wing
                    xi_LE   = eta  *np.tan(sweep)               # x location of leading edge center of wing
                    
                    zeta_LE_a = eta_a*np.tan(dihedral)          # z location of leading edge left corner of wing
                    zeta_LE_b = eta_b*np.tan(dihedral)          # z location of leading edge right of wing
                    zeta_LE   = eta  *np.tan(dihedral)          # z location of leading edge center of wing
                    
                    # determine section twist
                    section_twist_a = twist_rc + (eta_a * wing_twist_ratio)                     # twist at left side of panel
                    section_twist_b = twist_rc + (eta_b * wing_twist_ratio)                     # twist at right side of panel
                    section_twist   = twist_rc + (eta   * wing_twist_ratio)                     # twist at center local chord 
                    
                    xi_prime_a1  = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1-zeta_LE_a)   # x coordinate transformation of top left corner
                    xi_prime_ah  = xi_LE_a + np.cos(section_twist_a)*(xi_ah-xi_LE_a) + np.sin(section_twist_a)*(zeta_ah-zeta_LE_a)   # x coordinate transformation of bottom left corner
                    xi_prime_a2  = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner
                    xi_prime_ac  = xi_LE_a + np.cos(section_twist_a)*(xi_ac-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner of control point
                    xi_prime_bc  = xi_LE_b + np.cos(section_twist_b)*(xi_bc-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner of control point                         
                    xi_prime_b1  = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner 
                    xi_prime_bh  = xi_LE_b + np.cos(section_twist_b)*(xi_bh-xi_LE_b) + np.sin(section_twist_b)*(zeta_bh-zeta_LE_b)   # x coordinate transformation of top right corner 
                    xi_prime_b2  = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2-zeta_LE_b)   # x coordinate transformation of botton right corner 
                    xi_prime     = xi_LE   + np.cos(section_twist)  *(xi_c-xi_LE)    + np.sin(section_twist)*(zeta-zeta_LE)          # x coordinate transformation of control point
                    xi_prime_ch  = xi_LE   + np.cos(section_twist)  *(xi_ch-xi_LE)   + np.sin(section_twist)*(zeta_ch-zeta_LE)       # x coordinate transformation of center of horeshoe vortex 
                    
                    zeta_prime_a1  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1-zeta_LE_a) # z coordinate transformation of top left corner
                    zeta_prime_ah  = zeta_LE_a - np.sin(section_twist_a)*(xi_ah-xi_LE_a) + np.cos(section_twist_a)*(zeta_ah-zeta_LE_a) # z coordinate transformation of bottom left corner
                    zeta_prime_a2  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2-zeta_LE_a) # z coordinate transformation of bottom left corner
                    zeta_prime_ac  = zeta_LE_a - np.sin(section_twist_a)*(xi_ac-xi_LE_a) + np.cos(section_twist_a)*(zeta_ac-zeta_LE_a) # z coordinate transformation of bottom left corner
                    zeta_prime_bc  = zeta_LE_b - np.sin(section_twist_b)*(xi_bc-xi_LE_b) + np.cos(section_twist_b)*(zeta_bc-zeta_LE_b) # z coordinate transformation of top right corner                         
                    zeta_prime_b1  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1-zeta_LE_b) # z coordinate transformation of top right corner 
                    zeta_prime_bh  = zeta_LE_b - np.sin(section_twist_b)*(xi_bh-xi_LE_b) + np.cos(section_twist_b)*(zeta_bh-zeta_LE_b) # z coordinate transformation of top right corner 
                    zeta_prime_b2  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2-zeta_LE_b) # z coordinate transformation of botton right corner 
                    zeta_prime     = zeta_LE   - np.sin(section_twist)  *(xi_c-xi_LE)    + np.cos(-section_twist) *(zeta-zeta_LE)      # z coordinate transformation of control point
                    zeta_prime_ch  = zeta_LE   - np.sin(section_twist)  *(xi_ch-xi_LE)   + np.cos(-section_twist) *(zeta_ch-zeta_LE)   # z coordinate transformation of center of horseshoe
                               
                    # ** TO DO ** Get flap/aileron locations and deflection
                    
                    # store coordinates of panels, horseshoeces vortices and control points relative to wing root 
                    if vertical_wing:
                        xa1[idx] = xi_prime_a1 
                        za1[idx] = y_a[0][idx_y]
                        ya1[idx] = zeta_prime_a1
                        xa2[idx] = xi_prime_a2
                        za2[idx] = y_a[0][idx_y]
                        ya2[idx] = zeta_prime_a2
                        
                        xb1[idx] = xi_prime_b1 
                        zb1[idx] = y_b[0][idx_y]
                        yb1[idx] = zeta_prime_b1
                        xb2[idx] = xi_prime_b2 
                        zb2[idx] = y_b[0][idx_y]                        
                        yb2[idx] = zeta_prime_b2 
                        
                        xah[idx] = xi_prime_ah
                        zah[idx] = y_a[0][idx_y]
                        yah[idx] = zeta_prime_ah                    
                        xbh[idx] = xi_prime_bh 
                        zbh[idx] = y_b[0][idx_y]
                        ybh[idx] = zeta_prime_bh
                        
                        xch[idx] = xi_prime_ch
                        zch[idx] = y_b[0][idx_y] - delta_y/2                    
                        ych[idx] = zeta_prime_ch
                        
                        xc[idx]  = xi_prime 
                        zc[idx]  = y_b[0][idx_y] - delta_y/2 
                        yc[idx]  = zeta_prime 
                        
                        xac[idx] = xi_prime_ac 
                        zac[idx] = y_a[0][idx_y]
                        yac[idx] = zeta_prime_ac
                        xbc[idx] = xi_prime_bc
                        zbc[idx] = y_b[0][idx_y]                            
                        ybc[idx] = zeta_prime_bc                        
                        
                    else:
                        xa1[idx] = xi_prime_a1 
                        ya1[idx] = y_a[0][idx_y]
                        za1[idx] = zeta_prime_a1
                        xa2[idx] = xi_prime_a2
                        ya2[idx] = y_a[0][idx_y]
                        za2[idx] = zeta_prime_a2
                        
                        xb1[idx] = xi_prime_b1 
                        yb1[idx] = y_b[0][idx_y]
                        zb1[idx] = zeta_prime_b1
                        yb2[idx] = y_b[0][idx_y]
                        xb2[idx] = xi_prime_b2 
                        zb2[idx] = zeta_prime_b2 
                        
                        xah[idx] = xi_prime_ah
                        yah[idx] = y_a[0][idx_y]
                        zah[idx] = zeta_prime_ah                    
                        xbh[idx] = xi_prime_bh 
                        ybh[idx] = y_b[0][idx_y]
                        zbh[idx] = zeta_prime_bh
                        
                        xch[idx] = xi_prime_ch
                        ych[idx] = y_b[0][idx_y] - delta_y/2                    
                        zch[idx] = zeta_prime_ch
                        
                        xc[idx]  = xi_prime 
                        yc[idx]  = y_b[0][idx_y] - delta_y/2 
                        zc[idx]  = zeta_prime 
                        
                        xac[idx] = xi_prime_ac 
                        yac[idx] = y_a[0][idx_y]
                        zac[idx] = zeta_prime_ac
                        xbc[idx] = xi_prime_bc
                        ybc[idx] = y_b[0][idx_y]                            
                        zbc[idx] = zeta_prime_bc                  
                    idx += 1
                
                cs_w[idx_y] = wing_chord_section
                
               
                                                           
        # adjusting coordinate axis so reference point is at the nose of the aircraft
        xah = xah + wing_origin[0] # x coordinate of left corner of bound vortex 
        yah = yah + wing_origin[1] # y coordinate of left corner of bound vortex 
        zah = zah + wing_origin[2] # z coordinate of left corner of bound vortex 
        xbh = xbh + wing_origin[0] # x coordinate of right corner of bound vortex 
        ybh = ybh + wing_origin[1] # y coordinate of right corner of bound vortex 
        zbh = zbh + wing_origin[2] # z coordinate of right corner of bound vortex 
        xch = xch + wing_origin[0] # x coordinate of center of bound vortex on panel
        ych = ych + wing_origin[1] # y coordinate of center of bound vortex on panel
        zch = zch + wing_origin[2] # z coordinate of center of bound vortex on panel  
        
        xa1 = xa1 + wing_origin[0] # x coordinate of top left corner of panel
        ya1 = ya1 + wing_origin[1] # y coordinate of bottom left corner of panel
        za1 = za1 + wing_origin[2] # z coordinate of top left corner of panel
        xa2 = xa2 + wing_origin[0] # x coordinate of bottom left corner of panel
        ya2 = ya2 + wing_origin[1] # x coordinate of bottom left corner of panel
        za2 = za2 + wing_origin[2] # z coordinate of bottom left corner of panel  
        
        xb1 = xb1 + wing_origin[0] # x coordinate of top right corner of panel  
        yb1 = yb1 + wing_origin[1] # y coordinate of top right corner of panel 
        zb1 = zb1 + wing_origin[2] # z coordinate of top right corner of panel 
        xb2 = xb2 + wing_origin[0] # x coordinate of bottom rightcorner of panel 
        yb2 = yb2 + wing_origin[1] # y coordinate of bottom rightcorner of panel 
        zb2 = zb2 + wing_origin[2] # z coordinate of bottom right corner of panel                   
        
        xac = xac + wing_origin[0]  # x coordinate of control points on panel
        yac = yac + wing_origin[1]  # y coordinate of control points on panel
        zac = zac + wing_origin[2]  # z coordinate of control points on panel
        xbc = xbc + wing_origin[0]  # x coordinate of control points on panel
        ybc = ybc + wing_origin[1]  # y coordinate of control points on panel
        zbc = zbc + wing_origin[2]  # z coordinate of control points on panel
        
        xc  = xc + wing_origin[0]  # x coordinate of control points on panel
        yc  = yc + wing_origin[1]  # y coordinate of control points on panel
        zc  = zc + wing_origin[2]  # y coordinate of control points on panel
        
        # if symmetry, store points of mirrored wing 
        if sym_para is True :
            n_w += 1
            cs_w = np.concatenate([cs_w,cs_w])
            xah = np.concatenate([xah,xah])
            yah = np.concatenate([yah,-yah])
            zah = np.concatenate([zah,zah])
            xbh = np.concatenate([xbh,xbh])
            ybh = np.concatenate([ybh,-ybh])
            zbh = np.concatenate([zbh,zbh])
            xch = np.concatenate([xch,xch])
            ych = np.concatenate([ych,-ych])
            zch = np.concatenate([zch,zch])
                                         
            xa1 = np.concatenate([xa1,xa1])
            ya1 = np.concatenate([ya1,-ya1])
            za1 = np.concatenate([za1,za1])
            xa2 = np.concatenate([xa2,xa2])
            ya2 = np.concatenate([ya2,-ya2])
            za2 = np.concatenate([za2,za2])
                                      
            xb1 = np.concatenate([xb1,xb1])
            yb1 = np.concatenate([yb1,-yb1])    
            zb1 = np.concatenate([zb1,zb1])
            xb2 = np.concatenate([xb2,xb2])
            yb2 = np.concatenate([yb2,-yb2])            
            zb2 = np.concatenate([zb2,zb2])
                                      
            xac = np.concatenate([xac ,xac ])
            yac = np.concatenate([yac ,-yac ])
            zac = np.concatenate([zac ,zac ])            
            xbc = np.concatenate([xbc ,xbc ])
            ybc = np.concatenate([ybc ,-ybc ])
            zbc = np.concatenate([zbc ,zbc ])
            xc  = np.concatenate([xc ,xc ])
            yc  = np.concatenate([yc ,-yc])
            zc  = np.concatenate([zc ,zc ])
        
        n_cp += len(xch)        
            
        # store wing in vehicle vector
        XAH  = np.append(XAH,xah)
        YAH  = np.append(YAH,yah)
        ZAH  = np.append(ZAH,zah)
        XBH  = np.append(XBH,xbh)
        YBH  = np.append(YBH,ybh)
        ZBH  = np.append(ZBH,zbh)
        XCH  = np.append(XCH,xch)
        YCH  = np.append(YCH,ych)
        ZCH  = np.append(ZCH,zch)   
        
        XA1  = np.append(XA1,xa1)
        YA1  = np.append(YA1,ya1)
        ZA1  = np.append(ZA1,za1)
        XA2  = np.append(XA2,xa2)
        YA2  = np.append(YA2,ya2)
        ZA2  = np.append(ZA2,za2)
        
        XB1  = np.append(XB1,xb1)
        YB1  = np.append(YB1,yb1)
        ZB1  = np.append(ZB1,zb1)
        XB2  = np.append(XB2,xb2)                
        YB2  = np.append(YB2,yb2)        
        ZB2  = np.append(ZB2,zb2)
        
        XAC  = np.append(XAC,xac)
        YAC  = np.append(YAC,yac) 
        ZAC  = np.append(ZAC,zac) 
        XBC  = np.append(XBC,xbc)
        YBC  = np.append(YBC,ybc) 
        ZBC  = np.append(ZBC,zbc)  
        XC   = np.append(XC ,xc)
        YC   = np.append(YC ,yc)
        ZC   = np.append(ZC ,zc)  
        CS = np.append(CS,cs_w)   
    ## ------------------------------------------------------------------
    ## Point Verification Plots 
    ## ------------------------------------------------------------------
    #fig = plt.figure()
    #axes = fig.add_subplot(2,2,1)
    #axes.plot(YA1,XA1,'bo') 
    #axes.plot(YA2,XA2,'rx')
    #axes.plot(YB1,XB1,'g1') 
    #axes.plot(YB2,XB2,'yv')  
    #axes.plot(YC,XC ,'ks')
    #axes.plot(YAH,XAH,'cD')
    #axes.plot(YBH,XBH,'k+')    
    #axes.set_ylabel('chord ')
    #axes.set_xlabel('span ')
    #plt.axis('equal')
    #axes.grid(True)
    
    #axes = fig.add_subplot(2,2,2)
    #axes.plot(XA1,ZA1,'bo') 
    #axes.plot(XA2,ZA2,'rx')
    #axes.plot(XB1,ZB1,'g1') 
    #axes.plot(XB2,ZB2,'yv')  
    #axes.plot(XC ,ZC ,'ks')
    #axes.plot(XAH,ZAH,'cD')
    #axes.plot(XBH,ZBH,'k+')    
    #axes.set_ylabel('height ')
    #axes.set_xlabel('span ')
    #plt.axis('equal')
    #axes.grid(True)
    
    #axes = fig.add_subplot(2,2,3)
    #axes.plot(YA1,ZA1,'bo') 
    #axes.plot(YA2,ZA2,'rx')
    #axes.plot(YB1,ZB1,'g1') 
    #axes.plot(YB2,ZB2,'yv')  
    #axes.plot(YC,ZC ,'ks')
    #axes.plot(YAH,ZAH,'cD')
    #axes.plot(YBH,ZBH,'k+')    
    #axes.set_ylabel('height')
    #axes.set_xlabel('span ')
    #plt.axis('equal')
    #axes.grid(True)
    
    #axes = fig.add_subplot(2,2,4,projection='3d')
    #axes.plot3D(XA1,YA1,ZA1,'bo') 
    #axes.plot3D(XA2,YA2,ZA2,'rx')
    #axes.plot3D(XB1,YB1,ZB1,'g1') 
    #axes.plot3D(XB2,YB2,ZB2,'yv')  
    #axes.plot3D(XC ,YC,ZC ,'ks')
    #axes.plot3D(XAH,YAH,ZAH,'cD')
    #axes.plot3D(XBH,YBH,ZBH,'k+')    
    #axes.set_zlabel('height')
    #axes.set_xlabel('chord ')
    #axes.set_ylabel('span ')
    #axes.grid(True)
    #plt.show()
    
    ##axis scaling
    #max_range = np.array([XAH.max()-XAH.min(), YAH.max()-YAH.min(), ZAH.max()-ZAH.min()]).max() / 2.0    
    #mid_x = (XAH.max()+XAH.min()) * 0.5
    #mid_y = (YAH.max()+YAH.min()) * 0.5
    #mid_z = (ZAH.max()+ZAH.min()) * 0.5
    #axes.set_xlim(mid_x - max_range, mid_x + max_range)
    #axes.set_ylim(mid_y - max_range, mid_y + max_range)
    #axes.set_zlim(mid_z - max_range, mid_z + max_range)
    
    #plt.axis('equal')
    #fig = plt.figure()
    #axes = fig.add_subplot(2,2,1)
    #axes.plot(YAC,XAC,'bo') 
    #axes.plot(YAC,XAC,'rx')
    #axes.plot(YBC,XBC,'g1') 
    #axes.plot(YBC,XBC,'yv')  
    #axes.plot(YC,XC ,'ks') 
    #axes.plot(YCH,XCH ,'ks') 
    #axes.set_ylabel('chord ')
    #axes.set_xlabel('span ')
    #plt.axis('equal')
    #axes.grid(True)
    
    #axes = fig.add_subplot(2,2,2)
    #axes.plot(XAC,ZAC,'bo') 
    #axes.plot(XAC,ZAC,'rx')
    #axes.plot(XBC,ZBC,'g1') 
    #axes.plot(XBC,ZBC,'yv')  
    #axes.plot(XC ,ZC ,'ks')
    #axes.plot(XCH,ZCH,'cD')   
    #axes.set_ylabel('height ')
    #axes.set_xlabel('span ')
    #plt.axis('equal')
    #axes.grid(True)
    
    #axes = fig.add_subplot(2,2,3)
    #axes.plot(YAC,ZAC,'bo') 
    #axes.plot(YAC,ZAC,'rx')
    #axes.plot(YBC,ZBC,'g1') 
    #axes.plot(YBC,ZBC,'yv')  
    #axes.plot(YC,ZC ,'ks')
    #axes.plot(YAH,ZCH,'cD')   
    #axes.set_ylabel('height')
    #axes.set_xlabel('span ')
    #plt.axis('equal')
    #axes.grid(True)
      
    #plt.show()
    
    # ---------------------------------------------------------------------------------------
    # STEP 7: Compute velocity induced by horseshoe vortex segments on every control point by 
    #         every panel
    # ---------------------------------------------------------------------------------------    
    n_cppw     = n_cw*n_sw      # number of control points per wing
    n_cp       = n_w*n_cw*n_sw  # total number of control points 
    theta_w    = aoa[0][0]
    C_AB_34_ll = np.zeros((n_cp,n_cp,3))
    C_AB_ll    = np.zeros((n_cp,n_cp,3))
    C_AB_34_rl = np.zeros((n_cp,n_cp,3))    
    C_AB_rl    = np.zeros((n_cp,n_cp,3))
    C_AB_bv    = np.zeros((n_cp,n_cp,3))
    C_Ainf     = np.zeros((n_cp,n_cp,3))
    C_Binf     = np.zeros((n_cp,n_cp,3))
    
    C_mn = np.zeros((n_cp,n_cp,3))
    DW_mn = np.zeros((n_cp,n_cp,3))
    
    for m in range(n_cp): # control point m
        sw_idx = 1
        cw_idx = 0
        for n in range(n_cp): # horseshe vortex n  	         
            # trailing vortices  
            XC_hat  = np.cos(theta_w)*XC[m] + np.sin(theta_w)*ZC[m]
            YC_hat  = YC[m]
            ZC_hat  = -np.sin(theta_w)*XC[m] + np.cos(theta_w)*ZC[m] 
                               
            XA2_hat =  np.cos(theta_w)*XA2[(sw_idx*n_cw)-1]  + np.sin(theta_w)*ZA2[(sw_idx*n_cw)-1]  
            YA2_hat =  YA2[(sw_idx*n_cw)-1]            
            ZA2_hat = -np.sin(theta_w)*XA2[(sw_idx*n_cw)-1]  + np.cos(theta_w)*ZA2[(sw_idx*n_cw)-1]            
                                                              
            XB2_hat =  np.cos(theta_w)*XB2[(sw_idx*n_cw)-1]  + np.sin(theta_w)*ZB2[(sw_idx*n_cw)-1]  
            YB2_hat =  YB2[(sw_idx*n_cw)-1]            
            ZB2_hat = -np.sin(theta_w)*XB2[(sw_idx*n_cw)-1]  + np.cos(theta_w)*ZB2[(sw_idx*n_cw)-1]         
            
            # starboard (right) wing 
            if YBH[n] > 0:
                # compute influence of bound vortices 
                C_AB_bv[m,n] = vortex(XC[m], YC[m], ZC[m], XAH[n], YAH[n], ZAH[n], XBH[n], YBH[n], ZBH[n])   
                
                # compute influence of 3/4 left legs
                C_AB_34_ll[m,n] = vortex(XC[m], YC[m], ZC[m], XA2[n], YA2[n], ZA2[n], XAH[n], YAH[n], ZAH[n])      
                
                # compute influence of whole panel left legs 
                C_AB_ll[m,n]   =  vortex(XC[m], YC[m], ZC[m], XA2[n], YA2[n], ZA2[n], XA1[n], YA1[n], ZA1[n])      
                     
                # compute influence of 3/4 right legs
                C_AB_34_rl[m,n] = vortex(XC[m], YC[m], ZC[m], XBH[n], YBH[n], ZBH[n], XB2[n], YB2[n], ZB2[n])      
                     
                # compute influence of whole right legs 
                C_AB_rl[m,n] = vortex(XC[m], YC[m], ZC[m], XB1[n], YB1[n], ZB1[n], XB2[n], YB2[n], ZB2[n])    
                                
                # velocity induced by left leg of vortex (A to inf)
                C_Ainf[m,n]  = vortex_to_inf_l(XC_hat, YC_hat, ZC_hat, XA2_hat, YA2_hat, ZA2_hat,theta_w)     
                
                # velocity induced by right leg of vortex (B to inf)
                C_Binf[m,n]  = vortex_to_inf_r(XC_hat, YC_hat, ZC_hat, XB2_hat, YB2_hat, ZB2_hat,theta_w) 
                
            # port (left) wing 
            else: 
                # compute influence of bound vortices 
                C_AB_bv[m,n]   = vortex(XC[m], YC[m], ZC[m], XBH[n], YBH[n], ZBH[n], XAH[n], YAH[n], ZAH[n])                       
            
                # compute influence of 3/4 left legs
                C_AB_34_ll[m,n] = vortex(XC[m], YC[m], ZC[m], XB2[n], YB2[n], ZB2[n], XBH[n], YBH[n], ZBH[n])        
            
                # compute influence of whole panel left legs 
                C_AB_ll[m,n]    =  vortex(XC[m], YC[m], ZC[m], XB2[n], YB2[n], ZB2[n], XB1[n], YB1[n], ZB1[n])      
                     
                # compute influence of 3/4 right legs
                C_AB_34_rl[m,n] = vortex(XC[m], YC[m], ZC[m], XAH[n], YAH[n], ZAH[n], XA2[n], YA2[n], ZA2[n]) 
                     
                # compute influence of whole right legs 
                C_AB_rl[m,n] =  vortex(XC[m], YC[m], ZC[m], XA1[n], YA1[n], ZA1[n], XA2[n], YA2[n], ZA2[n])   
                
                # velocity induced by left leg of vortex (A to inf)
                C_Ainf[m,n]  = vortex_to_inf_l(XC_hat, YC_hat, ZC_hat, XB2_hat, YB2_hat, ZB2_hat,theta_w)  
                
                # velocity induced by right leg of vortex (B to inf)
                C_Binf[m,n]  =  vortex_to_inf_r(XC_hat, YC_hat, ZC_hat, XA2_hat, YA2_hat, ZA2_hat,theta_w)   
            cw_idx += 1
            if cw_idx == n_cw:
                cw_idx = 0
                sw_idx += 1
    # Summation of Influences
    for m in range(n_cp):
        cw_idx = 0 
        sw_idx = 1
        for n in range(n_cp):
            C_AB_rl_ll = np.sum(C_AB_ll[m,(n+1):(n_cw*sw_idx)] +  C_AB_rl[m,(n+1):(n_cw*sw_idx)])
            C_mn[m,n,:]  = ( C_AB_rl_ll + C_AB_34_ll[m,n] + C_AB_bv[m,n] + C_AB_34_rl[m,n] + C_Ainf[m,n] + C_Binf[m,n]) # induced velocity from panel m  with i,j,k components
            DW_mn[m,n,:] =  C_Ainf[m,n] + C_Binf[m,n]                                                                  # induced downwash velocity from panel m with i,j,k components   
            cw_idx += 1 
            if cw_idx == n_cw:
                cw_idx = 0 
                sw_idx += 1
                
    # ---------------------------------------------------------------------------------------
    # STEP 8: Compute total velocity induced by horseshoe all vortices on every control point by 
    #         every panel
    # ---------------------------------------------------------------------------------------
    C_mn_i  = C_mn[:,:,0]
    C_mn_j  = C_mn[:,:,1]
    C_mn_k  = C_mn[:,:,2]
    DW_mn_k = DW_mn[:,:,2]
    
    # ---------------------------------------------------------------------------------------
    # STEP 9: Solve for vortex strengths and induced velocities 
    # ---------------------------------------------------------------------------------------
    phi   = np.arctan((ZBC - ZAC)/(YBC - YAC))
    delta = np.arctan((ZC - ZCH)/(XC - XCH)) 
    AoA   = np.ones_like(delta)*aoa[0][0]
    A     = np.zeros((n_cp,n_cp))
    
    for m in range(n_cp):
        for n in range(n_cp):     
            A[m,n] = C_mn_k[m,n] - C_mn_i[m,n]*np.tan(delta[n]) - C_mn_j[m,n]*np.tan(phi[n]) 
    
    RHS = np.tan(delta)*np.cos(AoA) - np.sin(AoA)
    
    # Vortex strength computation by matrix inversion
    gamma = np.linalg.solve(A,RHS)
    
    # induced velocities     
    u = np.dot(C_mn_i ,gamma)
    v = np.dot(C_mn_j ,gamma)
    w = np.dot(C_mn_k,gamma)    
    
    # ---------------------------------------------------------------------------------------
    # STEP 10: Compute aerodynamic coefficients 
    # ---------------------------------------------------------------------------------------    
    i = 0
    c_bar = np.mean(CS)# mean aerodynamic chord 
    
    X_M = np.ones(n_cp)*x_m  
    CL_wing  = np.zeros(n_w)
    CDi_wing  = np.zeros(n_w)
    Cl_wing = np.zeros(n_w*n_sw)
    Cdi_wing = np.zeros(n_w*n_sw)
    Y_val  = np.zeros(n_w*n_sw)
    
    Del_Y = np.abs(YB1 - YA1)
    i = 0
    for j in range(n_w):
        L_wing      = np.dot((u[j*n_cppw:(j+1)*n_cppw] +1),gamma[j*n_cppw:(j+1)*n_cppw] * Del_Y[j*n_cppw:(j+1)*n_cppw]) # wing lift coefficient
        CL_wing[j]  = L_wing/(wing_areas[j])
        Di_wing     = np.dot(-w[j*n_cppw:(j+1)*n_cppw]    ,gamma[j*n_cppw:(j+1)*n_cppw] * Del_Y[j*n_cppw:(j+1)*n_cppw]) # wing induced drag coefficient
        CDi_wing[j] = Di_wing/(wing_areas[j])
        for k in range(n_sw):   
            l_wing      = np.dot((u[i*n_cw:(i+1)*n_cw] +1),gamma[i*n_cw:(i+1)*n_cw] * Del_Y[i*n_cw:(i+1)*n_cw]) # sectional lift coefficients 
            Cl_wing[i]  = 2*l_wing/(CS[i])
            di_wing     = np.dot(-w[i*n_cw:(i+1)*n_cw]  ,gamma[i*n_cw:(i+1)*n_cw] * Del_Y[i*n_cw:(i+1)*n_cw])   # sectional induced drag coefficients
            Cdi_wing[i] = 2*di_wing/(CS[i])
            Y_val[i]    = YC[i*n_cw]
            i += 1
            
    
    L  = np.dot((1+u),gamma*Del_Y)
    CL = 2*L/(Sref) 
    
    D  =  -np.dot(w,gamma*Del_Y)
    CDi = 2*D/(np.pi*Sref) 
    
    CM  = np.dot((X_M - XCH),Del_Y*gamma)/(Sref*c_bar)   
     
    tf = time.time()
    print ('Time taken for VLM: ' + str(tf-ti))  
    
    #print ('VLM Discretization')
    #print (Y_val)
    
    return CL, Cl_wing, CDi, Cdi_wing, CM 

def vortex(X,Y,Z,X1,Y1,Z1,X2,Y2,Z2):
    R1R2X  =   (Y-Y1)*(Z-Z2) - (Z-Z1)*(Y-Y2) 
    R1R2Y  = -((X-X1)*(Z-Z2) - (Z-Z1)*(X-X2))
    R1R2Z  =   (X-X1)*(Y-Y2) - (Y-Y1)*(X-X2)
    SQUARE = R1R2X**2 + R1R2Y**2 + R1R2Z**2
    R1     = np.sqrt((X-X1)**2 + (Y-Y1)**2 + (Z-Z1)**2)
    R2     = np.sqrt((X-X2)**2 + (Y-Y2)**2 + (Z-Z2)**2)
    R0R1   = (X2-X1)*(X-X1) + (Y2-Y1)*(Y-Y1) + (Z2-Z1)*(Z-Z1)
    R0R2   = (X2-X1)*(X-X2) + (Y2-Y1)*(Y-Y2) + (Z2-Z1)*(Z-Z2)
    RVEC   = np.array([R1R2X,R1R2Y,R1R2Z])
    COEF   = (1/(4*np.pi))*(RVEC/SQUARE) * (R0R1/R1 - R0R2/R2)
    return COEF

def vortex_to_inf_l(X,Y,Z,X1,Y1,Z1,tw):
    DENUM =  (Z-Z1)**2 + (Y1-Y)**2
    XVEC  = -(Y1-Y)*np.sin(tw)/DENUM
    YVEC  =  (Z-Z1)/DENUM
    ZVEC  =  (Y1-Y)*np.cos(tw)/DENUM
    BRAC  =  1 + ((X-X1) / (np.sqrt((X-X1)**2+ (Y-Y1)**2+ (Z-Z1)**2)))
    RVEC   = np.array([XVEC, YVEC, ZVEC])
    COEF  = (1/(4*np.pi))*RVEC*BRAC         
    return COEF

def vortex_to_inf_r(X,Y,Z,X1,Y1,Z1,tw):
    DENUM =  (Z-Z1)**2 + (Y1-Y)**2
    XVEC  = (Y1-Y)*np.sin(tw)/DENUM
    YVEC  = -(Z-Z1)/DENUM
    ZVEC  = -(Y1-Y)*np.cos(tw)/DENUM
    BRAC  =  1 + ((X-X1) / (np.sqrt((X-X1)**2+ (Y-Y1)**2+ (Z-Z1)**2)))
    RVEC   = np.array([XVEC, YVEC, ZVEC])
    COEF  = (1/(4*np.pi))*RVEC*BRAC         
    return COEF
