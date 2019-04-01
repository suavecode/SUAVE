## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# weissinger_VLM.py
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

# ----------------------------------------------------------------------
#  Weissinger Vortex Lattice
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def weissinger_wing_VLM(conditions,configuration,wing):
    """Uses the vortex lattice method to compute the lift coefficient and induced drag component

    Assumptions:
    None

    Source:
    An Introduction to Theoretical and Computational Aerodynamics by Jack Moran

    Inputs:
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
    configuration.number_panels_spanwise    [Unitless]
    configuration.number_panels_chordwise   [Unitless]
    conditions.aerodynamics.angle_of_attack [radians]

    Outputs:
    Cl                                      [Unitless]
    Cd                                      [Unitless]

    Properties Used:
    N/A
    """ 

    #unpack
    span        = wing.spans.projected
    root_chord  = wing.chords.root
    tip_chord   = wing.chords.tip
    sweep       = wing.sweeps.quarter_chord
    taper       = wing.taper
    twist_rc    = wing.twists.root
    twist_tc    = wing.twists.tip
    sym_para    = wing.symmetric
    Sref        = wing.areas.reference
    orientation = wing.vertical
    
    n_sw  = configuration.number_panels_spanwise
    
    # conditions
    aoa = conditions.aerodynamics.angle_of_attack
    
    # chord difference
    dchord = (root_chord-tip_chord)
    if sym_para is True :
        span = span/2
        
    delta_y  = span/n_sw    
    sin_aoa = np.sin(aoa)
    cos_aoa = np.cos(aoa)

    if orientation == False :

        # Determine if wing segments are defined  
        n_segments           = len(wing.Segments.keys())
        segment_vortex_index = np.zeros(n_segments)
        # If spanwise stations are setup
        if n_segments>0:
            # discretizing the wing sections into panels
            i             = np.arange(0,n_sw)
            j             = np.arange(0,n_sw+1)
            y_coordinates = (j)*delta_y             
            segment_chord = np.zeros(n_segments)
            segment_twist = np.zeros(n_segments)
            segment_sweep = np.zeros(n_segments)
            segment_span  = np.zeros(n_segments)
            segment_chord_x_offset = np.zeros(n_segments)
            section_stations       = np.zeros(n_segments)
            
            # obtain chord and twist at the beginning/end of each segment
            for i_seg in range(n_segments):                
                segment_chord[i_seg]    = wing.Segments[i_seg].root_chord_percent*root_chord
                segment_twist[i_seg]    = wing.Segments[i_seg].twist
                segment_sweep[i_seg]    = wing.Segments[i_seg].sweeps.quarter_chord
                section_stations[i_seg] = wing.Segments[i_seg].percent_span_location*span
                
                if i_seg == 0:
                    segment_span[i_seg]           = 0.0
                    segment_chord_x_offset[i_seg] = 0.25*root_chord # weissinger uses quarter chord as reference
                else:
                    segment_span[i_seg]           = wing.Segments[i_seg].percent_span_location*span - wing.Segments[i_seg-1].percent_span_location*span
                    segment_chord_x_offset[i_seg] = segment_chord_x_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_sweep[i_seg-1])
            
            # shift spanwise vortices onto section breaks 
            for i_seg in range(n_segments):
                idx =  (np.abs(y_coordinates-section_stations[i_seg])).argmin()
                y_coordinates[idx] = section_stations[i_seg]
            
            # define y coordinates of horseshoe vortices      
            ya     = np.atleast_2d(y_coordinates[i])           
            yb     = np.atleast_2d(y_coordinates[i+1])          
            delta_y = y_coordinates[i+1] - y_coordinates[i]
            xa     = np.zeros(n_sw)
            x      = np.zeros(n_sw)
            y      = np.zeros(n_sw)
            section_twist   = np.zeros(n_sw)
            local_wing_chord = np.zeros(n_sw)
            
            # define coordinates of horseshoe vortices and control points
            i_seg = 0
            for idx in range(n_sw):
                section_twist[idx]   =  segment_twist[i_seg] + ((yb[0][idx] - delta_y[idx]/2 - section_stations[i_seg]) * (segment_twist[i_seg+1] - segment_twist[i_seg])/segment_span[i_seg+1])     
                local_wing_chord[idx] =  segment_chord[i_seg] + ((yb[0][idx] - delta_y[idx]/2 - section_stations[i_seg]) * (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1])
                xa[idx]             = segment_chord_x_offset[i_seg] + (yb[0][idx] - delta_y[idx]/2 - section_stations[i_seg])*np.tan(segment_sweep[i_seg])                                                    # computer quarter chord points for each horseshoe vortex
                x[idx]              = segment_chord_x_offset[i_seg] + (yb[0][idx] - delta_y[idx]/2 - section_stations[i_seg])*np.tan(segment_sweep[i_seg])  + 0.5*local_wing_chord[idx]                         # computer three-quarter chord control points for each horseshoe vortex
                y[idx]              = (yb[0][idx] -  delta_y[idx]/2)                
                
                if y_coordinates[idx] == wing.Segments[i_seg+1].percent_span_location*span: 
                    i_seg += 1                    
                if y_coordinates[idx+1] == span:
                    continue
                                  
            ya = np.atleast_2d(ya)  # y coordinate of start of horseshoe vortex on panel
            yb = np.atleast_2d(yb)  # y coordinate of end horseshoe vortex on panel
            xa = np.atleast_2d(xa)  # x coordinate of horseshoe vortex on panel
            x  = np.atleast_2d(x)   # x coordinate of control points on panel
            y  = np.atleast_2d(y)   # y coordinate of control points on panel
            
            RHS  = np.atleast_2d(np.sin(section_twist+aoa))  
   
        else:   # no segments defined on wing 
            # discretizing the wing sections into panels 
            i              = np.arange(0,n_sw)
            local_wing_chord = dchord/span*(span-(i+1)*delta_y+delta_y/2) + tip_chord
            section_twist   = twist_rc + i/float(n_sw)*(twist_tc-twist_rc)
            
            ya   = np.atleast_2d((i)*delta_y)                                                  # y coordinate of start of horseshoe vortex on panel
            yb   = np.atleast_2d((i+1)*delta_y)                                                # y coordinate of end horseshoe vortex on panel
            xa   = np.atleast_2d(((i+1)*delta_y-delta_y/2)*np.tan(sweep) + 0.25*local_wing_chord) # x coordinate of horseshoe vortex on panel
            x    = np.atleast_2d(((i+1)*delta_y-delta_y/2)*np.tan(sweep) + 0.75*local_wing_chord) # x coordinate of control points on panel
            y    = np.atleast_2d(((i+1)*delta_y-delta_y/2))                                     # y coordinate of control points on panel 
                    
            RHS  = np.atleast_2d(np.sin(section_twist+aoa))                                  
                
        
        A = (whav(x,y,xa.T,ya.T)-whav(x,y,xa.T,yb.T)\
            -whav(x,y,xa.T,-ya.T)+whav(x,y,xa.T,-yb.T))*0.25/np.pi
    
        # Vortex strength computation by matrix inversion
        T = np.linalg.solve(A.T,RHS.T).T
        
        # Calculating the effective velocty         
        A_v = A*0.25/np.pi*T
        v   = np.sum(A_v,axis=1)
        
        Lfi = -T * (sin_aoa-v)
        Lfk =  T * cos_aoa 
        Lft = -Lfi * sin_aoa + Lfk * cos_aoa
        Dg  =  Lfi * cos_aoa + Lfk * sin_aoa
            
        L  = delta_y * Lft
        Di  = delta_y * Dg
        
        # Total lift
        LT = np.sum(L)
        DTi = np.sum(Di)
    
        CL = 2*LT/(0.5*Sref)
        CDi = 2*DTi/(0.5*Sref)     
        
    else:
        
        CL = 0.0
        CDi = 0.0    

        
    return CL, CDi 


def weissinger_vehicle_VLM(conditions,configuration,geometry):
    """Uses the vortex lattice method to compute the lift coefficient and induced drag component

    Assumptions:
    None

    Source:
    1. An Introduction to Theoretical and Computational Aerodynamics by Jack Moran
    
    2. Yahyaoui, M. "Generalized Vortex Lattice Method for Predicting Characteristics of Wings
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
    Cl                                      [Unitless]
    Cd                                      [Unitless]

    Properties Used:
    N/A
    """ 
    
    # ---------------------------------------------------------------------------------------
    # STEP 1: 
    # ---------------------------------------------------------------------------------------
    
    # unpack settings
    n_sw   = configuration.number_panels_spanwise   # per wing, if segments are defined, per segment
    n_cw   = configuration.number_panels_chordwise  # per wing, if segments are defined, per segment     
    aoa = conditions.aerodynamics.angle_of_attack   # ***may be unneessary ***      
    
    XA1 = np.empty(shape=[0,1])
    XA2 = np.empty(shape=[0,1])
    YA  = np.empty(shape=[0,1])
    ZA1 = np.empty(shape=[0,1])
    ZA2 = np.empty(shape=[0,1])
    
    XB1 = np.empty(shape=[0,1])
    XB2 = np.empty(shape=[0,1])
    YB  = np.empty(shape=[0,1])
    ZB1 = np.empty(shape=[0,1])
    ZB2 = np.empty(shape=[0,1]) 
    
    XC = np.empty(shape=[0,1])
    YC = np.empty(shape=[0,1])
    ZC = np.empty(shape=[0,1])  
    
    RHS = np.empty(shape=[0,1])
    
    for wing in geometry.wings.values():
        # ---------------------------------------------------------------------------------------
        # STEP 2: Unpack aircraft wing geometry 
        # ---------------------------------------------------------------------------------------
        span        = wing.spans.projected
        root_chord  = wing.chords.root
        tip_chord   = wing.chords.tip
        sweep       = wing.sweeps.quarter_chord
        taper       = wing.taper
        twist_rc    = wing.twists.root
        twist_tc    = wing.twists.tip
        dihedral    = wing.dihedral
        sym_para    = wing.symmetric
        Sref        = wing.areas.reference
        orientation = wing.vertical
        wing_origin = wing.origin
        

        # determine of vehicle has symmetry 
        if sym_para is True :
            span = span/2
            
        delta_y  = span/n_sw    
        sin_aoa = np.sin(aoa) # ***may be unneessary ***
        cos_aoa = np.cos(aoa) # ***may be unneessary ***
        
        # condition for exclusion of vertical tail
        num_wing = 0
        if orientation == False:
            num_wing += 1
            # ---------------------------------------------------------------------------------------
            # STEP 3: Determine if wing segments are defined  
            # ---------------------------------------------------------------------------------------
            n_segments           = len(wing.Segments.keys())
            
            if n_segments>0:
                # ---------------------------------------------------------------------------------------
                # STEP 4A: Discretizing the wing sections into panels
                # ---------------------------------------------------------------------------------------
                i             = np.arange(0,n_cw+1)
                j             = np.arange(0,n_sw+1)
                y_coordinates = (j)*delta_y  
                
                segment_chord          = np.zeros(n_segments)
                segment_twist          = np.zeros(n_segments)
                segment_sweep          = np.zeros(n_segments)
                segment_span           = np.zeros(n_segments)
                segment_dihedral       = np.zeros(n_segments)
                segment_x_coord        = np.zeros(n_segments) 
                segment_camber         = np.zeros(n_segments)
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
                            chord_fraction       = 0.25                          
                            seg_root_chord       = root_chord*wing.Segments[i_seg].root_chord_percent
                            seg_tip_chord        = root_chord*wing.Segments[i_seg+1].root_chord_percent
                            seg_span             = span*(wing.Segments[i_seg+1].percent_span_location - wing.Segments[i_seg].percent_span_location )
                            segment_sweep[i_seg] = np.arctan(((seg_root_chord*chord_fraction) + (np.tan(sweep_quarter_chord)*seg_span - chord_fraction*seg_tip_chord)) /seg_span)  
                    
                    if i_seg == 0:
                        segment_span[i_seg]           = 0.0
                        segment_chord_x_offset[i_seg] = 0.0  
                        segment_chord_z_offset[i_seg] = 0.0
                    else:
                        segment_span[i_seg]           = wing.Segments[i_seg].percent_span_location*span - wing.Segments[i_seg-1].percent_span_location*span
                        segment_chord_x_offset[i_seg] = segment_chord_x_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_sweep[i_seg-1])
                        segment_chord_x_offset[i_seg] = segment_chord_z_offset[i_seg-1] + segment_span[i_seg]*np.tan(segment_dihedral[i_seg-1])
                    
                    # Get airfoil section data  
                    if wing.Segments[i_seg].Airfoil: 
                        airfoil_data = read_wing_airfoil(wing.Segments[i_seg].Airfoil.aifoil.tag)    
                        segment_camber[i_seg] = airfoil_data.camber_coordinates 
                        segment_x_coord[i_seg] = airfoil_data.x_coordinates 
                    else:
                        segment_camber[i_seg] = np.zeros(61) # dimension of Selig airfoil data file
                        segment_x_coord[i_seg] = airfoil_data.x_coordinates
                   
                    # ** TO DO ** Get flap/aileron locations and deflection
                
                #Shift spanwise vortices onto section breaks  
                for i_seg in range(n_segments):
                    idx =  (np.abs(y_coordinates-section_stations[i_seg])).argmin()
                    y_coordinates[idx] = section_stations[i_seg]                
                
                # ---------------------------------------------------------------------------------------
                # STEP 6A: Define coordinates of panels horseshoe vortices and control points 
                # ---------------------------------------------------------------------------------------
                delta_y = y_coordinates[i+1] - y_coordinates[i]
                
                xa1  = np.zeros(n_sw*n_cw)
                xa2  = np.zeros(n_sw*n_cw)
                za1  = np.zeros(n_sw*n_cw)
                ya     = np.atleast_2d(y_coordinates[i]) 
                za2  = np.zeros(n_sw*n_cw)
                
                xb1  = np.zeros(n_sw*n_cw)
                xb2  = np.zeros(n_sw*n_cw)  
                yb     = np.atleast_2d(y_coordinates[i+1])
                zb1  = np.zeros(n_sw*n_cw)
                zb2  = np.zeros(n_sw*n_cw)
                
                xc   = np.zeros(n_sw*n_cw)
                yc   = np.zeros(n_sw*n_cw)
                zc   = np.zeros(n_sw*n_cw)
                
                section_twist= np.zeros(n_sw) 
                
                # define coordinates of horseshoe vortices and control points
                i_seg = 0
                idx = 0
                for idx_y in range(n_sw):                    
                    for idx_x in range(n_cw):
                        eta_a = (ya[0][idx_y] - section_stations[i_seg])  
                        eta_b = (yb[0][idx_y] - section_stations[i_seg]) 
                        eta   = (yb[0][idx_y] - delta_y[idx_y]/2 - section_stations[i_seg]) 
                        
                        segment_chord_ratio = (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1]
                        segment_twist_ratio = (segment_twist[i_seg+1] - segment_twist[i_seg])/segment_span[i_seg+1]
                        
                        wing_chord_section_a  = segment_chord[i_seg] + (eta_a*segment_chord_ratio) 
                        wing_chord_section_b  = segment_chord[i_seg] + (eta_b*segment_chord_ratio)
                        wing_chord_section  = segment_chord[i_seg]   + (eta*segment_chord_ratio)
                        
                        delta_x_a = wing_chord_section_a/n_cw   
                        delta_x_b = wing_chord_section_b/n_cw   
                        delta_x = wing_chord_section/n_cw                                  
                        
                        xi_a1 = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x                # x coordinate of top left corner of panel
                        xi_a2 = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x + delta_x_a    # x coordinate of bottom left corner of panel
                        xi_b1 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x                # x coordinate of top right corner of panel        
                        xi_b2 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x + delta_x_b    # x coordinate of bottom right corner of panel                       
                        xi    = segment_chord_x_offset[i_seg]  + eta *np.tan(segment_sweep[i_seg])  + delta_x  *idx_x + delta_x*0.75 # x coordinate three-quarter chord control point for each panel                        

                        # camber
                        section_camber_a  = segment_camber[i_seg]*wing_chord_section_a  
                        section_camber_b  = segment_camber[i_seg]*wing_chord_section_b  
                        section_camber    = segment_camber[i_seg]*wing_chord_section
                        
                        section_x_coord_a = segment_x_coord[i_seg]*wing_chord_section_a
                        section_x_coord_b = segment_x_coord[i_seg]*wing_chord_section_b
                        section_x_coord   = segment_x_coord[i_seg]*wing_chord_section
                        
                        z_c_a1 = np.interp(xi_a1,section_x_coord_a[i_seg],section_camber_a) 
                        z_c_a2 = np.interp(xi_a2,section_x_coord_a[i_seg],section_camber_a) 
                        z_c_b1 = np.interp(xi_b1,section_x_coord_b[i_seg],section_camber_b)                             
                        z_c_b2 = np.interp(xi_b2,section_x_coord_b[i_seg],section_camber_b) 
                        z_c    = np.interp(xi,section_x_coord[i_seg],section_camber) 
                        
                        zeta_a1 = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a1  # z coordinate of top left corner of panel
                        zeta_a2 = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a2  # z coordinate of bottom left corner of panel
                        zeta_b1 = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b1  # z coordinate of top right corner of panel        
                        zeta_b2 = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b2  # z coordinate of bottom right corner of panel                 
                        zeta    = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])    + z_c     # z coordinate three-quarter chord control point for each panel
                        
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
                        
                        xi_prime_a1  = xi_LE_a + np.cos(-section_twist_a)*(xi_a1-xi_LE_a) + np.sin(-section_twist_a)*(zeta_a1-zeta_LE_a)   # x coordinate transformation of top left corner
                        xi_prime_a2  = xi_LE_a + np.cos(-section_twist_a)*(xi_a2-xi_LE_a) + np.sin(-section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner
                        xi_prime_b1  = xi_LE_b + np.cos(-section_twist_b)*(xi_b1-xi_LE_b) + np.sin(-section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner 
                        xi_prime_b2  = xi_LE_b + np.cos(-section_twist_b)*(xi_b2-xi_LE_b) + np.sin(-section_twist_b)*(zeta_b2-zeta_LE_b)   # x coordinate transformation of botton right corner 
                        xi_prime     = xi_LE + np.cos(-section_twist)*(xi-xi_LE) + np.sin(-section_twist)*(zeta-zeta_LE)                   # x coordinate transformation of control point
                        
                        zeta_prime_a1  = zeta_LE_a - np.sin(-section_twist_a)*(xi_a1-xi_LE_a) + np.cos(-section_twist_a)*(zeta_a1-zeta_LE_a) # z coordinate transformation of top left corner
                        zeta_prime_a2  = zeta_LE_a - np.sin(-section_twist_a)*(xi_a2-xi_LE_a) + np.cos(-section_twist_a)*(zeta_a2-zeta_LE_a) # z coordinate transformation of bottom left corner
                        zeta_prime_b1  = zeta_LE_b - np.sin(-section_twist_b)*(xi_b1-xi_LE_b) + np.cos(-section_twist_b)*(zeta_b1-zeta_LE_b) # z coordinate transformation of top right corner 
                        zeta_prime_b2  = zeta_LE_b - np.sin(-section_twist_b)*(xi_b2-xi_LE_b) + np.cos(-section_twist_b)*(zeta_b2-zeta_LE_b) # z coordinate transformation of botton right corner 
                        zeta_prime     = zeta_LE - np.sin(-section_twist)*(xi-xi_LE) + np.cos(-section_twist)*(zeta-zeta_LE)                 # z coordinate transformation of control point
                        
                        # ** TO DO ** Get flap/aileron locations and deflection
                        
                        # store coordinates of panels, horseshoeces vortices and control points relative to wing root 
                        xa1[idx] = xi_prime_a1 
                        xa2[idx] = xi_prime_a2 
                        za1[idx] = zeta_prime_a1 
                        za2[idx] = zeta_prime_a2 
                        
                        xb1[idx] = xi_prime_b1 
                        xb2[idx] = xi_prime_b2 
                        zb1[idx] = zeta_prime_b1 
                        zb2[idx] = zeta_prime_b2 
                        
                        xc[idx] = xi_prime 
                        yc[idx] = yb[0][idx_y] - delta_y[idx_y]/2   
                        zc[idx] = zeta_prime                                                            
                                                
                    if y_coordinates[idx] == wing.Segments[i_seg+1].percent_span_location*span: 
                        i_seg += 1                    
                    if y_coordinates[idx+1] == span:
                        continue
                
                # adjusting coordinate axis so reference point is at the nose of the aircraft
                xa1 = xa1 + wing_origin[0] # x coordinate of top left corner of horseshoe vortex on panel
                xa2 = xa2 + wing_origin[0] # x coordinate of bottom left corner of horseshoe vortex on panel
                ya =  ya  + wing_origin[1] # y coordinate of horseshoe vortex on panel
                za1 = za1 + wing_origin[2] # z coordinate of top left corner of horseshoe vortex on panel
                za2 = za2 + wing_origin[2] # z coordinate of bottom left corner of horseshoe vortex on panel  
                
                xb1 = xb1 + wing_origin[0] # x coordinate of top right corner of horseshoe vortex on panel   
                xb2 = xb2 + wing_origin[0] # x coordinate of bottom rightcorner of horseshoe vortex on panel                
                yb  = yb  + wing_origin[1] # y coordinate of horseshoe vortex on panel
                zb1 = xb1 + wing_origin[2] # z coordinate of top right corner of horseshoe vortex on panel   
                zb2 = xb2 + wing_origin[2] # z coordinate of bottom right corner of horseshoe vortex on panel                   
                              
                xc = xc + wing_origin[0] # x coordinate of control points on panel
                yc = yc + wing_origin[1] # y coordinate of control points on panel
                zc = zc + wing_origin[2] # y coordinate of control points on panel
                
                # store wing in vehicle vector
                XA1  = np.append(XA1,xa1[:])
                XA2  = np.append(XA2,xa2[:])
                YA   = np.append(YA,ya[:])
                ZA1  = np.append(ZA1,za1[:])
                ZA2  = np.append(ZA2,za2[:])
                
                XB1  = np.append(XB1,xb1[:])
                XB2  = np.append(XB2,xb2[:])                
                YB   = np.append(YB,yb[:])
                ZB1  = np.append(ZB1,zb1[:])
                ZB2  = np.append(ZB2,zb2[:])                
                  
                XC   = np.append(XC,xc[:])
                YC   = np.append(YC,yc[:])
                ZC   = np.append(ZC,yc[:])
                
                #RHS = np.append(RHS,rhs[:])                             
                #rhs = np.atleast_2d(np.sin(section_twist+aoa)) 
       
            else:   # no segments defined on wing  
                # ---------------------------------------------------------------------------------------
                # STEP 6B: Define coordinates of panels horseshoe vortices and control points 
                # ---------------------------------------------------------------------------------------
                i              = np.arange(0,n_sw)
                delta_y = y_coordinates[i+1] - y_coordinates[i]
            
                xa1  = np.zeros(n_sw*n_cw)
                xa2  = np.zeros(n_sw*n_cw)
                za1  = np.zeros(n_sw*n_cw)
                ya   = np.atleast_2d((i)*delta_y) 
                za2  = np.zeros(n_sw*n_cw)
            
                xb1  = np.zeros(n_sw*n_cw)
                xb2  = np.zeros(n_sw*n_cw)  
                yb   = np.atleast_2d((i+1)*delta_y)
                zb1  = np.zeros(n_sw*n_cw)
                zb2  = np.zeros(n_sw*n_cw)
            
                xc   = np.zeros(n_sw*n_cw)
                yc   = np.zeros(n_sw*n_cw)
                zc   = np.zeros(n_sw*n_cw)
            
                section_twist= np.zeros(n_sw) 
                       
                wing_chord_ratio = (tip_chord-root_chord)/span
                wing_twist_ratio = (twist_tc-twist_rc)/span    
                
                # Get airfoil section data  
                if wing.Airfoil: 
                    airfoil_data = read_wing_airfoil(wing.Segments[i_seg].Airfoil.aifoil.tag)    
                    wing_camber  = airfoil_data.camber_coordinates * section_stations[i_seg]
                    wing_x_coord = airfoil_data.x_coordinates * section_stations[i_seg] 
                    
                for idx_y in range(n_sw):                    
                    for idx_x in range(n_cw):  
                        eta_a = (ya[0][idx_y])  
                        eta_b = (yb[0][idx_y]) 
                        eta   = (yb[0][idx_y] - delta_y[idx_y]/2) 
                         
                        wing_chord_section_a  = root_chord + (eta_a*wing_chord_ratio) 
                        wing_chord_section_b  = root_chord + (eta_b*wing_chord_ratio)
                        wing_chord_section    = root_chord + (eta*wing_chord_ratio)
                        
                        delta_x_a = wing_chord_section_a/n_cw   
                        delta_x_b = wing_chord_section_b/n_cw   
                        delta_x = wing_chord_section/n_cw                                  
                        
                        xi_a1 = eta_a*np.tan(sweep) + delta_x_a*idx_x                # x coordinate of top left corner of panel
                        xi_a2 = eta_a*np.tan(sweep) + delta_x_a*idx_x + delta_x_a    # x coordinate of bottom left corner of panel
                        xi_b1 = eta_b*np.tan(sweep) + delta_x_b*idx_x                # x coordinate of top right corner of panel        
                        xi_b2 = eta_b*np.tan(sweep) + delta_x_b*idx_x + delta_x_b    # x coordinate of bottom right corner of panel                       
                        xi    = eta *np.tan(sweep)  + delta_x  *idx_x + delta_x*0.75 # x coordinate three-quarter chord control point for each panel                        

                        # camber
                        wing_camber_a  = wing_camber*wing_chord_section_a  
                        wing_camber_b  = wing_camber*wing_chord_section_b  
                        wing_camber    = wing_camber*wing_chord_section
                        
                        section_x_coord_a = wing_x_coord*wing_chord_section_a
                        section_x_coord_b = wing_x_coord*wing_chord_section_b
                        section_x_coord   = wing_x_coord*wing_chord_section
                        
                        z_c_a1 = np.interp(xi_a1,section_x_coord_a,wing_camber_a) 
                        z_c_a2 = np.interp(xi_a2,section_x_coord_a,wing_camber_a) 
                        z_c_b1 = np.interp(xi_b1,section_x_coord_b,wing_camber_b)                             
                        z_c_b2 = np.interp(xi_b2,section_x_coord_b,wing_camber_b) 
                        z_c    = np.interp(xi,section_x_coord,wing_camber) 
                        
                        zeta_a1 =  eta_a*np.tan(dihedral)  + z_c_a1  # z coordinate of top left corner of panel
                        zeta_a2 =  eta_a*np.tan(dihedral)  + z_c_a2  # z coordinate of bottom left corner of panel
                        zeta_b1 =  eta_b*np.tan(dihedral)  + z_c_b1  # z coordinate of top right corner of panel        
                        zeta_b2 =  eta_b*np.tan(dihedral)  + z_c_b2  # z coordinate of bottom right corner of panel                 
                        zeta    =  eta*np.tan(dihedral)    + z_c     # z coordinate three-quarter chord control point for each panel
                        
                        # adjustment of panels for twist  
                        xi_LE_a = eta_a*np.tan(sweep)               # x location of leading edge left corner of wing
                        xi_LE_b = eta_b*np.tan(sweep)               # x location of leading edge right of wing
                        xi_LE   = eta*np.tan(sweep)                 # x location of leading edge center of wing
                        
                        zeta_LE_a = eta_a*np.tan(dihedral)          # z location of leading edge left corner of wing
                        zeta_LE_b = eta_b*np.tan(dihedral)          # z location of leading edge right of wing
                        zeta_LE   = eta*np.tan(dihedral)            # z location of leading edge center of wing
                        
                        # determine section twist
                        section_twist_a = twist_rc + (eta_a * wing_twist_ratio)                     # twist at left side of panel
                        section_twist_b = twist_rc + (eta_b * wing_twist_ratio)                      # twist at right side of panel
                        section_twist   = twist_rc + (eta * wing_twist_ratio)                       # twist at center local chord 
                        
                        xi_prime_a1  = xi_LE_a + np.cos(-section_twist_a)*(xi_a1-xi_LE_a) + np.sin(-section_twist_a)*(zeta_a1-zeta_LE_a)   # x coordinate transformation of top left corner
                        xi_prime_a2  = xi_LE_a + np.cos(-section_twist_a)*(xi_a2-xi_LE_a) + np.sin(-section_twist_a)*(zeta_a2-zeta_LE_a)   # x coordinate transformation of bottom left corner
                        xi_prime_b1  = xi_LE_b + np.cos(-section_twist_b)*(xi_b1-xi_LE_b) + np.sin(-section_twist_b)*(zeta_b1-zeta_LE_b)   # x coordinate transformation of top right corner 
                        xi_prime_b2  = xi_LE_b + np.cos(-section_twist_b)*(xi_b2-xi_LE_b) + np.sin(-section_twist_b)*(zeta_b2-zeta_LE_b)   # x coordinate transformation of botton right corner 
                        xi_prime     = xi_LE + np.cos(-section_twist)*(xi-xi_LE) + np.sin(-section_twist)*(zeta-zeta_LE)                   # x coordinate transformation of control point
                        
                        zeta_prime_a1  = zeta_LE_a - np.sin(-section_twist_a)*(xi_a1-xi_LE_a) + np.cos(-section_twist_a)*(zeta_a1-zeta_LE_a) # z coordinate transformation of top left corner
                        zeta_prime_a2  = zeta_LE_a - np.sin(-section_twist_a)*(xi_a2-xi_LE_a) + np.cos(-section_twist_a)*(zeta_a2-zeta_LE_a) # z coordinate transformation of bottom left corner
                        zeta_prime_b1  = zeta_LE_b - np.sin(-section_twist_b)*(xi_b1-xi_LE_b) + np.cos(-section_twist_b)*(zeta_b1-zeta_LE_b) # z coordinate transformation of top right corner 
                        zeta_prime_b2  = zeta_LE_b - np.sin(-section_twist_b)*(xi_b2-xi_LE_b) + np.cos(-section_twist_b)*(zeta_b2-zeta_LE_b) # z coordinate transformation of botton right corner 
                        zeta_prime     = zeta_LE - np.sin(-section_twist)*(xi-xi_LE) + np.cos(-section_twist)*(zeta-zeta_LE)                 # z coordinate transformation of control point
                        
                        # ** TO DO ** Get flap/aileron locations and deflection
                        
                        # store coordinates of panels, horseshoeces vortices and control points relative to wing root 
                        xa1[idx] = xi_prime_a1 
                        xa2[idx] = xi_prime_a2 
                        za1[idx] = zeta_prime_a1 
                        za2[idx] = zeta_prime_a2 
                        
                        xb1[idx] = xi_prime_b1 
                        xb2[idx] = xi_prime_b2 
                        zb1[idx] = zeta_prime_b1 
                        zb2[idx] = zeta_prime_b2 
                        
                        xc[idx] = xi_prime 
                        yc[idx] = yb[0][idx_y] - delta_y[idx_y]/2   
                        zc[idx] = zeta_prime                                                            
                                                               
                # adjusting coordinate axis so reference point is at the nose of the aircraft
                xa1 = xa1 + wing_origin[0] # x coordinate of top left corner of horseshoe vortex on panel
                xa2 = xa2 + wing_origin[0] # x coordinate of bottom left corner of horseshoe vortex on panel
                ya =  ya  + wing_origin[1] # y coordinate of horseshoe vortex on panel
                za1 = za1 + wing_origin[2] # z coordinate of top left corner of horseshoe vortex on panel
                za2 = za2 + wing_origin[2] # z coordinate of bottom left corner of horseshoe vortex on panel  
                
                xb1 = xb1 + wing_origin[0] # x coordinate of top right corner of horseshoe vortex on panel   
                xb2 = xb2 + wing_origin[0] # x coordinate of bottom rightcorner of horseshoe vortex on panel                
                yb  = yb  + wing_origin[1] # y coordinate of horseshoe vortex on panel
                zb1 = xb1 + wing_origin[2] # z coordinate of top right corner of horseshoe vortex on panel   
                zb2 = xb2 + wing_origin[2] # z coordinate of bottom right corner of horseshoe vortex on panel                   
                              
                xc = xc + wing_origin[0] # x coordinate of control points on panel
                yc = yc + wing_origin[1] # y coordinate of control points on panel
                zc = zc + wing_origin[2] # y coordinate of control points on panel
                
                # store wing in vehicle vector
                XA1  = np.append(XA1,xa1[:])
                XA2  = np.append(XA2,xa2[:])
                YA   = np.append(YA,ya[:])
                ZA1  = np.append(ZA1,za1[:])
                ZA2  = np.append(ZA2,za2[:])
                
                XB1  = np.append(XB1,xb1[:])
                XB2  = np.append(XB2,xb2[:])                
                YB   = np.append(YB,yb[:])
                ZB1  = np.append(ZB1,zb1[:])
                ZB2  = np.append(ZB2,zb2[:])                
                  
                XC   = np.append(XC,xc[:])
                YC   = np.append(YC,yc[:])
                ZC   = np.append(ZC,yc[:])
                
                #RHS = np.append(RHS,rhs[:])                             
                #rhs = np.atleast_2d(np.sin(section_twist+aoa)) 

    XA1 = np.atleast_2d(XA1)
    XA2 = np.atleast_2d(XA2)
    YA  = np.atleast_2d(YA)
    ZA1 = np.atleast_2d(ZA1)
    ZA2 = np.atleast_2d(ZA2)
    
    XB1 = np.atleast_2d(XB1)
    XB2 = np.atleast_2d(XB2)
    YB  = np.atleast_2d(YB)
    ZB1 = np.atleast_2d(ZB1)
    ZB2 = np.atleast_2d(ZB2)
        
    XC = np.atleast_2d(XC)
    YC = np.atleast_2d(YC)
    ZC = np.atleast_2d(ZC)
    
    #RHS = np.atleast_2d(RHS)
    
    # ---------------------------------------------------------------------------------------
    # STEP 7: Compute velocity induced by horseshoe vortex segments on every control point by 
    #         every panel
    # ---------------------------------------------------------------------------------------    
    num_cp = n_cw*n_sw*num_wing
    
    F1a_H = np.zeros((num_cp,num_cp))
    F2a_H = np.zeros((num_cp,num_cp))
    F1b_H = np.zeros((num_cp,num_cp))
    F2b_H = np.zeros((num_cp,num_cp))
    F1a   = np.zeros((num_cp,num_cp))
    F2a   = np.zeros((num_cp,num_cp))
    F1b   = np.zeros((num_cp,num_cp))
    F2b   = np.zeros((num_cp,num_cp))
    
    
    for m in range(num_cp): # control point m
        for n in range(num_cp): # panel n 
            # condition if horseshoe vortex starts on panel
            
            # left leg of vortex 
            XA_H = XA1[n] + (XA2[n]-XA1[n])*0.25 # horseshoe starts at quarter chord of panel               
            YA_H = YA[n]  
            ZA_H = ZA1[n] + (ZA2[n]-ZA1[n])*0.25 # horseshoe starts at quarter chord of panel
            
            num_a_H   = [((YC[m] - YA_H)*(ZC[m] - ZA2[n]) - (YC[m] - YB[n])*(ZC[m] - ZA_H)) , - ((XC[m] - XA_H)*(ZC[m] - ZA2[n]) - (XC[m] - XA2[n])*(ZC[m] - ZA_H))   ,  ((XC[m] - XA_H)*(YC[m] - YB[n]) - (XC[m] - XA2[n])*(YC[m] - YA_H))] 
            denum_a_H = (((YC[m] - YA_H)*(ZC[m] - ZA2[n]) - (YC[m]- YB[n])*(ZC[m] - ZA_H))**2 + ((XC[m] - XA_H)*(ZC[m] - ZA2[n]) - (XC[m] - XA2[n])*(ZC[m] - ZA_H))**2 + ((XC[m] - XA_H)*(YC[m] - YB[n]) - (XC[m] - XA2[n])*(YC[m] - YA_H))**2)
            F1a_H[m,n]  = num_a_H/denum_a_H
            
            F2a_H[m,n]  =  ((XA2[n] - XA_H)*(XC[m] - XA_H) + (YB[n] - YA_H)*(YC[m] - YA_H) + (ZA2[n] - ZA_H)*(ZC[m] - ZA_H))/(np.sqrt((XC[m] - XA_H)**2 + (YC[m] - YA_H)**2 + (ZC[m] - ZA_H)**2 )) - \
                      ((XA2[n] - XA_H)*(XC[m] - XA2[n])    + (YB[n] - YA_H)*(YC[m] - YB[n])    + (ZA2[n] - ZA_H)*(ZC[m] - ZA2[n]))   /(np.sqrt((XC[m] - XA2[n])**2    + (YC[m] - YB[n])**2    + (ZC[m] - ZA2[n])**2 ))
            
            # right leg of vortex 
            XB_H = XB1[n] + (XB2[n]-XB1[n])*0.25 # horseshoe starts at quarter chord of panel               
            YB_H = YB[n]  
            ZB_H = ZB1[n] + (ZB2[n]-ZB1[n])*0.25 # horseshoe starts at quarter chord of panel
            
            num_b_H =   [((YC[m] - YB_H)*(ZC[m] - ZB2[n]) - (YC[m] - YB[n])*(ZC[m] - ZB_H)) , - ((XC[m] - XB_H)*(ZC[m] - ZB2[n]) - (XC[m] - XB2[n])*(ZC[m] - ZB_H))   ,  ((XC[m] - XB_H)*(YC[m] - YB[n]) - (XC[m] - XB2[n])*(YC[m] - YB_H))] 
            denum_b_H = (((YC[m] - YB_H)*(ZC[m] - ZB2[n]) - (YC[m]- YB[n])*(ZC[m] - ZB_H))**2 + ((XC[m] - XB_H)*(ZC[m] - ZB2[n]) - (XC[m] - XB2[n])*(ZC[m] - ZB_H))**2 + ((XC[m] - XB_H)*(YC[m] - YB[n]) - (XC[m] - XB2[n])*(YC[m] - YB_H))**2)
            F1b_H[m,n]  = num_b_H/denum_b_H
            
            F2b_H[m,n]  =  ((XB2[n] - XB_H)*(XC[m] - XB_H) + (YB[n] - YB_H)*(YC[m] - YB_H) + (ZB2[n] - ZB_H)*(ZC[m] - ZB_H))/(np.sqrt((XC[m] - XB_H)**2 + (YC[m] - YB_H)**2 + (ZC[m] - ZB_H)**2 )) - \
                      ((XB2[n] - XB_H)*(XC[m] - XB2[n])    + (YB[n] - YB_H)*(YC[m] - YB[n])    + (ZB2[n] - ZB_H)*(ZC[m] - ZB2[n]))   /(np.sqrt((XC[m] - XB2[n])**2    + (YC[m] - YB[n])**2    + (ZC[m] - ZB2[n])**2 ))
            
            
            # condition if horseshoe vortex does not start on panel
            # left leg of vortex
            num_a = [((YC[m] - YA[n])(ZC[m] - ZA2[n]) - (YC[m] - YB[n])(ZC[m] - ZA1[n])) , -((XC[m] - XA1[n])(ZC[m] - ZA2[n]) - (XC[m] - XA2[n])(ZC[m] - ZA1[n])) , ((XC[m] - XA1[n])(YC[m] - YB[n]) - (XC[m] - XA2[n])(YC[m] - YA[n]))] 
            denum_a = (((YC[m] - YA[n])(ZC[m] - ZA2[n]) - (YC[m] - YB[n])(ZC[m] - ZA1[n]))**2 + ((XC[m] - XA1[n])(ZC[m] - ZA2[n]) - (XC[m] - XA2[n])(ZC[m] - ZA1[n]))**2 + ((XC[m] - XA1[n])(YC[m] - YB[n]) - (XC[m] - XA2[n])(YC[m] - YA[n]))**2)
            F1a[m,n]  = num_a/denum_a
            
            F2a[m,n]  =  ((XA2[n] - XA1[n])(XC[m] - XA1[n]) + (YB[n] - YA[n])(YC[m] - YA[n]) + (ZA2[n] - ZA1[n])(ZC[m] - ZA1[n]))/(np.sqrt((XC[m] - XA1[n])**2 + (YC[m] - YA[n])**2 + (ZC[m] - ZA1[n])**2 )) - \
                     ((XA2[n] - XA1[n])(XC[m] - XA2[n])+ (YB[n] - YA[n])(YC[m] - YB[n]) + (ZA2[n] - ZA1[n])(ZC[m] - ZA2[n]))/(np.sqrt((XC[m] - XA2[n])**2 + (YC[m] - YB[n])**2 + (ZC[m] - ZA2[n])**2 ))
            
            # left leg of vortex
            num_b = [((YC[m] - YA[n])(ZC[m] - ZB2[n]) - (YC[m] - YB[n])(ZC[m] - ZB1[n])) , -((XC[m] - XB1[n])(ZC[m] - ZB2[n]) - (XC[m] - XB2[n])(ZC[m] - ZB1[n])) , ((XC[m] - XB1[n])(YC[m] - YB[n]) - (XC[m] - XB2[n])(YC[m] - YA[n]))] 
            denum_b = (((YC[m] - YA[n])(ZC[m] - ZB2[n]) - (YC[m] - YB[n])(ZC[m] - ZB1[n]))**2 + ((XC[m] - XB1[n])(ZC[m] - ZB2[n]) - (XC[m] - XB2[n])(ZC[m] - ZB1[n]))**2 + ((XC[m] - XB1[n])(YC[m] - YB[n]) - (XC[m] - XB2[n])(YC[m] - YA[n]))**2)
            F1b[m,n]  = num_b/denum_b
              
            F2b[m,n]  =  ((XB2[n] - XB1[n])(XC[m] - XB1[n]) + (YB[n] - YA[n])(YC[m] - YA[n]) + (ZB2[n] - ZB1[n])(ZC[m] - ZB1[n]))/(np.sqrt((XC[m] - XB1[n])**2 + (YC[m] - YA[n])**2 + (ZC[m] - ZB1[n])**2 )) - \
                     ((XB2[n] - XB1[n])(XC[m] - XB2[n])+ (YB[n] - YA[n])(YC[m] - YB[n]) + (ZB2[n] - ZB1[n])(ZC[m] - ZB2[n]))/(np.sqrt((XC[m] - XB2[n])**2 + (YC[m] - YB[n])**2 + (ZC[m] - ZB2[n])**2 ))
     
     
    # ---------------------------------------------------------------------------------------
    # STEP 8: Compute total velocity induced by horseshoe all vortices on every control point by 
    #         every panel
    # ---------------------------------------------------------------------------------------
    C_m_i = np.zeros((num_cp,3))
    for m in range(num_cp): # control point m
        i = 0
        C_m_i_sum  = 0  
        for j in range(n_sw*num_wing):  
            for k in range(n_cw):  
                C_m_i_sum +=  F1a_H[m,i +j*n_cw]*F2a_H[m,i +j*n_cw] + k*F1a[m,i +j*n_cw]*F2a[m,i +j*n_cw] + \
                              F1b_H[m,i +j*n_cw]*F2b_H[m,i +j*n_cw] + k* F1b[m,i +j*n_cw]*F2b[m,i +j*n_cw]     
        C_m_i[m,:] = C_m_i_sum/(2*np.pi)
             
    
    # find derivatives df/deta and df/dzeta
    # equation 22 
    # Solve VLM
    A = (whav(X,Y,XA.T,YA.T)-whav(X,Y,XA.T,YB.T)\
        -whav(X,Y,XA.T,-YA.T)+whav(X,Y,XA.T,-YB.T))*0.25/np.pi
    
            
    # Vortex strength computation by matrix inversion
    T = np.linalg.solve(A.T,RHS.T).T
    
    # Calculating the effective velocty         
    A_v = A*0.25/np.pi*T
    v   = np.sum(A_v,axis=1)
    
    Lfi = -T * (sin_aoa-v)*2
    Lfk =  T * cos_aoa *2
    Lft = -Lfi * sin_aoa + Lfk * cos_aoa
    Dg  =  Lfi * cos_aoa + Lfk * sin_aoa
        
    L   = delta_y * Lft
    Di  = delta_y * Dg
    
    # Total lift
    LT  = np.sum(L)
    DTi = np.sum(Di)

    CL  = 2*LT/(vehicle_Sref)
    #CDi = 2*DTi/(vehicle_Sref)  
    CDi = CL**2/(np.pi*AR*e)
  
    return CL, CDi 


# ----------------------------------------------------------------------
#   Helper Functions
# ----------------------------------------------------------------------
def whav(x,y,z,x1,y1,z1,x2,y2,z2):
    """ Helper function of vortex lattice method      
        Inputs:
            x , y , z - coordinates of control point 
            x1, y1,z1 - coordinates of finite segment start
            x1, y1,z1 - coordinates of finite segment end 
    """    

    use_base    = 1 - np.isclose(x1,x2)*1.
    no_use_base = np.isclose(x1,x2)*1.
    
    whv = 1/(y1-y2)*(1+ (np.sqrt((x1-x2)**2+(y1-y2)**2)/(x1-x2)))*use_base + (1/(y1 -y2))*no_use_base
    
    return whv
