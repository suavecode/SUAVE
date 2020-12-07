## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
# generate_wing_vortex_distribution.py
# 
# Created:  May 2018, M. Clarke
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import numpy as np
from SUAVE.Core import  Data
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def generate_wing_vortex_distribution(geometry,settings):
    ''' Compute the coordinates of panels, vortices , control points
    and geometry used to build the influence coefficient matrix.
    

    Assumptions: 
    Below is a schematic of the coordinates of an arbitrary panel  
    
    XA1 ____________________________ XB1    
       |                            |
       |        bound vortex        |
    XAH|  ________________________  |XBH
       | |           XCH          | |
       | |                        | |
       | |                        | |     
       | |                        | |
       | |                        | |
       | |           0 <--control | |       
       | |          XC     point  | |  
       | |                        | |
   XA2 |_|________________________|_|XB2
         |                        |     
         |       trailing         |  
         |   <--  vortex   -->    |  
         |         legs           | 
             
    
    Source:  
    None

    Inputs:
    geometry.wings                                [Unitless]  
       
    Outputs:                                   
    VD - vehicle vortex distribution              [Unitless] 

    Properties Used:
    N/A 
         
    '''
    # ---------------------------------------------------------------------------------------
    # STEP 1: Define empty vectors for coordinates of panes, control points and bound vortices
    # ---------------------------------------------------------------------------------------
    VD = Data()

    VD.XAH    = np.empty(shape=[0,1])
    VD.YAH    = np.empty(shape=[0,1])
    VD.ZAH    = np.empty(shape=[0,1])
    VD.XBH    = np.empty(shape=[0,1])
    VD.YBH    = np.empty(shape=[0,1])
    VD.ZBH    = np.empty(shape=[0,1])
    VD.XCH    = np.empty(shape=[0,1])
    VD.YCH    = np.empty(shape=[0,1])
    VD.ZCH    = np.empty(shape=[0,1])     
    VD.XA1    = np.empty(shape=[0,1])
    VD.YA1    = np.empty(shape=[0,1])  
    VD.ZA1    = np.empty(shape=[0,1])
    VD.XA2    = np.empty(shape=[0,1])
    VD.YA2    = np.empty(shape=[0,1])    
    VD.ZA2    = np.empty(shape=[0,1])    
    VD.XB1    = np.empty(shape=[0,1])
    VD.YB1    = np.empty(shape=[0,1])  
    VD.ZB1    = np.empty(shape=[0,1])
    VD.XB2    = np.empty(shape=[0,1])
    VD.YB2    = np.empty(shape=[0,1])    
    VD.ZB2    = np.empty(shape=[0,1])     
    VD.XAC    = np.empty(shape=[0,1])
    VD.YAC    = np.empty(shape=[0,1])
    VD.ZAC    = np.empty(shape=[0,1]) 
    VD.XBC    = np.empty(shape=[0,1])
    VD.YBC    = np.empty(shape=[0,1])
    VD.ZBC    = np.empty(shape=[0,1]) 
    VD.XC_TE  = np.empty(shape=[0,1])
    VD.YC_TE  = np.empty(shape=[0,1])
    VD.ZC_TE  = np.empty(shape=[0,1])     
    VD.XA_TE  = np.empty(shape=[0,1])
    VD.YA_TE  = np.empty(shape=[0,1])
    VD.ZA_TE  = np.empty(shape=[0,1]) 
    VD.XB_TE  = np.empty(shape=[0,1])
    VD.YB_TE  = np.empty(shape=[0,1])
    VD.ZB_TE  = np.empty(shape=[0,1])  
    VD.XC     = np.empty(shape=[0,1])
    VD.YC     = np.empty(shape=[0,1])
    VD.ZC     = np.empty(shape=[0,1])    
    VD.FUS_XC = np.empty(shape=[0,1])
    VD.FUS_YC = np.empty(shape=[0,1])
    VD.FUS_ZC = np.empty(shape=[0,1])      
    VD.CS     = np.empty(shape=[0,1]) 
    VD.X      = np.empty(shape=[0,1])
    VD.Y      = np.empty(shape=[0,1])
    VD.Z      = np.empty(shape=[0,1])
    VD.Y_SW   = np.empty(shape=[0,1])
    n_sw = settings.number_spanwise_vortices 
    n_cw = settings.number_chordwise_vortices     

    # ---------------------------------------------------------------------------------------
    # STEP 2: Unpack aircraft wing geometry 
    # ---------------------------------------------------------------------------------------    
    n_w        = 0  # instantiate the number of wings counter  
    n_cp       = 0  # instantiate number of bound vortices counter     
    wing_areas = [] # instantiate wing areas  
    
    for wing in geometry.wings:
        # get geometry of wing  
        span          = wing.spans.projected
        root_chord    = wing.chords.root
        tip_chord     = wing.chords.tip
        sweep_qc      = wing.sweeps.quarter_chord
        sweep_le      = wing.sweeps.leading_edge 
        twist_rc      = wing.twists.root
        twist_tc      = wing.twists.tip
        dihedral      = wing.dihedral
        sym_para      = wing.symmetric 
        vertical_wing = wing.vertical
        wing_origin   = wing.origin[0] 
        
        # determine if vehicle has symmetry 
        if sym_para is True :
            span = span/2
        
        # discretize wing using cosine spacing
        n               = np.linspace(n_sw+1,0,n_sw+1)         # vectorize
        thetan          = n*(np.pi/2)/(n_sw+1)                 # angular stations
        y_coordinates   = span*np.cos(thetan)                  # y locations based on the angular spacing
        
        # create empty vectors for coordinates 
        xah   = np.zeros(n_cw*n_sw)
        yah   = np.zeros(n_cw*n_sw)
        zah   = np.zeros(n_cw*n_sw)
        xbh   = np.zeros(n_cw*n_sw)
        ybh   = np.zeros(n_cw*n_sw)
        zbh   = np.zeros(n_cw*n_sw)    
        xch   = np.zeros(n_cw*n_sw)
        ych   = np.zeros(n_cw*n_sw)
        zch   = np.zeros(n_cw*n_sw)    
        xa1   = np.zeros(n_cw*n_sw)
        ya1   = np.zeros(n_cw*n_sw)
        za1   = np.zeros(n_cw*n_sw)
        xa2   = np.zeros(n_cw*n_sw)
        ya2   = np.zeros(n_cw*n_sw)
        za2   = np.zeros(n_cw*n_sw)    
        xb1   = np.zeros(n_cw*n_sw)
        yb1   = np.zeros(n_cw*n_sw)
        zb1   = np.zeros(n_cw*n_sw)
        xb2   = np.zeros(n_cw*n_sw) 
        yb2   = np.zeros(n_cw*n_sw) 
        zb2   = np.zeros(n_cw*n_sw)    
        xac   = np.zeros(n_cw*n_sw)
        yac   = np.zeros(n_cw*n_sw)
        zac   = np.zeros(n_cw*n_sw)    
        xbc   = np.zeros(n_cw*n_sw)
        ybc   = np.zeros(n_cw*n_sw)
        zbc   = np.zeros(n_cw*n_sw)    
        xa_te = np.zeros(n_cw*n_sw)
        ya_te = np.zeros(n_cw*n_sw)
        za_te = np.zeros(n_cw*n_sw)    
        xb_te = np.zeros(n_cw*n_sw)
        yb_te = np.zeros(n_cw*n_sw)
        zb_te = np.zeros(n_cw*n_sw)  
        xc    = np.zeros(n_cw*n_sw) 
        yc    = np.zeros(n_cw*n_sw) 
        zc    = np.zeros(n_cw*n_sw) 
        x     = np.zeros((n_cw+1)*(n_sw+1)) 
        y     = np.zeros((n_cw+1)*(n_sw+1)) 
        z     = np.zeros((n_cw+1)*(n_sw+1))         
        cs_w  = np.zeros(n_sw)

        # ---------------------------------------------------------------------------------------
        # STEP 3: Determine if wing segments are defined  
        # ---------------------------------------------------------------------------------------
        n_segments           = len(wing.Segments.keys())
        if n_segments>0:            
            # ---------------------------------------------------------------------------------------
            # STEP 4A: Discretizing the wing sections into panels
            # ---------------------------------------------------------------------------------------
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
            #          If applicable, append airfoil section VD and flap/aileron deflection angles.
            # --------------------------------------------------------------------------------------- 
            for i_seg in range(n_segments):   
                segment_chord[i_seg]    = wing.Segments[i_seg].root_chord_percent*root_chord
                segment_twist[i_seg]    = wing.Segments[i_seg].twist
                section_stations[i_seg] = wing.Segments[i_seg].percent_span_location*span  
                segment_dihedral[i_seg] = wing.Segments[i_seg].dihedral_outboard                    

                # change to leading edge sweep, if quarter chord sweep givent, convert to leading edge sweep 
                if (i_seg == n_segments-1):
                    segment_sweep[i_seg] = 0                                  
                else: 
                    if wing.Segments[i_seg].sweeps.leading_edge != None:
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
                    segment_area[i_seg]           = 0.5*(root_chord*wing.Segments[i_seg-1].root_chord_percent + root_chord*wing.Segments[i_seg].root_chord_percent)*segment_span[i_seg]

                # Get airfoil section VD  
                if wing.Segments[i_seg].Airfoil: 
                    airfoil_data = import_airfoil_geometry([wing.Segments[i_seg].Airfoil.airfoil.coordinate_file])    
                    segment_camber.append(airfoil_data.camber_coordinates[0])
                    segment_x_coord.append(airfoil_data.x_lower_surface[0]) 
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
            y_a   = y_coordinates[:-1] 
            y_b   = y_coordinates[1:]             
            del_y = y_coordinates[1:] - y_coordinates[:-1]           
            i_seg = 0           
            for idx_y in range(n_sw):
                # define coordinates of horseshoe vortices and control points
                idx_x = np.arange(n_cw) 
                eta_a = (y_a[idx_y] - section_stations[i_seg])  
                eta_b = (y_b[idx_y] - section_stations[i_seg]) 
                eta   = (y_b[idx_y] - del_y[idx_y]/2 - section_stations[i_seg]) 

                segment_chord_ratio = (segment_chord[i_seg+1] - segment_chord[i_seg])/segment_span[i_seg+1]
                segment_twist_ratio = (segment_twist[i_seg+1] - segment_twist[i_seg])/segment_span[i_seg+1]

                wing_chord_section_a  = segment_chord[i_seg] + (eta_a*segment_chord_ratio) 
                wing_chord_section_b  = segment_chord[i_seg] + (eta_b*segment_chord_ratio)
                wing_chord_section    = segment_chord[i_seg] + (eta*segment_chord_ratio)

                delta_x_a = wing_chord_section_a/n_cw  
                delta_x_b = wing_chord_section_b/n_cw      
                delta_x   = wing_chord_section/n_cw                                       

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

                # adjustment of coordinates for camber
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

                # adjustment of coordinates for twist  
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
                zeta_prime     = zeta_LE   - np.sin(section_twist)*(xi_c-xi_LE)      + np.cos(-section_twist)*(zeta-zeta_LE)            # z coordinate transformation of control point
                zeta_prime_ch  = zeta_LE   - np.sin(section_twist)*(xi_ch-xi_LE)     + np.cos(-section_twist)*(zeta_ch-zeta_LE)            # z coordinate transformation of center of horseshoe

                # ** TO DO ** Get flap/aileron locations and deflection
                # store coordinates of panels, horseshoeces vortices and control points relative to wing root 
                if vertical_wing:
                    xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                    za1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    ya1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                    xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                    za2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    ya2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2

                    xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                    zb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    yb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                        
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 

                    xah[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ah
                    zah[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    yah[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ah                    
                    xbh[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bh 
                    zbh[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    ybh[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bh

                    xch[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ch
                    zch[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)                   
                    ych[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ch

                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

                    xac[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ac 
                    zac[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    yac[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ac
                    xbc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bc
                    zbc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                            
                    ybc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bc
                    x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([xi_prime_a1,np.array([xi_prime_a2[-1]])])
                    z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*y_a[idx_y] 
                    y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([zeta_prime_a1,np.array([zeta_prime_a2[-1]])])

                else:     
                    xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                    ya1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    za1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                    xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                    ya2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    za2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2

                    xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                    yb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 

                    xah[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ah
                    yah[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    zah[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ah                    
                    xbh[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bh 
                    ybh[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zbh[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bh

                    xch[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ch
                    ych[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)                    
                    zch[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ch

                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

                    xac[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ac 
                    yac[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    zac[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ac
                    xbc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bc
                    ybc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                            
                    zbc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bc   
                    x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([xi_prime_a1,np.array([xi_prime_a2[-1]])])
                    y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*y_a[idx_y] 
                    z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([zeta_prime_a1,np.array([zeta_prime_a2[-1]])])                   

                idx += 1

                cs_w[idx_y] = wing_chord_section       

                if y_b[idx_y] == section_stations[i_seg+1]: 
                    i_seg += 1      

            if vertical_wing:    
                x[-(n_cw+1):] = np.concatenate([xi_prime_b1,np.array([xi_prime_b2[-1]])])
                z[-(n_cw+1):] = np.ones(n_cw+1)*y_b[idx_y] 
                y[-(n_cw+1):] = np.concatenate([zeta_prime_b1,np.array([zeta_prime_b2[-1]])])
            else:    
                x[-(n_cw+1):] = np.concatenate([xi_prime_b1,np.array([xi_prime_b2[-1]])])
                y[-(n_cw+1):] = np.ones(n_cw+1)*y_b[idx_y] 
                z[-(n_cw+1):] = np.concatenate([zeta_prime_b1,np.array([zeta_prime_b2[-1]])])                

        else:   # when no segments are defined on wing  
            # ---------------------------------------------------------------------------------------
            # STEP 6B: Define coordinates of panels horseshoe vortices and control points 
            # ---------------------------------------------------------------------------------------
            y_a   = y_coordinates[:-1] 
            y_b   = y_coordinates[1:] 
            
            if sweep_le != None:
                sweep = sweep_le
            else:                                                                
                cf    = 0.25                          
                sweep = np.arctan(((root_chord*cf) + (np.tan(sweep_qc)*span - cf*tip_chord)) /span)  
           
            wing_chord_ratio = (tip_chord-root_chord)/span
            wing_twist_ratio = (twist_tc-twist_rc)/span                    
            wing_areas.append(0.5*(root_chord+tip_chord)*span) 
            if sym_para is True :
                wing_areas.append(0.5*(root_chord+tip_chord)*span)   

            # Get airfoil section VD  
            if wing.Airfoil: 
                airfoil_data = import_airfoil_geometry(wing.Airfoil.airfoil.coordinate_file)    
                wing_camber  = airfoil_data.camber_coordinates
                wing_x_coord = airfoil_data.x_lower_surface
            else:
                wing_camber  = np.zeros(30) # dimension of Selig airfoil VD file
                wing_x_coord = np.linspace(0,1,30)

            delta_y = y_b - y_a
            for idx_y in range(n_sw):  
                idx_x = np.arange(n_cw) 
                eta_a = (y_a[idx_y])  
                eta_b = (y_b[idx_y]) 
                eta   = (y_b[idx_y] - delta_y[idx_y]/2) 
                
                # get spanwise discretization points
                wing_chord_section_a  = root_chord + (eta_a*wing_chord_ratio) 
                wing_chord_section_b  = root_chord + (eta_b*wing_chord_ratio)
                wing_chord_section    = root_chord + (eta*wing_chord_ratio)
                
                # get chordwise discretization points
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
                
                # adjustment of coordinates for camber
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

                # adjustment of coordinates for twist  
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

                # store coordinates of panels, horseshoe vortices and control points relative to wing root 
                if vertical_wing:
                    xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                    za1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    ya1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                    xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                    za2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    ya2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2

                    xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                    zb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    yb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                        
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 

                    xah[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ah
                    zah[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    yah[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ah                    
                    xbh[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bh 
                    zbh[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    ybh[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bh

                    xch[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ch
                    zch[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - delta_y[idx_y]/2)                   
                    ych[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ch

                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - delta_y[idx_y]/2) 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

                    xac[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ac 
                    zac[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    yac[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ac
                    xbc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bc
                    zbc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                            
                    ybc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bc        
                    x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([xi_prime_a1,np.array([xi_prime_a2[-1]])])
                    z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*y_a[idx_y] 
                    y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([zeta_prime_a1,np.array([zeta_prime_a2[-1]])])                    

                else: 
                    xa1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a1 
                    ya1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    za1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a1
                    xa2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_a2
                    ya2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    za2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_a2

                    xb1[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b1 
                    yb1[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zb1[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b1
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2 
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 

                    xah[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ah
                    yah[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    zah[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ah                    
                    xbh[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bh 
                    ybh[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zbh[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bh

                    xch[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ch
                    ych[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - delta_y[idx_y]/2)                   
                    zch[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ch

                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - delta_y[idx_y]/2) 
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

                    xac[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_ac 
                    yac[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_a[idx_y]
                    zac[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_ac
                    xbc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_bc
                    ybc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]                            
                    zbc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_bc       
                    x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([xi_prime_a1,np.array([xi_prime_a2[-1]])])
                    y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*y_a[idx_y] 
                    z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([zeta_prime_a1,np.array([zeta_prime_a2[-1]])])              

                cs_w[idx_y] = wing_chord_section

            if vertical_wing:    
                x[-(n_cw+1):] = np.concatenate([xi_prime_b1,np.array([xi_prime_b2[-1]])])
                z[-(n_cw+1):] = np.ones(n_cw+1)*y_b[idx_y] 
                y[-(n_cw+1):] = np.concatenate([zeta_prime_b1,np.array([zeta_prime_b2[-1]])])
            else:           
                x[-(n_cw+1):] = np.concatenate([xi_prime_b1,np.array([xi_prime_b2[-1]])])
                y[-(n_cw+1):] = np.ones(n_cw+1)*y_b[idx_y] 
                z[-(n_cw+1):] = np.concatenate([zeta_prime_b1,np.array([zeta_prime_b2[-1]])])   

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
        x   = x + wing_origin[0]  # x coordinate of control points on panel
        y   = y + wing_origin[1]  # y coordinate of control points on panel
        z   = z + wing_origin[2]  # y coordinate of control points on panel
 
        # find the location of the trailing edge panels of each wing
        locations = ((np.linspace(1,n_sw,n_sw, endpoint = True) * n_cw) - 1).astype(int)
        xc_te1 = np.repeat(np.atleast_2d(xc[locations]), n_cw , axis = 0)
        yc_te1 = np.repeat(np.atleast_2d(yc[locations]), n_cw , axis = 0)
        zc_te1 = np.repeat(np.atleast_2d(zc[locations]), n_cw , axis = 0)        
        xa_te1 = np.repeat(np.atleast_2d(xa2[locations]), n_cw , axis = 0)
        ya_te1 = np.repeat(np.atleast_2d(ya2[locations]), n_cw , axis = 0)
        za_te1 = np.repeat(np.atleast_2d(za2[locations]), n_cw , axis = 0)
        xb_te1 = np.repeat(np.atleast_2d(xb2[locations]), n_cw , axis = 0)
        yb_te1 = np.repeat(np.atleast_2d(yb2[locations]), n_cw , axis = 0)
        zb_te1 = np.repeat(np.atleast_2d(zb2[locations]), n_cw , axis = 0)     
        
        xc_te = np.hstack(xc_te1.T)
        yc_te = np.hstack(yc_te1.T)
        zc_te = np.hstack(zc_te1.T)        
        xa_te = np.hstack(xa_te1.T)
        ya_te = np.hstack(ya_te1.T)
        za_te = np.hstack(za_te1.T)
        xb_te = np.hstack(xb_te1.T)
        yb_te = np.hstack(yb_te1.T)
        zb_te = np.hstack(zb_te1.T) 
        
        # find spanwise locations 
        y_sw = yc[locations]        

        # if symmetry, store points of mirrored wing 
        n_w += 1  
        if sym_para is True :
            n_w += 1 
            # append wing spans          
            if vertical_wing:
                cs_w = np.concatenate([cs_w,cs_w])
                xah = np.concatenate([xah,xah])
                yah = np.concatenate([yah,yah])
                zah = np.concatenate([zah,-zah])
                xbh = np.concatenate([xbh,xbh])
                ybh = np.concatenate([ybh,ybh])
                zbh = np.concatenate([zbh,-zbh])
                xch = np.concatenate([xch,xch])
                ych = np.concatenate([ych,ych])
                zch = np.concatenate([zch,-zch])
    
                xa1 = np.concatenate([xa1,xa1])
                ya1 = np.concatenate([ya1,ya1])
                za1 = np.concatenate([za1,-za1])
                xa2 = np.concatenate([xa2,xa2])
                ya2 = np.concatenate([ya2,ya2])
                za2 = np.concatenate([za2,-za2])
    
                xb1 = np.concatenate([xb1,xb1])
                yb1 = np.concatenate([yb1,yb1])    
                zb1 = np.concatenate([zb1,-zb1])
                xb2 = np.concatenate([xb2,xb2])
                yb2 = np.concatenate([yb2,yb2])            
                zb2 = np.concatenate([zb2,-zb2])
    
                xac   = np.concatenate([xac ,xac ])
                yac   = np.concatenate([yac ,yac ])
                zac   = np.concatenate([zac ,-zac ])            
                xbc   = np.concatenate([xbc ,xbc ])
                ybc   = np.concatenate([ybc ,ybc ])
                zbc   = np.concatenate([zbc ,-zbc ]) 
                xc_te = np.concatenate([xc_te , xc_te ])
                yc_te = np.concatenate([yc_te , yc_te ])
                zc_te = np.concatenate([zc_te ,-zc_te ])                 
                xa_te = np.concatenate([xa_te , xa_te ])
                ya_te = np.concatenate([ya_te , ya_te ])
                za_te = np.concatenate([za_te ,-za_te ])            
                xb_te = np.concatenate([xb_te , xb_te ])
                yb_te = np.concatenate([yb_te , yb_te ])
                zb_te = np.concatenate([zb_te ,-zb_te ])
                
                y_sw  = np.concatenate([y_sw,-y_sw ])
                xc    = np.concatenate([xc ,xc ])
                yc    = np.concatenate([yc ,yc]) 
                zc    = np.concatenate([zc ,-zc ])
                x     = np.concatenate([x , x ])
                y     = np.concatenate([y ,y])
                z     = np.concatenate([z ,-z ])                  
                
            else:
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
    
                xac   = np.concatenate([xac ,xac ])
                yac   = np.concatenate([yac ,-yac ])
                zac   = np.concatenate([zac ,zac ])            
                xbc   = np.concatenate([xbc ,xbc ])
                ybc   = np.concatenate([ybc ,-ybc ])
                zbc   = np.concatenate([zbc ,zbc ]) 
                xc_te = np.concatenate([xc_te , xc_te ])
                yc_te = np.concatenate([yc_te ,-yc_te ])
                zc_te = np.concatenate([zc_te , zc_te ])                   
                xa_te = np.concatenate([xa_te , xa_te ])
                ya_te = np.concatenate([ya_te ,-ya_te ])
                za_te = np.concatenate([za_te , za_te ])            
                xb_te = np.concatenate([xb_te , xb_te ])
                yb_te = np.concatenate([yb_te ,-yb_te ])
                zb_te = np.concatenate([zb_te , zb_te ]) 
                
                y_sw  = np.concatenate([y_sw,-y_sw ])
                xc    = np.concatenate([xc ,xc ])
                yc    = np.concatenate([yc ,-yc]) 
                zc    = np.concatenate([zc ,zc ])
                x     = np.concatenate([x , x ])
                y     = np.concatenate([y ,-y])
                z     = np.concatenate([z , z ])            

        n_cp += len(xch)        

        # ---------------------------------------------------------------------------------------
        # STEP 7: Store wing in vehicle vector
        # ---------------------------------------------------------------------------------------       
        VD.XAH    = np.append(VD.XAH,xah)
        VD.YAH    = np.append(VD.YAH,yah)
        VD.ZAH    = np.append(VD.ZAH,zah)
        VD.XBH    = np.append(VD.XBH,xbh)
        VD.YBH    = np.append(VD.YBH,ybh)
        VD.ZBH    = np.append(VD.ZBH,zbh)
        VD.XCH    = np.append(VD.XCH,xch)
        VD.YCH    = np.append(VD.YCH,ych)
        VD.ZCH    = np.append(VD.ZCH,zch)            
        VD.XA1    = np.append(VD.XA1,xa1)
        VD.YA1    = np.append(VD.YA1,ya1)
        VD.ZA1    = np.append(VD.ZA1,za1)
        VD.XA2    = np.append(VD.XA2,xa2)
        VD.YA2    = np.append(VD.YA2,ya2)
        VD.ZA2    = np.append(VD.ZA2,za2)        
        VD.XB1    = np.append(VD.XB1,xb1)
        VD.YB1    = np.append(VD.YB1,yb1)
        VD.ZB1    = np.append(VD.ZB1,zb1)
        VD.XB2    = np.append(VD.XB2,xb2)                
        VD.YB2    = np.append(VD.YB2,yb2)        
        VD.ZB2    = np.append(VD.ZB2,zb2)    
        VD.XC_TE  = np.append(VD.XC_TE,xc_te)
        VD.YC_TE  = np.append(VD.YC_TE,yc_te) 
        VD.ZC_TE  = np.append(VD.ZC_TE,zc_te)          
        VD.XA_TE  = np.append(VD.XA_TE,xa_te)
        VD.YA_TE  = np.append(VD.YA_TE,ya_te) 
        VD.ZA_TE  = np.append(VD.ZA_TE,za_te) 
        VD.XB_TE  = np.append(VD.XB_TE,xb_te)
        VD.YB_TE  = np.append(VD.YB_TE,yb_te) 
        VD.ZB_TE  = np.append(VD.ZB_TE,zb_te)  
        VD.XAC    = np.append(VD.XAC,xac)
        VD.YAC    = np.append(VD.YAC,yac) 
        VD.ZAC    = np.append(VD.ZAC,zac) 
        VD.XBC    = np.append(VD.XBC,xbc)
        VD.YBC    = np.append(VD.YBC,ybc) 
        VD.ZBC    = np.append(VD.ZBC,zbc)  
        VD.XC     = np.append(VD.XC ,xc)
        VD.YC     = np.append(VD.YC ,yc)
        VD.ZC     = np.append(VD.ZC ,zc)  
        VD.X      = np.append(VD.X ,x)
        VD.Y_SW   = np.append(VD.Y_SW ,y_sw)
        VD.Y      = np.append(VD.Y ,y)
        VD.Z      = np.append(VD.Z ,z)         
        VD.CS     = np.append(VD.CS,cs_w)        

    # ---------------------------------------------------------------------------------------
    # STEP 8.1: Unpack aircraft fuselage geometry NOTE THAT FUSELAGE GOMETRY IS OMITTED FROM VLM
    # --------------------------------------------------------------------------------------- 
    VD.n_fus = 0
    for fus in geometry.fuselages:   
        VD = generate_fuselage_vortex_distribution(VD,fus,n_cw,n_sw) 
        VD = generate_fuselage_surface_points(VD,fus)     
         
    VD.n_w        = n_w
    VD.n_sw       = n_sw
    VD.n_cw       = n_cw    
    VD.n_cp       = n_cp  
    VD.wing_areas = np.array(wing_areas)   
    VD.Stot       = sum(wing_areas)

    geometry.vortex_distribution = VD

    # Compute Panel Areas 
    VD.panel_areas = compute_panel_area(VD)      

    return VD 

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def generate_fuselage_vortex_distribution(VD,fus,n_cw,n_sw):
    """ This generates the vortex distribution points on the fuselage 

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """    
    
    fhs_xc    = np.zeros(n_cw*n_sw)
    fhs_yc    = np.zeros(n_cw*n_sw)
    fhs_zc    = np.zeros(n_cw*n_sw) 
    fhs_x     = np.zeros((n_cw+1)*(n_sw+1))
    fhs_y     = np.zeros((n_cw+1)*(n_sw+1))
    fhs_z     = np.zeros((n_cw+1)*(n_sw+1))          
    fus_h_cs  = np.zeros(n_sw)     

    fvs_xc    = np.zeros(n_cw*n_sw)
    fvs_zc    = np.zeros(n_cw*n_sw)
    fvs_yc    = np.zeros(n_cw*n_sw)   
    fvs_x     = np.zeros((n_cw+1)*(n_sw+1))
    fvs_y     = np.zeros((n_cw+1)*(n_sw+1))
    fvs_z     = np.zeros((n_cw+1)*(n_sw+1))   
    fus_v_cs  = np.zeros(n_sw)     

    semispan_h = fus.width * 0.5  
    semispan_v = fus.heights.maximum * 0.5
    origin     = fus.origin[0]

    # Compute the curvature of the nose/tail given fineness ratio. Curvature is derived from general quadratic equation
    # This method relates the fineness ratio to the quadratic curve formula via a spline fit interpolation
    vec1               = [2 , 1.5, 1.2 , 1]
    vec2               = [1  ,1.57 , 3.2,  8]
    x                  = np.linspace(0,1,4)
    fus_nose_curvature =  np.interp(np.interp(fus.fineness.nose,vec2,x), x , vec1)
    fus_tail_curvature =  np.interp(np.interp(fus.fineness.tail,vec2,x), x , vec1) 

    # Horizontal Sections of fuselage
    fhs        = Data()        
    fhs.origin = np.zeros((n_sw+1,3))        
    fhs.chord  = np.zeros((n_sw+1))         
    fhs.sweep  = np.zeros((n_sw+1))     
                 
    fvs        = Data() 
    fvs.origin = np.zeros((n_sw+1,3))
    fvs.chord  = np.zeros((n_sw+1)) 
    fvs.sweep  = np.zeros((n_sw+1)) 

    si         = np.arange(1,((n_sw*2)+2))
    spacing    = np.cos((2*si - 1)/(2*len(si))*np.pi)     
    h_array    = semispan_h*spacing[0:int((len(si)+1)/2)][::-1]  
    v_array    = semispan_v*spacing[0:int((len(si)+1)/2)][::-1]  

    for i in range(n_sw+1): 
        fhs_cabin_length  = fus.lengths.total - (fus.lengths.nose + fus.lengths.tail)
        fhs.nose_length   = ((1 - ((abs(h_array[i]/semispan_h))**fus_nose_curvature ))**(1/fus_nose_curvature))*fus.lengths.nose
        fhs.tail_length   = ((1 - ((abs(h_array[i]/semispan_h))**fus_tail_curvature ))**(1/fus_tail_curvature))*fus.lengths.tail
        fhs.nose_origin   = fus.lengths.nose - fhs.nose_length 
        fhs.origin[i][:]  = np.array([origin[0] + fhs.nose_origin , origin[1] + h_array[i], origin[2]])
        fhs.chord[i]      = fhs_cabin_length + fhs.nose_length + fhs.tail_length          

        fvs_cabin_length  = fus.lengths.total - (fus.lengths.nose + fus.lengths.tail)
        fvs.nose_length   = ((1 - ((abs(v_array[i]/semispan_v))**fus_nose_curvature ))**(1/fus_nose_curvature))*fus.lengths.nose
        fvs.tail_length   = ((1 - ((abs(v_array[i]/semispan_v))**fus_tail_curvature ))**(1/fus_tail_curvature))*fus.lengths.tail
        fvs.nose_origin   = fus.lengths.nose - fvs.nose_length 
        fvs.origin[i][:]  = np.array([origin[0] + fvs.nose_origin , origin[1] , origin[2]+  v_array[i]])
        fvs.chord[i]      = fvs_cabin_length + fvs.nose_length + fvs.tail_length

    fhs.sweep[:] = np.concatenate([np.arctan((fhs.origin[:,0][1:] - fhs.origin[:,0][:-1])/(fhs.origin[:,1][1:]  - fhs.origin[:,1][:-1])) ,np.zeros(1)])
    fvs.sweep[:] = np.concatenate([np.arctan((fvs.origin[:,0][1:] - fvs.origin[:,0][:-1])/(fvs.origin[:,2][1:]  - fvs.origin[:,2][:-1])) ,np.zeros(1)])

    # ---------------------------------------------------------------------------------------
    # STEP 9: Define coordinates of panels horseshoe vortices and control points  
    # ---------------------------------------------------------------------------------------        
    fhs_eta_a = h_array[:-1] 
    fhs_eta_b = h_array[1:]            
    fhs_del_y = h_array[1:] - h_array[:-1]
    fhs_eta   = h_array[1:] - fhs_del_y/2

    fvs_eta_a = v_array[:-1] 
    fvs_eta_b = v_array[1:]                  
    fvs_del_y = v_array[1:] - v_array[:-1]
    fvs_eta   = v_array[1:] - fvs_del_y/2 

    fhs_cs = np.concatenate([fhs.chord,fhs.chord])
    fvs_cs = np.concatenate([fvs.chord,fvs.chord])

    # define coordinates of horseshoe vortices and control points       
    for idx_y in range(n_sw):  
        idx_x = np.arange(n_cw)

        # fuselage horizontal section 
        delta_x_a = fhs.chord[idx_y]/n_cw      
        delta_x_b = fhs.chord[idx_y + 1]/n_cw    
        delta_x   = (fhs.chord[idx_y]+fhs.chord[idx_y + 1])/(2*n_cw)

        fhs_xi_a1 = fhs.origin[idx_y][0] + delta_x_a*idx_x                    # x coordinate of top left corner of panel
        fhs_xi_ah = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.25   # x coordinate of left corner of panel
        fhs_xi_a2 = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a        # x coordinate of bottom left corner of bound vortex 
        fhs_xi_ac = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.75   # x coordinate of bottom left corner of control point vortex  
        fhs_xi_b1 = fhs.origin[idx_y+1][0] + delta_x_b*idx_x                  # x coordinate of top right corner of panel      
        fhs_xi_bh = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
        fhs_xi_b2 = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel
        fhs_xi_bc = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
        fhs_xi_c  = (fhs.origin[idx_y][0] + fhs.origin[idx_y+1][0])/2  + delta_x*idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
        fhs_xi_ch = (fhs.origin[idx_y][0] + fhs.origin[idx_y+1][0])/2  + delta_x*idx_x + delta_x*0.25   # x coordinate center of bound vortex of each panel 

        fhs_xc [idx_y*n_cw:(idx_y+1)*n_cw] = fhs_xi_c                        + fus.origin[0][0]  
        fhs_yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fhs_eta[idx_y]    + fus.origin[0][1]  
        fhs_zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                  + fus.origin[0][2]               
        fhs_x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([fhs_xi_a1,np.array([fhs_xi_a2[-1]])])+ fus.origin[0][0]  
        fhs_y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*fhs_eta_a[idx_y]  + fus.origin[0][1]                             
        fhs_z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.zeros(n_cw+1)                  + fus.origin[0][2]


        # fuselage vertical section                      
        delta_x_a = fvs.chord[idx_y]/n_cw      
        delta_x_b = fvs.chord[idx_y + 1]/n_cw    
        delta_x   = (fvs.chord[idx_y]+fvs.chord[idx_y + 1])/(2*n_cw)   

        fvs_xi_a1 = fvs.origin[idx_y][0] + delta_x_a*idx_x                    # z coordinate of top left corner of panel
        fvs_xi_ah = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.25   # z coordinate of left corner of panel
        fvs_xi_a2 = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a        # z coordinate of bottom left corner of bound vortex 
        fvs_xi_ac = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.75   # z coordinate of bottom left corner of control point vortex  
        fvs_xi_b1 = fvs.origin[idx_y+1][0] + delta_x_b*idx_x                    # z coordinate of top right corner of panel      
        fvs_xi_bh = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.25   # z coordinate of right corner of bound vortex         
        fvs_xi_b2 = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b        # z coordinate of bottom right corner of panel
        fvs_xi_bc = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.75   # z coordinate of bottom right corner of control point vortex         
        fvs_xi_c  = (fvs.origin[idx_y][0] + fvs.origin[idx_y+1][0])/2 + delta_x *idx_x + delta_x*0.75     # z coordinate three-quarter chord control point for each panel
        fvs_xi_ch = (fvs.origin[idx_y][0] + fvs.origin[idx_y+1][0])/2 + delta_x *idx_x + delta_x*0.25     # z coordinate center of bound vortex of each panel 

        fvs_xc [idx_y*n_cw:(idx_y+1)*n_cw] = fvs_xi_c                       + fus.origin[0][0]  
        fvs_zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*fvs_eta[idx_y]   + fus.origin[0][2]  
        fvs_yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.zeros(n_cw)                 + fus.origin[0][1]  
        fvs_x[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.concatenate([fvs_xi_a1,np.array([fvs_xi_a2[-1]])]) + fus.origin[0][0]  
        fvs_z[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.ones(n_cw+1)*fvs_eta_a[idx_y] + fus.origin[0][2]               
        fvs_y[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)] = np.zeros(n_cw+1)                 + fus.origin[0][1]               

    fhs_x[-(n_cw+1):] = np.concatenate([fhs_xi_b1,np.array([fhs_xi_b2[-1]])])+ fus.origin[0][0]  
    fhs_y[-(n_cw+1):] = np.ones(n_cw+1)*fhs_eta_b[idx_y]  + fus.origin[0][1]                             
    fhs_z[-(n_cw+1):] = np.zeros(n_cw+1)                  + fus.origin[0][2]        
    fvs_x[-(n_cw+1):] = np.concatenate([fvs_xi_a1,np.array([fvs_xi_a2[-1]])]) + fus.origin[0][0]  
    fvs_z[-(n_cw+1):] = np.ones(n_cw+1)*fvs_eta_a[idx_y] + fus.origin[0][2]               
    fvs_y[-(n_cw+1):] = np.zeros(n_cw+1)                 + fus.origin[0][1]   
    fhs_cs =  (fhs.chord[:-1]+fhs.chord[1:])/2
    fvs_cs =  (fvs.chord[:-1]+fvs.chord[1:])/2     

    # Append Horizontal Fuselage Sections  
    fhs_xc    = np.concatenate([fhs_xc[::-1] , fhs_xc ])
    fhs_yc    = np.concatenate([fhs_yc[::-1] ,-fhs_yc])
    fhs_zc    = np.concatenate([fhs_zc[::-1] , fhs_zc ])     
    fhs_x     = np.concatenate([fhs_x  , fhs_x  ])
    fhs_y     = np.concatenate([fhs_y  ,-fhs_y ])
    fhs_z     = np.concatenate([fhs_z  , fhs_z  ])    
    VD.FUS_XC = np.append(VD.FUS_XC ,fhs_xc)
    VD.FUS_YC = np.append(VD.FUS_YC ,fhs_yc)
    VD.FUS_ZC = np.append(VD.FUS_ZC ,fhs_zc)   

    # Append Vertical Fuselage Sections  
    fvs_xc    = np.concatenate([fvs_xc[::-1], fvs_xc ])
    fvs_yc    = np.concatenate([fvs_yc[::-1], fvs_yc ])
    fvs_zc    = np.concatenate([fvs_zc[::-1],-fvs_zc ])
    fvs_x     = np.concatenate([fhs_x  , fhs_x  ])
    fvs_y     = np.concatenate([fhs_y  , fhs_y ])
    fvs_z     = np.concatenate([fhs_z  , -fhs_z ])   
    
    # increment fuslage lifting surface sections  
    VD.n_fus  += 4    
    
    # Currently, fuselage is only used for plotting not analysis 
    VD.FUS_XC = np.append(VD.FUS_XC ,fvs_xc)
    VD.FUS_YC = np.append(VD.FUS_YC ,fvs_yc)
    VD.FUS_ZC = np.append(VD.FUS_ZC ,fvs_zc) 
    VD.X      = np.append(VD.X  ,fvs_x )
    VD.Y      = np.append(VD.Y  ,fvs_y )
    VD.Z      = np.append(VD.Z  ,fvs_z )     
    
    return VD

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def generate_fuselage_surface_points(VD,fus):
    """ This generates the coordinate points on the surface of the fuselage

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """      
    num_fus_segs = len(fus.Segments.keys())
    tessellation = 24
    if num_fus_segs > 0:  
        fus_pts = np.zeros((num_fus_segs,tessellation ,3))
        for i_seg in range(num_fus_segs):
            theta    = np.linspace(0,2*np.pi,tessellation +1)[:-1] 
            a        = fus.Segments[i_seg].width/2            
            b        = fus.Segments[i_seg].height/2 
            r        = np.sqrt((b*np.sin(theta))**2  + (a*np.cos(theta))**2)  
            fus_ypts = r*np.cos(theta)
            fus_zpts = r*np.sin(theta) 
            fus_pts[i_seg,:,0] = fus.Segments[i_seg].origin[0]
            fus_pts[i_seg,:,1] = fus_ypts + fus.Segments[i_seg].origin[1]
            fus_pts[i_seg,:,2] = fus_zpts + fus.Segments[i_seg].origin[2]
        
        # store points
        VD.FUS_SURF_PTS = fus_pts
    else:
        VD.FUS_SURF_PTS = None # future work
        
    return VD

## @ingroup Methods-Aerodynamics-Common-Fidelity_Zero-Lift
def compute_panel_area(VD):
    """ This computed the area of the panels on the lifting surface of the vehicle 

    Assumptions: 
    None

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution    
    
    Properties Used:
    N/A
    """     
    
    # create vectors for panel corders
    P1P2 = np.array([VD.XB1 - VD.XA1,VD.YB1 - VD.YA1,VD.ZB1 - VD.ZA1]).T
    P1P3 = np.array([VD.XA2 - VD.XA1,VD.YA2 - VD.YA1,VD.ZA2 - VD.ZA1]).T
    P2P3 = np.array([VD.XA2 - VD.XB1,VD.YA2 - VD.YB1,VD.ZA2 - VD.ZB1]).T
    P2P4 = np.array([VD.XB2 - VD.XB1,VD.YB2 - VD.YB1,VD.ZB2 - VD.ZB1]).T   
    
    # compute area of quadrilateral panel
    A_panel = 0.5*(np.linalg.norm(np.cross(P1P2,P1P3)) + np.linalg.norm(np.cross(P2P3, P2P4)))
    return A_panel