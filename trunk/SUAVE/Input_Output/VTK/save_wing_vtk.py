## @ingroup Time_Accurate-Simulations
# save_wing_vtk.py
# 
# Created:    Jun 2021, R. Erhard
# Modified: 
#           
import SUAVE
import numpy as np
#from SUAVE.Plots.Geometry_Plots.plot_vehicle import generate_wing_points # feature-time_accurate
from SUAVE.Core import Data

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_wing_vortex_distribution import generate_wing_vortex_distribution


from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_naca_4series import compute_naca_4series  


def save_wing_vtk(vehicle, wing_instance, settings, filename, Results):
    "Saves a SUAVE wing object as a VTK in legacy format."
    
    # generate VD for this wing alone
    wing_vehicle = SUAVE.Vehicle() 
    wing_vehicle.append_component(wing_instance)
    
    #VD = generate_wing_points(wing_vehicle,settings) 
    VD = generate_wing_vortex_distribution(wing_vehicle,settings)
    
    #VD   = vehicle.vortex_distribution 
    n_cw = VD.n_cw[0]
    n_sw = VD.n_sw[0]
    n_cp = VD.n_cp # number of control points and panels on wing
    
    ## check for wing segments:
    #n_w        = len(vehicle.wings)
    #main_wing  = vehicle.wings.main_wing
    #n_segments = len(main_wing.Segments.keys())
    #n_sw       = n_sw*(n_segments+1)
    
    
    symmetric = vehicle.wings[wing_instance.tag].symmetric
    
    if symmetric:
        half_l = int(len(VD.XA1)/2)
        
        # number panels per half span
        n_cp   = int(n_cp/2)
        n_cw   = n_cw
        n_sw   = n_sw
        
        # split wing into two separate wings
        Rwing = Data()
        Lwing = Data()
        
        Rwing.XA1 = VD.XA1[0:half_l]
        Rwing.XA2 = VD.XA1[0:half_l]
        Rwing.XB1 = VD.XB1[0:half_l]
        Rwing.XB2 = VD.XB1[0:half_l]
        Rwing.YA1 = VD.YA1[0:half_l]
        Rwing.YA2 = VD.YA1[0:half_l]
        Rwing.YB1 = VD.YB1[0:half_l]
        Rwing.YB2 = VD.YB1[0:half_l]
        Rwing.ZA1 = VD.ZA1[0:half_l]
        Rwing.ZA2 = VD.ZA1[0:half_l]
        Rwing.ZB1 = VD.ZB1[0:half_l]
        Rwing.ZB2 = VD.ZB1[0:half_l]        
        
        Lwing.XA1 = VD.XA1[half_l:]
        Lwing.XA2 = VD.XA1[half_l:]
        Lwing.XB1 = VD.XB1[half_l:]
        Lwing.XB2 = VD.XB1[half_l:]  
        Lwing.YA1 = VD.YA1[half_l:]
        Lwing.YA2 = VD.YA1[half_l:]
        Lwing.YB1 = VD.YB1[half_l:]
        Lwing.YB2 = VD.YB1[half_l:]   
        Lwing.ZA1 = VD.ZA1[half_l:]
        Lwing.ZA2 = VD.ZA1[half_l:]
        Lwing.ZB1 = VD.ZB1[half_l:]
        Lwing.ZB2 = VD.ZB1[half_l:]       
        
        sep  = filename.find('.')
        
        Lfile = filename[0:sep]+"_L"+filename[sep:]
        Rfile = filename[0:sep]+"_R"+filename[sep:]
        
        # write vtks for each half wing
        write_wing_vtk(Lwing,n_cw,n_sw,n_cp,Results,Lfile)
        write_wing_vtk(Rwing,n_cw,n_sw,n_cp,Results,Rfile)
        
        
    else:
        wing = VD
        write_wing_vtk(wing,n_cw,n_sw,n_cp,Results,filename)

    return


def write_wing_vtk(wing,n_cw,n_sw,n_cp,Results,filename):
    # Create file
    with open(filename, 'w') as f:
    
        #---------------------
        # Write header
        #---------------------
        l1 = "# vtk DataFile Version 4.0"     # File version and identifier
        l2 = "\nSUAVE Model of PROWIM Wing "  # Title 
        l3 = "\nASCII"                        # Data type
        l4 = "\nDATASET UNSTRUCTURED_GRID"    # Dataset structure / topology

        header = [l1, l2, l3, l4]
        f.writelines(header) 
        
        #---------------------    
        # Write Points:
        #---------------------    
        n_indices = (n_cw+1)*(n_sw+1)    # total number of cell vertices
        points_header = "\n\nPOINTS "+str(n_indices) +" float"
        f.write(points_header)
        
        cw_laps =0
        for i in range(n_indices):
            
            if i == n_indices-1:
                # Last index; use B2 to get rightmost TE node
                xp = round(wing.XB2[i-cw_laps-n_cw-1],4)
                yp = round(wing.YB2[i-cw_laps-n_cw-1],4)
                zp = round(wing.ZB2[i-cw_laps-n_cw-1],4)
            elif i > (n_indices-1-(n_cw+1)):
                # Last spanwise set; use B1 to get rightmost node indices
                xp = round(wing.XB1[i-cw_laps-n_cw],4)
                yp = round(wing.YB1[i-cw_laps-n_cw],4)
                zp = round(wing.ZB1[i-cw_laps-n_cw],4)
            elif i==0:
                # first point
                xp = round(wing.XA1[i],4)
                yp = round(wing.YA1[i],4)
                zp = round(wing.ZA1[i],4)            
                
            elif i==cw_laps + n_cw*(cw_laps+1): #i%n_cw==0:
                # Last chordwise station for this spanwise location; use A2 to get left TE node
                cw_laps = cw_laps +1
                xp = round(wing.XA2[i-cw_laps],4)
                yp = round(wing.YA2[i-cw_laps],4)
                zp = round(wing.ZA2[i-cw_laps],4)  
                
            else:
                # print the point index (Left LE --> Left TE --> Right LE --> Right TE)
                xp = round(wing.XA1[i-cw_laps],4)
                yp = round(wing.YA1[i-cw_laps],4)
                zp = round(wing.ZA1[i-cw_laps],4)
            
            new_point = "\n"+str(xp)+" "+str(yp)+" "+str(zp)
            f.write(new_point)
    
        #---------------------    
        # Write Cells:
        #---------------------
        n            = n_cp # total number of cells
        v_per_cell   = 4 # quad cells
        size         = n*(1+v_per_cell) # total number of integer values required to represent the list
        cell_header  = "\n\nCELLS "+str(n)+" "+str(size)
        f.write(cell_header)
        
        
        for i in range(n_cp):
            if i==0:
                node = i
            elif i%n_cw ==0:
                node = node+1
            new_cell = "\n4 "+str(node)+" "+str(node+1)+" "+str(node+n_cw+2)+" "+str(node+n_cw+1)
            f.write(new_cell)
            
            # update node:
            node = node+1
    
        #---------------------        
        # Write Cell Types:
        #---------------------
        cell_type_header  = "\n\nCELL_TYPES "+str(n)
        f.write(cell_type_header)        
        for i in range(n_cp):
            f.write("\n9")
            
        #--------------------------        
        # Write Scalar Cell Data:
        #--------------------------
        cell_data_header  = "\n\nCELL_DATA "+str(n)
        f.write(cell_data_header)    
        
        # First scalar value
        f.write("\nSCALARS cl float 1")
        f.write("\nLOOKUP_TABLE default")   
        cl = Results['cl_y_DVE'][0]
        for i in range(n_cp):
            new_cl = str(cl[int(i/n_cw)])
            f.write("\n"+new_cl)
            
        f.write("\nSCALARS Cl/CL float 1")
        f.write("\nLOOKUP_TABLE default")   
        cl = Results['cl_y_DVE'][0]
        CL = Results['CL_wing_DVE'][0][0]
        for i in range(n_cp):
            new_cl_CL = str(cl[int(i/n_cw)]/CL)
            f.write("\n"+new_cl_CL)
            
        f.write("\nSCALARS cd float 1")
        f.write("\nLOOKUP_TABLE default")   
        cd = Results['cdi_y_DVE'][0]
        for i in range(n_cp):
            new_cd = str(cd[int(i/n_cw)])
            f.write("\n"+new_cd)  
            
        f.write("\nSCALARS cd_CD float 1")
        f.write("\nLOOKUP_TABLE default")   
        cd = Results['cdi_y_DVE'][0]
        CD = Results['CDi_wing_DVE'][0][0]
        for i in range(n_cp):
            new_cd_CD = str(cd[int(i/n_cw)]/CD)
            f.write("\n"+new_cd_CD)               
    
    f.close()
    return


def generate_wing_points(vehicle,settings):
    ''' 
        _ts  = true surface 
    '''
    # ---------------------------------------------------------------------------------------
    # STEP 1: Define empty vectors for coordinates of panes, control points and bound vortices
    # ---------------------------------------------------------------------------------------
    VD = Data()
 
    VD.XA1         = np.empty(shape=[0,1])
    VD.YA1         = np.empty(shape=[0,1])  
    VD.ZA1         = np.empty(shape=[0,1])
    VD.XA2         = np.empty(shape=[0,1])
    VD.YA2         = np.empty(shape=[0,1])    
    VD.ZA2         = np.empty(shape=[0,1])    
    VD.XB1         = np.empty(shape=[0,1])
    VD.YB1         = np.empty(shape=[0,1])  
    VD.ZB1         = np.empty(shape=[0,1])
    VD.XB2         = np.empty(shape=[0,1])
    VD.YB2         = np.empty(shape=[0,1])    
    VD.ZB2         = np.empty(shape=[0,1])  
    VD.XA1_ts      = np.empty(shape=[0,1])
    VD.YA1_ts      = np.empty(shape=[0,1])  
    VD.ZA1_ts      = np.empty(shape=[0,1])
    VD.XA2_ts      = np.empty(shape=[0,1])
    VD.YA2_ts      = np.empty(shape=[0,1])    
    VD.ZA2_ts      = np.empty(shape=[0,1])    
    VD.XB1_ts      = np.empty(shape=[0,1])
    VD.YB1_ts      = np.empty(shape=[0,1])  
    VD.ZB1_ts      = np.empty(shape=[0,1])
    VD.XB2_ts      = np.empty(shape=[0,1])
    VD.YB2_ts      = np.empty(shape=[0,1])    
    VD.ZB2_ts      = np.empty(shape=[0,1])  
    VD.XC          = np.empty(shape=[0,1])
    VD.YC          = np.empty(shape=[0,1])
    VD.ZC          = np.empty(shape=[0,1])     
    n_sw           = settings.number_spanwise_vortices 
    n_cw           = settings.number_chordwise_vortices     
    spc            = settings.spanwise_cosine_spacing 

    # ---------------------------------------------------------------------------------------
    # STEP 2: Unpack aircraft wings  
    # ---------------------------------------------------------------------------------------    
    n_w         = 0  # instantiate the number of wings counter  
    n_cp        = 0  # instantiate number of bound vortices counter     
    n_sp        = 0  # instantiate number of true surface panels    
    wing_areas  = [] # instantiate wing areas
    vortex_lift = []
    
    for wing in vehicle.wings: 
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
        vortex_lift.append(wing.vortex_lift)
        
        # determine if vehicle has symmetry 
        if sym_para is True :
            span = span/2
            vortex_lift.append(wing.vortex_lift)
        
        if spc == True:
            
            # discretize wing using cosine spacing
            n               = np.linspace(n_sw+1,0,n_sw+1)         # vectorize
            thetan          = n*(np.pi/2)/(n_sw+1)                 # angular stations
            y_coordinates   = span*np.cos(thetan)                  # y locations based on the angular spacing
        else:
        
            # discretize wing using linear spacing
            y_coordinates   = np.linspace(0,span,n_sw+1) 
        
        # create empty vectors for coordinates 
        xa1        = np.zeros(n_cw*n_sw)
        ya1        = np.zeros(n_cw*n_sw)
        za1        = np.zeros(n_cw*n_sw)
        xa2        = np.zeros(n_cw*n_sw)
        ya2        = np.zeros(n_cw*n_sw)
        za2        = np.zeros(n_cw*n_sw)    
        xb1        = np.zeros(n_cw*n_sw)
        yb1        = np.zeros(n_cw*n_sw)
        zb1        = np.zeros(n_cw*n_sw)
        xb2        = np.zeros(n_cw*n_sw) 
        yb2        = np.zeros(n_cw*n_sw) 
        zb2        = np.zeros(n_cw*n_sw)  
        xa1_ts     = np.zeros(2*n_cw*n_sw)
        ya1_ts     = np.zeros(2*n_cw*n_sw)
        za1_ts     = np.zeros(2*n_cw*n_sw)
        xa2_ts     = np.zeros(2*n_cw*n_sw)
        ya2_ts     = np.zeros(2*n_cw*n_sw)
        za2_ts     = np.zeros(2*n_cw*n_sw)    
        xb1_ts     = np.zeros(2*n_cw*n_sw)
        yb1_ts     = np.zeros(2*n_cw*n_sw)
        zb1_ts     = np.zeros(2*n_cw*n_sw)
        xb2_ts     = np.zeros(2*n_cw*n_sw) 
        yb2_ts     = np.zeros(2*n_cw*n_sw) 
        zb2_ts     = np.zeros(2*n_cw*n_sw)           
        xc         = np.zeros(n_cw*n_sw) 
        yc         = np.zeros(n_cw*n_sw) 
        zc         = np.zeros(n_cw*n_sw)         
        cs_w       = np.zeros(n_sw)

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
            segment_top_surface    = []
            segment_bot_surface    = []
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
                    segment_top_surface.append(airfoil_data.y_upper_surface[0])
                    segment_bot_surface.append(airfoil_data.y_lower_surface[0])
                    segment_x_coord.append(airfoil_data.x_lower_surface[0]) 
                else:
                    dummy_dimension  = 30 
                    segment_camber.append(np.zeros(dummy_dimension))   
                    airfoil_data = compute_naca_4series(0.0, 0.0,wing.thickness_to_chord,dummy_dimension*2 - 2)
                    segment_top_surface.append(airfoil_data.y_upper_surface[0])
                    segment_bot_surface.append(airfoil_data.y_lower_surface[0])                    
                    segment_x_coord.append(np.linspace(0,1,30))  

            wing_areas.append(np.sum(segment_area[:]))
            if sym_para is True :
                wing_areas.append(np.sum(segment_area[:]))            

            # Shift spanwise vortices onto section breaks  
            if len(y_coordinates) < n_segments:
                raise ValueError('Not enough spanwise VLM stations for segment breaks')

            last_idx = None            
            for i_seg in range(n_segments):
                idx =  (np.abs(y_coordinates-section_stations[i_seg])).argmin()
                if last_idx is not None and idx <= last_idx:
                    idx = last_idx + 1
                y_coordinates[idx] = section_stations[i_seg]   
                last_idx = idx


            for i_seg in range(n_segments):
                if section_stations[i_seg] not in y_coordinates:
                    raise ValueError('VLM did not capture all section breaks')
                
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
                xi_a2 = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg]) + delta_x_a*idx_x + delta_x_a      # x coordinate of bottom left corner of bound vortex  
                xi_b1 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x                  # x coordinate of top right corner of panel     
                xi_b2 = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg]) + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel      
                xi_c  = segment_chord_x_offset[i_seg] + eta *np.tan(segment_sweep[i_seg])  + delta_x  *idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
                 

                # adjustment of coordinates for camber
                section_camber_a  = segment_camber[i_seg]*wing_chord_section_a  
                section_top_a     = segment_top_surface[i_seg]*wing_chord_section_a
                section_bot_a     = segment_bot_surface[i_seg]*wing_chord_section_a
                section_camber_b  = segment_camber[i_seg]*wing_chord_section_b  
                section_top_b     = segment_top_surface[i_seg]*wing_chord_section_b
                section_bot_b     = segment_bot_surface[i_seg]*wing_chord_section_b                
                section_camber_c  = segment_camber[i_seg]*wing_chord_section                
                section_x_coord_a = segment_x_coord[i_seg]*wing_chord_section_a
                section_x_coord_b = segment_x_coord[i_seg]*wing_chord_section_b
                section_x_coord   = segment_x_coord[i_seg]*wing_chord_section

                z_c_a1     = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_camber_a)  
                z_c_a1_top = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_top_a)
                z_c_a1_bot = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_bot_a) 
                z_c_a2     = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_camber_a)  
                z_c_a2_top = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_top_a) 
                z_c_a2_bot = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_bot_a)   
                z_c_b1     = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_camber_b)  
                z_c_b1_top = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_top_b)  
                z_c_b1_bot = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_bot_b)  
                z_c_b2     = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_camber_b) 
                z_c_b2_top = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_top_b) 
                z_c_b2_bot = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_bot_b)   
                z_c        = np.interp((idx_x    *delta_x   + delta_x  *0.75) ,section_x_coord,section_camber_c)  

                zeta_a1     = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a1      # z coordinate of top left corner of panel 
                zeta_a1_top = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a1_top
                zeta_a1_bot = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a1_bot 
                zeta_a2     = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a2      # z coordinate of bottom left corner of panel
                zeta_a2_top = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a2_top 
                zeta_a2_bot = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])  + z_c_a2_bot                         
                zeta_b1     = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b1      # z coordinate of top right corner of panel  
                zeta_b1_top = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b1_top
                zeta_b1_bot = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b1_bot   
                zeta_b2     = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b2      # z coordinate of bottom right corner of panel        
                zeta_b2_top = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b2_top 
                zeta_b2_bot = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])  + z_c_b2_bot 
                zeta        = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])    + z_c         # z coordinate three-quarter chord control point for each panel 

                # adjustment of coordinates for twist  
                xi_LE_a = segment_chord_x_offset[i_seg] + eta_a*np.tan(segment_sweep[i_seg])                       # x location of leading edge left corner of wing
                xi_LE_b = segment_chord_x_offset[i_seg] + eta_b*np.tan(segment_sweep[i_seg])                       # x location of leading edge right of wing
                xi_LE   = segment_chord_x_offset[i_seg] + eta*np.tan(segment_sweep[i_seg])                         # x location of leading edge center of wing
                                                                                                                   
                zeta_LE_a = segment_chord_z_offset[i_seg] + eta_a*np.tan(segment_dihedral[i_seg])                  # z location of leading edge left corner of wing
                zeta_LE_b = segment_chord_z_offset[i_seg] + eta_b*np.tan(segment_dihedral[i_seg])                  # z location of leading edge right of wing
                zeta_LE   = segment_chord_z_offset[i_seg] + eta*np.tan(segment_dihedral[i_seg])                    # z location of leading edge center of wing
                                                                                                                   
                # determine section twist                                                                          
                section_twist_a = segment_twist[i_seg] + (eta_a * segment_twist_ratio)                             # twist at left side of panel
                section_twist_b = segment_twist[i_seg] + (eta_b * segment_twist_ratio)                             # twist at right side of panel
                section_twist   = segment_twist[i_seg] + (eta* segment_twist_ratio)                                # twist at center local chord 

                xi_prime_a1        = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1-zeta_LE_a)        # x coordinate transformation of top left corner
                xi_prime_a1_top    = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1_top-zeta_LE_a)    
                xi_prime_a1_bot    = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1_bot-zeta_LE_a)    
                xi_prime_a2        = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)        # x coordinate transformation of bottom left corner
                xi_prime_a2_top    = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2_top-zeta_LE_a)    
                xi_prime_a2_bot    = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2_bot-zeta_LE_a)                          
                xi_prime_b1        = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)        # x coordinate transformation of top right corner  
                xi_prime_b1_top    = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1_top-zeta_LE_b)    
                xi_prime_b1_bot    = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1_bot-zeta_LE_b)    
                xi_prime_b2        = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2-zeta_LE_b)        # x coordinate transformation of botton right corner 
                xi_prime_b2_top    = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2_top-zeta_LE_b)    
                xi_prime_b2_bot    = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2_bot-zeta_LE_b)    
                xi_prime           = xi_LE   + np.cos(section_twist)  *(xi_c-xi_LE)    + np.sin(section_twist)*(zeta-zeta_LE)               # x coordinate transformation of control point 

                zeta_prime_a1      = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1-zeta_LE_a)      # z coordinate transformation of top left corner 
                zeta_prime_a1_top  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1_top-zeta_LE_a)
                zeta_prime_a1_bot  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1_bot-zeta_LE_a)
                zeta_prime_a2      = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2-zeta_LE_a)      # z coordinate transformation of bottom left corner
                zeta_prime_a2_top  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2_top-zeta_LE_a)
                zeta_prime_a2_bot  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2_bot-zeta_LE_a)                         
                zeta_prime_b1      = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1-zeta_LE_b)      # z coordinate transformation of top right corner  
                zeta_prime_b1_top  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1_top-zeta_LE_b)
                zeta_prime_b1_bot  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1_bot-zeta_LE_b)
                zeta_prime_b2      = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2-zeta_LE_b)      # z coordinate transformation of botton right corner 
                zeta_prime_b2_top  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2_top-zeta_LE_b) 
                zeta_prime_b2_bot  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2_bot-zeta_LE_b) 
                zeta_prime         = zeta_LE   - np.sin(section_twist)*(xi_c-xi_LE)      + np.cos(-section_twist)*(zeta-zeta_LE)            # z coordinate transformation of control point 

                # ** TO DO ** Get flap/aileron locations and deflection
                # store coordinates of panels, horseshoeces vortices and control points relative to wing root 
                if vertical_wing:
                    # mean camber line surface 
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
                                                                                                                                   
                    xa1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a1_top          ,xi_prime_a1_bot              ])
                    za1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    ya1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a1_top        ,zeta_prime_a1_bot            ])
                    xa2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a2_top          ,xi_prime_a2_bot              ])
                    za2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    ya2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a2_top        ,zeta_prime_a2_bot            ])
                    xb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b1_top          ,xi_prime_b1_bot              ])
                    zb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])
                    yb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b1_top        ,zeta_prime_b1_bot            ])
                    xb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b2_top          ,xi_prime_b2_bot              ])
                    zb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])                      
                    yb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b2_top        ,zeta_prime_b2_bot            ]) 

                    xc[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    zc[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                    yc[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime 

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
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2 
                    
                    xa1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a1_top          ,xi_prime_a1_bot              ])
                    ya1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    za1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a1_top        ,zeta_prime_a1_bot            ])
                    xa2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a2_top          ,xi_prime_a2_bot              ])
                    ya2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    za2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a2_top        ,zeta_prime_a2_bot            ])
                    xb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b1_top          ,xi_prime_b1_bot              ])
                    yb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])
                    zb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b1_top        ,zeta_prime_b1_bot            ])
                    xb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b2_top          ,xi_prime_b2_bot              ])
                    yb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])                      
                    zb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b2_top        ,zeta_prime_b2_bot            ])
 
                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime                  

                idx += 1

                cs_w[idx_y] = wing_chord_section       

                if y_b[idx_y] == section_stations[i_seg+1]: 
                    i_seg += 1                     

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
                airfoil_data     = import_airfoil_geometry([wing.Airfoil.airfoil.coordinate_file])    
                wing_camber      = airfoil_data.camber_coordinates[0]
                wing_top_surface = airfoil_data.y_upper_surface[0] 
                wing_bot_surface = airfoil_data.y_lower_surface[0] 
                wing_x_coord     = airfoil_data.x_lower_surface[0]
            else:
                dummy_dimension  = 30
                wing_camber      = np.zeros(dummy_dimension) # dimension of Selig airfoil VD file
                airfoil_data     = compute_naca_4series(0.0, 0.0,wing.thickness_to_chord,dummy_dimension*2 - 2 )
                wing_top_surface = airfoil_data.y_upper_surface[0] 
                wing_bot_surface = airfoil_data.y_lower_surface[0]                 
                wing_x_coord     = np.linspace(0,1,30) 
                    
            del_y = y_b - y_a
            for idx_y in range(n_sw):  
                idx_x = np.arange(n_cw) 
                eta_a = (y_a[idx_y])  
                eta_b = (y_b[idx_y]) 
                eta   = (y_b[idx_y] - del_y[idx_y]/2) 
                
                # get spanwise discretization points
                wing_chord_section_a  = root_chord + (eta_a*wing_chord_ratio) 
                wing_chord_section_b  = root_chord + (eta_b*wing_chord_ratio)
                wing_chord_section    = root_chord + (eta*wing_chord_ratio)
                
                # get chordwise discretization points
                delta_x_a = wing_chord_section_a/n_cw   
                delta_x_b = wing_chord_section_b/n_cw   
                delta_x   = wing_chord_section/n_cw                                  

                xi_a1 = eta_a*np.tan(sweep) + delta_x_a*idx_x                  # x coordinate of top left corner of panel 
                xi_a2 = eta_a*np.tan(sweep) + delta_x_a*idx_x + delta_x_a      # x coordinate of bottom left corner of bound vortex  
                xi_b1 = eta_b*np.tan(sweep) + delta_x_b*idx_x                  # x coordinate of top right corner of panel            
                xi_b2 = eta_b*np.tan(sweep) + delta_x_b*idx_x + delta_x_b
                xi_c  =  eta *np.tan(sweep)  + delta_x  *idx_x + delta_x*0.75  # x coordinate three-quarter chord control point for each panel 
                
                # adjustment of coordinates for camber
                section_camber_a  = wing_camber*wing_chord_section_a
                section_top_a     = wing_top_surface*wing_chord_section_a
                section_bot_a     = wing_bot_surface*wing_chord_section_a
                section_camber_b  = wing_camber*wing_chord_section_b                  
                section_top_b     = wing_top_surface*wing_chord_section_b                  
                section_bot_b     = wing_bot_surface*wing_chord_section_b  
                section_camber_c  = wing_camber*wing_chord_section  

                section_x_coord_a = wing_x_coord*wing_chord_section_a
                section_x_coord_b = wing_x_coord*wing_chord_section_b
                section_x_coord   = wing_x_coord*wing_chord_section 

                z_c_a1     = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_camber_a)  
                z_c_a1_top = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_top_a) 
                z_c_a1_bot = np.interp((idx_x    *delta_x_a)                  ,section_x_coord_a,section_bot_a) 
                z_c_a2     = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_camber_a) 
                z_c_a2_top = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_top_a)
                z_c_a2_bot = np.interp(((idx_x+1)*delta_x_a)                  ,section_x_coord_a,section_bot_a) 
                z_c_b1     = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_camber_b)    
                z_c_b1_top = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_top_b)
                z_c_b1_bot = np.interp((idx_x    *delta_x_b)                  ,section_x_coord_b,section_bot_b)
                z_c_b2     = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_camber_b) 
                z_c_b2_top = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_top_b) 
                z_c_b2_bot = np.interp(((idx_x+1)*delta_x_b)                  ,section_x_coord_b,section_bot_b)  
                z_c        = np.interp((idx_x    *delta_x   + delta_x  *0.75) ,section_x_coord  ,section_camber_c)  

                zeta_a1     = eta_a*np.tan(dihedral)  + z_c_a1      # z coordinate of top left corner of panel 
                zeta_a1_top = eta_a*np.tan(dihedral)  + z_c_a1_top  # z coordinate of top left corner of panel on surface  
                zeta_a1_bot = eta_a*np.tan(dihedral)  + z_c_a1_bot  # z coordinate of top left corner of panel on surface  
                zeta_a2     = eta_a*np.tan(dihedral)  + z_c_a2      # z coordinate of bottom left corner of panel
                zeta_a2_top = eta_a*np.tan(dihedral)  + z_c_a2_top  # z coordinate of bottom left corner of panel
                zeta_a2_bot = eta_a*np.tan(dihedral)  + z_c_a2_bot  # z coordinate of bottom left corner of panel                    
                zeta_b1     = eta_b*np.tan(dihedral)  + z_c_b1      # z coordinate of top right corner of panel    
                zeta_b1_top = eta_b*np.tan(dihedral)  + z_c_b1_top  # z coordinate of top right corner of panel    
                zeta_b1_bot = eta_b*np.tan(dihedral)  + z_c_b1_bot  # z coordinate of top right corner of panel    
                zeta_b2     = eta_b*np.tan(dihedral)  + z_c_b2      # z coordinate of bottom right corner of panel   
                zeta_b2_top = eta_b*np.tan(dihedral)  + z_c_b2_top  # z coordinate of bottom right corner of panel  
                zeta_b2_bot = eta_b*np.tan(dihedral)  + z_c_b2_bot  # z coordinate of bottom right corner of panel  
                zeta        =   eta*np.tan(dihedral)  + z_c         # z coordinate three-quarter chord control point for each panel  

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
 
                xi_prime_a1      = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1-zeta_LE_a)         # x coordinate transformation of top left corner
                xi_prime_a1_top  = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1_top-zeta_LE_a)     
                xi_prime_a1_bot  = xi_LE_a + np.cos(section_twist_a)*(xi_a1-xi_LE_a) + np.sin(section_twist_a)*(zeta_a1_bot-zeta_LE_a)      
                xi_prime_a2      = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2-zeta_LE_a)         # x coordinate transformation of bottom left corner
                xi_prime_a2_top  = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2_top-zeta_LE_a)      
                xi_prime_a2_bot  = xi_LE_a + np.cos(section_twist_a)*(xi_a2-xi_LE_a) + np.sin(section_twist_a)*(zeta_a2_bot-zeta_LE_a)                             
                xi_prime_b1      = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1-zeta_LE_b)         # x coordinate transformation of top right corner  
                xi_prime_b1_top  = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1_top-zeta_LE_b)      
                xi_prime_b1_bot  = xi_LE_b + np.cos(section_twist_b)*(xi_b1-xi_LE_b) + np.sin(section_twist_b)*(zeta_b1_bot-zeta_LE_b)     
                xi_prime_b2      = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2-zeta_LE_b)         # x coordinate transformation of botton right corner 
                xi_prime_b2_top  = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2_top-zeta_LE_b)    
                xi_prime_b2_bot  = xi_LE_b + np.cos(section_twist_b)*(xi_b2-xi_LE_b) + np.sin(section_twist_b)*(zeta_b2_bot-zeta_LE_b)    
                xi_prime         = xi_LE   + np.cos(section_twist)  *(xi_c-xi_LE)    + np.sin(section_twist)*(zeta-zeta_LE)                # x coordinate transformation of control point 

                zeta_prime_a1      = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1-zeta_LE_a)     # z coordinate transformation of top left corner 
                zeta_prime_a1_top  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1_top-zeta_LE_a)
                zeta_prime_a1_bot  = zeta_LE_a - np.sin(section_twist_a)*(xi_a1-xi_LE_a) + np.cos(section_twist_a)*(zeta_a1_bot-zeta_LE_a)
                zeta_prime_a2      = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2-zeta_LE_a)     # z coordinate transformation of bottom left corner
                zeta_prime_a2_top  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2_top-zeta_LE_a)
                zeta_prime_a2_bot  = zeta_LE_a - np.sin(section_twist_a)*(xi_a2-xi_LE_a) + np.cos(section_twist_a)*(zeta_a2_bot-zeta_LE_a)                         
                zeta_prime_b1      = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1-zeta_LE_b)     # z coordinate transformation of top right corner  
                zeta_prime_b1_top  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1_top-zeta_LE_b)
                zeta_prime_b1_bot  = zeta_LE_b - np.sin(section_twist_b)*(xi_b1-xi_LE_b) + np.cos(section_twist_b)*(zeta_b1_bot-zeta_LE_b)
                zeta_prime_b2      = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2-zeta_LE_b)      # z coordinate transformation of botton right corner 
                zeta_prime_b2_top  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2_top-zeta_LE_b) 
                zeta_prime_b2_bot  = zeta_LE_b - np.sin(section_twist_b)*(xi_b2-xi_LE_b) + np.cos(section_twist_b)*(zeta_b2_bot-zeta_LE_b) 
                zeta_prime         = zeta_LE   - np.sin(section_twist)*(xi_c-xi_LE)      + np.cos(-section_twist)*(zeta-zeta_LE)            # z coordinate transformation of control point 
  
 
                # store coordinates of panels, horseshoe vortices and control points relative to wing root 
                if vertical_wing: 
                    
                    # mean camber line surface 
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
                
                    xa1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a1_top          ,xi_prime_a1_bot              ])
                    za1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    ya1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a1_top        ,zeta_prime_a1_bot            ])
                    xa2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a2_top          ,xi_prime_a2_bot              ])
                    za2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    ya2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a2_top        ,zeta_prime_a2_bot            ])
                    xb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b1_top          ,xi_prime_b1_bot              ])
                    zb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])
                    yb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b1_top        ,zeta_prime_b1_bot            ])
                    xb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b2_top          ,xi_prime_b2_bot              ])
                    zb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])                      
                    yb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b2_top        ,zeta_prime_b2_bot            ])   

                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime                       

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
                    xb2[idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime_b2
                    yb2[idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*y_b[idx_y]
                    zb2[idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime_b2   
                    
                    xa1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a1_top          ,xi_prime_a1_bot              ])
                    ya1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    za1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a1_top        ,zeta_prime_a1_bot            ])
                    xa2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_a2_top          ,xi_prime_a2_bot              ])
                    ya2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_a[idx_y] ,np.ones(n_cw)*y_a[idx_y]     ])
                    za2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_a2_top        ,zeta_prime_a2_bot            ])
                    xb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b1_top          ,xi_prime_b1_bot              ])
                    yb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])
                    zb1_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b1_top        ,zeta_prime_b1_bot            ])
                    xb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ xi_prime_b2_top          ,xi_prime_b2_bot              ])
                    yb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ np.ones(n_cw)*y_b[idx_y] ,np.ones(n_cw)*y_b[idx_y]     ])                      
                    zb2_ts[idx_y*(2*n_cw):(idx_y+1)*(2*n_cw)] = np.concatenate([ zeta_prime_b2_top        ,zeta_prime_b2_bot            ])
                    
                    
                    xc [idx_y*n_cw:(idx_y+1)*n_cw] = xi_prime 
                    yc [idx_y*n_cw:(idx_y+1)*n_cw] = np.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2) 
                    zc [idx_y*n_cw:(idx_y+1)*n_cw] = zeta_prime       

                cs_w[idx_y] = wing_chord_section 
    
        # adjusting coordinate axis so reference point is at the nose of the aircraft 

        xa1 = xa1 + wing_origin[0]         # x coordinate of top left corner of panel
        ya1 = ya1 + wing_origin[1]         # y coordinate of bottom left corner of panel
        za1 = za1 + wing_origin[2]         # z coordinate of top left corner of panel
        xa2 = xa2 + wing_origin[0]         # x coordinate of bottom left corner of panel
        ya2 = ya2 + wing_origin[1]         # x coordinate of bottom left corner of panel
        za2 = za2 + wing_origin[2]         # z coordinate of bottom left corner of panel   
        xb1 = xb1 + wing_origin[0]         # x coordinate of top right corner of panel  
        yb1 = yb1 + wing_origin[1]         # y coordinate of top right corner of panel 
        zb1 = zb1 + wing_origin[2]         # z coordinate of top right corner of panel 
        xb2 = xb2 + wing_origin[0]         # x coordinate of bottom rightcorner of panel 
        yb2 = yb2 + wing_origin[1]         # y coordinate of bottom rightcorner of panel 
        zb2 = zb2 + wing_origin[2]         # z coordinate of bottom right corner of panel            
        
        xa1_ts   = xa1_ts + wing_origin[0] # x coordinate of top left corner of panel
        ya1_ts   = ya1_ts + wing_origin[1] # y coordinate of bottom left corner of panel
        za1_ts   = za1_ts + wing_origin[2] # z coordinate of top left corner of panel
        xa2_ts   = xa2_ts + wing_origin[0] # x coordinate of bottom left corner of panel
        ya2_ts   = ya2_ts + wing_origin[1] # x coordinate of bottom left corner of panel
        za2_ts   = za2_ts + wing_origin[2] # z coordinate of bottom left corner of panel   
        xb1_ts   = xb1_ts + wing_origin[0] # x coordinate of top right corner of panel  
        yb1_ts   = yb1_ts + wing_origin[1] # y coordinate of top right corner of panel 
        zb1_ts   = zb1_ts + wing_origin[2] # z coordinate of top right corner of panel 
        xb2_ts   = xb2_ts + wing_origin[0] # x coordinate of bottom rightcorner of panel 
        yb2_ts   = yb2_ts + wing_origin[1] # y coordinate of bottom rightcorner of panel 
        zb2_ts   = zb2_ts + wing_origin[2] # z coordinate of bottom right corner of panel      

        # if symmetry, store points of mirrored wing 
        n_w += 1  
        if sym_para is True :
            n_w += 1 
            # append wing spans          
            if vertical_wing:
                del_y    = np.concatenate([del_y,del_y]) 
                cs_w     = np.concatenate([cs_w,cs_w]) 
                         
                xa1   = np.concatenate([xa1,xa1])
                ya1   = np.concatenate([ya1,ya1])
                za1   = np.concatenate([za1,-za1])
                xa2   = np.concatenate([xa2,xa2])
                ya2   = np.concatenate([ya2,ya2])
                za2   = np.concatenate([za2,-za2]) 
                xb1   = np.concatenate([xb1,xb1])
                yb1   = np.concatenate([yb1,yb1])    
                zb1   = np.concatenate([zb1,-zb1])
                xb2   = np.concatenate([xb2,xb2])
                yb2   = np.concatenate([yb2,yb2])            
                zb2   = np.concatenate([zb2,-zb2])
                
                xa1_ts   = np.concatenate([xa1_ts, xa1_ts])
                ya1_ts   = np.concatenate([ya1_ts, ya1_ts])
                za1_ts   = np.concatenate([za1_ts,-za1_ts])
                xa2_ts   = np.concatenate([xa2_ts, xa2_ts])
                ya2_ts   = np.concatenate([ya2_ts, ya2_ts])
                za2_ts   = np.concatenate([za2_ts,-za2_ts]) 
                xb1_ts   = np.concatenate([xb1_ts, xb1_ts])
                yb1_ts   = np.concatenate([yb1_ts, yb1_ts])    
                zb1_ts   = np.concatenate([zb1_ts,-zb1_ts])
                xb2_ts   = np.concatenate([xb2_ts, xb2_ts])
                yb2_ts   = np.concatenate([yb2_ts, yb2_ts])            
                zb2_ts   = np.concatenate([zb2_ts,-zb2_ts])
                 
                xc       = np.concatenate([xc ,xc ])
                yc       = np.concatenate([yc ,yc]) 
                zc       = np.concatenate([zc ,-zc ])                 
                
            else: 
                xa1   = np.concatenate([xa1,xa1])
                ya1   = np.concatenate([ya1,-ya1])
                za1   = np.concatenate([za1,za1])
                xa2   = np.concatenate([xa2,xa2])
                ya2   = np.concatenate([ya2,-ya2])
                za2   = np.concatenate([za2,za2]) 
                xb1   = np.concatenate([xb1,xb1])
                yb1   = np.concatenate([yb1,-yb1])    
                zb1   = np.concatenate([zb1,zb1])
                xb2   = np.concatenate([xb2,xb2])
                yb2   = np.concatenate([yb2,-yb2])            
                zb2   = np.concatenate([zb2,zb2])
                
                xa1_ts  = np.concatenate([xa1_ts, xa1_ts])
                ya1_ts  = np.concatenate([ya1_ts,-ya1_ts])
                za1_ts  = np.concatenate([za1_ts, za1_ts])
                xa2_ts  = np.concatenate([xa2_ts, xa2_ts])
                ya2_ts  = np.concatenate([ya2_ts,-ya2_ts])
                za2_ts  = np.concatenate([za2_ts, za2_ts]) 
                xb1_ts  = np.concatenate([xb1_ts, xb1_ts])
                yb1_ts  = np.concatenate([yb1_ts,-yb1_ts])    
                zb1_ts  = np.concatenate([zb1_ts, zb1_ts])
                xb2_ts  = np.concatenate([xb2_ts, xb2_ts])
                yb2_ts  = np.concatenate([yb2_ts,-yb2_ts])            
                zb2_ts  = np.concatenate([zb2_ts, zb2_ts])
                
                xc       = np.concatenate([xc , xc ])
                yc       = np.concatenate([yc ,-yc]) 
                zc       = np.concatenate([zc , zc ])            

        n_cp += len(xa1)        
        n_sp += len(xa1_ts)  
        # ---------------------------------------------------------------------------------------
        # STEP 7: Store wing in vehicle vector
        # ---------------------------------------------------------------------------------------               
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
                  
        VD.XA1_ts = np.append(VD.XA1_ts,xa1_ts)
        VD.YA1_ts = np.append(VD.YA1_ts,ya1_ts)
        VD.ZA1_ts = np.append(VD.ZA1_ts,za1_ts)
        VD.XA2_ts = np.append(VD.XA2_ts,xa2_ts)
        VD.YA2_ts = np.append(VD.YA2_ts,ya2_ts)
        VD.ZA2_ts = np.append(VD.ZA2_ts,za2_ts)        
        VD.XB1_ts = np.append(VD.XB1_ts,xb1_ts)
        VD.YB1_ts = np.append(VD.YB1_ts,yb1_ts)
        VD.ZB1_ts = np.append(VD.ZB1_ts,zb1_ts)
        VD.XB2_ts = np.append(VD.XB2_ts,xb2_ts)                
        VD.YB2_ts = np.append(VD.YB2_ts,yb2_ts)        
        VD.ZB2_ts = np.append(VD.ZB2_ts,zb2_ts)    
        VD.XC     = np.append(VD.XC ,xc)
        VD.YC     = np.append(VD.YC ,yc)
        VD.ZC     = np.append(VD.ZC ,zc)     
        
    VD.n_sw       = n_sw
    VD.n_cw       = n_cw 
    VD.n_cp       = n_cp  
    VD.n_sp       = n_sp  
    
    # Compute Panel Normals
    #VD.normals = compute_unit_normal(VD)
    
    return VD 