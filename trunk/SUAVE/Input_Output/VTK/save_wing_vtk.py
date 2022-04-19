## @ingroup Input_Output-VTK
# save_wing_vtk.py
#
# Created:    Jun 2021, R. Erhard
# Modified:
#
import SUAVE
from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.generate_vortex_distribution import generate_vortex_distribution
from copy import deepcopy
import numpy as np

from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.extract_wing_VD import extract_wing_collocation_points

## @ingroup Input_Output-VTK
def save_wing_vtk(vehicle, wing_instance, settings, filename, Results,time_step,origin_offset):
    """
    Saves a SUAVE wing object as a VTK in legacy format.

    Inputs:
       vehicle        Data structure of SUAVE vehicle                [Unitless]
       wing_instance  Data structure of SUAVE wing                   [Unitless]
       filename       Name of vtk file to save                       [String]
       Results        Data structure of wing and propeller results   [Unitless]

    Outputs:
       N/A

    Properties Used:
       N/A

    Assumptions:
       N/A

    Source:
       None

    """


    try:
        VD = vehicle.vortex_distribution
    except:
        # generate VD for this wing alone
        wing_vehicle = SUAVE.Vehicle()
        wing_vehicle.append_component(wing_instance)
        VD = generate_vortex_distribution(wing_vehicle,settings)
    
    if "VLM_wings" in vehicle.vortex_distribution.keys():
        wings = vehicle.vortex_distribution.VLM_wings
    else:
        wings = vehicle.wings
        
    for i,wing in enumerate(wings):
        if wing == wing_instance:
            wing_instance_idx = i
            
    VD_wing, ids = extract_wing_collocation_points(vehicle, wing_instance_idx)
    
    symmetric = wing_instance.symmetric
    
    if symmetric:
        # Split into half wings for plotting
        sec_start = 0
        sec_end   = len(VD_wing.XA1)
        half_l    =  sec_end // 2
        
        n_cw_L = VD_wing.n_cw[0]
        n_sw_L = VD_wing.n_sw[0]
        n_cp_L = n_cw_L*n_sw_L

        n_cw_R = VD_wing.n_cw[1]
        n_sw_R = VD_wing.n_sw[1]    
        n_cp_R = n_cw_R*n_sw_R

        # split wing into two separate wings
        Rwing = Data()
        Lwing = Data()

        Rwing.XA1 = VD_wing.XA1[sec_start:half_l] + origin_offset[0]
        Rwing.XA2 = VD_wing.XA2[sec_start:half_l] + origin_offset[0]
        Rwing.XB1 = VD_wing.XB1[sec_start:half_l] + origin_offset[0]
        Rwing.XB2 = VD_wing.XB2[sec_start:half_l] + origin_offset[0]
        Rwing.YA1 = VD_wing.YA1[sec_start:half_l] + origin_offset[1]
        Rwing.YA2 = VD_wing.YA2[sec_start:half_l] + origin_offset[1]
        Rwing.YB1 = VD_wing.YB1[sec_start:half_l] + origin_offset[1]
        Rwing.YB2 = VD_wing.YB2[sec_start:half_l] + origin_offset[1]
        Rwing.ZA1 = VD_wing.ZA1[sec_start:half_l] + origin_offset[2]
        Rwing.ZA2 = VD_wing.ZA2[sec_start:half_l] + origin_offset[2]
        Rwing.ZB1 = VD_wing.ZB1[sec_start:half_l] + origin_offset[2]
        Rwing.ZB2 = VD_wing.ZB2[sec_start:half_l] + origin_offset[2]
        
        Lwing.XA1 = VD_wing.XA1[half_l:sec_end] + origin_offset[0]
        Lwing.XA2 = VD_wing.XA2[half_l:sec_end] + origin_offset[0]
        Lwing.XB1 = VD_wing.XB1[half_l:sec_end] + origin_offset[0]
        Lwing.XB2 = VD_wing.XB2[half_l:sec_end] + origin_offset[0]
        Lwing.YA1 = VD_wing.YA1[half_l:sec_end] + origin_offset[1]
        Lwing.YA2 = VD_wing.YA2[half_l:sec_end] + origin_offset[1]
        Lwing.YB1 = VD_wing.YB1[half_l:sec_end] + origin_offset[1]
        Lwing.YB2 = VD_wing.YB2[half_l:sec_end] + origin_offset[1]
        Lwing.ZA1 = VD_wing.ZA1[half_l:sec_end] + origin_offset[2]
        Lwing.ZA2 = VD_wing.ZA2[half_l:sec_end] + origin_offset[2]
        Lwing.ZB1 = VD_wing.ZB1[half_l:sec_end] + origin_offset[2]
        Lwing.ZB2 = VD_wing.ZB2[half_l:sec_end] + origin_offset[2]

        R_Results = deepcopy(Results)
        L_Results = deepcopy(Results)
        if 'vlm_results' in Results.keys():
            if Results.vlm_results is not None:      
                
                spanwise_start_idx = VD.spanwise_breaks[wing_instance_idx]
                spanwise_mid_idx   = VD.spanwise_breaks[wing_instance_idx+1]
                spanwise_end_idx   = VD.spanwise_breaks[wing_instance_idx+1]+VD.n_sw[wing_instance_idx]
                
                span_ids_R = range(spanwise_start_idx, spanwise_mid_idx)
                span_ids_L = range(spanwise_mid_idx, spanwise_end_idx)
                    
                R_Results.vlm_results.CP     = Results.vlm_results.CP[0][ids][0:half_l]
                R_Results.vlm_results.cl_y   = Results.vlm_results.cl_y[0][span_ids_R]
                R_Results.vlm_results.cdi_y  = Results.vlm_results.cdi_y[0][span_ids_R]
                #R_Results.vlm_results.LIFT   = Results.vlm_results.LIFT[0][span_ids]
                #R_Results.vlm_results.DRAG   = Results.vlm_results.DRAG[0][span_ids]
                #R_Results.vlm_results.MOMENT = Results.vlm_results.CM_y[0][span_ids]
                
                L_Results.vlm_results.CP     = Results.vlm_results.CP[0][ids][half_l:]
                L_Results.vlm_results.cl_y   = Results.vlm_results.cl_y[0][span_ids_L]
                L_Results.vlm_results.cdi_y  = Results.vlm_results.cdi_y[0][span_ids_L]
                #L_Results.vlm_results.LIFT   = Results.vlm_results.LIFT[0][span_ids]
                #L_Results.vlm_results.DRAG   = Results.vlm_results.DRAG[0][span_ids]
                #L_Results.vlm_results.MOMENT = Results.vlm_results.CM_y[0][span_ids]
                
                # Check for isolated 
                R_Results.vlm_results.CP_ISO = Results.vlm_results.iso.CP[0][ids][0:half_l]
                L_Results.vlm_results.CP_ISO = Results.vlm_results.iso.CP[0][ids][half_l:]

        sep  = filename.rfind('.')

        Lfile = filename[0:sep]+"_L"+"_t"+str(time_step)+filename[sep:]
        Rfile = filename[0:sep]+"_R"+"_t"+str(time_step)+filename[sep:]

        # write vtks for each half wing
        write_wing_vtk(Lwing,n_cw_L,n_sw_L,n_cp_L,L_Results,Lfile)
        write_wing_vtk(Rwing,n_cw_R,n_sw_R,n_cp_R,R_Results,Rfile)        
    else:
        n_cw = VD.n_cw[0]
        n_sw = VD.n_sw[0]
        
        VD_wing.XA1 += origin_offset[0]
        VD_wing.XA2 += origin_offset[0]
        VD_wing.XB1 += origin_offset[0]
        VD_wing.XB2 += origin_offset[0]
        VD_wing.YA1 += origin_offset[1]
        VD_wing.YA2 += origin_offset[1]
        VD_wing.YB1 += origin_offset[1]
        VD_wing.YB2 += origin_offset[1]
        VD_wing.ZA1 += origin_offset[2]
        VD_wing.ZA2 += origin_offset[2]
        VD_wing.ZB1 += origin_offset[2]
        VD_wing.ZB2 += origin_offset[2]
        
        n_cp = n_cw*n_sw
        sep  = filename.rfind('.')
        file = filename[0:sep]+"_t"+str(time_step)+filename[sep:]

        if 'vlm_results' in Results.keys():
            span_ids = range(VD.spanwise_breaks[wing_instance_idx], VD.spanwise_breaks[wing_instance_idx]+VD.n_sw[wing_instance_idx])    
            
            Results.vlm_results.CP     = Results.vlm_results.CP[0][ids]
            Results.vlm_results.cl_y   = Results.vlm_results.cl_y[0][span_ids]
            Results.vlm_results.cdi_y  = Results.vlm_results.cdi_y[0][span_ids]
            #Results.vlm_results.LIFT   = Results.vlm_results.LIFT[0][span_ids]
            #Results.vlm_results.DRAG   = Results.vlm_results.DRAG[0][span_ids]
            #Results.vlm_results.MOMENT = Results.vlm_results.CM_y[0][span_ids]

        write_wing_vtk(VD_wing,n_cw,n_sw,n_cp,Results,file)    
    
    
    return

## @ingroup Input_Output-VTK
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

            elif i//(n_cw+cw_laps*(n_cw+1))==1:
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
        f.write("\nSCALARS i float 1")
        f.write("\nLOOKUP_TABLE default")
        for i in range(n_cp):
            new_idx = str(i)
            f.write("\n"+new_idx)
        
        if Results is not None:
            if 'vlm_results' in Results.keys():
                # for spanwise elements, repeat along chord

                cl = Results.vlm_results.cl_y
                f.write("\nSCALARS cl float 1")
                f.write("\nLOOKUP_TABLE default")
                for i in range(n_cp):
                    new_cl = str(cl[i//n_cw])
                    f.write("\n"+new_cl)
                
                cl = Results.vlm_results.cl_y
                CL = Results.vlm_results.CL[0][0]
                f.write("\nSCALARS Cl/CL float 1")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_cp):
                    new_cl_CL = str(cl[i//n_cw]/CL)
                    f.write("\n"+new_cl_CL)
                
                cd = Results.vlm_results.cdi_y
                f.write("\nSCALARS cdi float 1")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_cp):
                    new_cd = str(cd[i//n_cw])
                    f.write("\n"+new_cd)

                cd = Results.vlm_results.cdi_y
                CD = Results.vlm_results.CDi[0][0]

                f.write("\nSCALARS cd_CD float 1")
                f.write("\nLOOKUP_TABLE default")

                for i in range(n_cp):
                    new_cd_CD = str(cd[i//n_sw]/CD)
                    f.write("\n"+new_cd_CD)
 
                
                CP     = Results.vlm_results.CP
                CP_iso = Results.vlm_results.CP_ISO
                #LIFT   = Results.vlm_results.LIFT
                #DRAG   = Results.vlm_results.DRAG
                #MOMENT = Results.vlm_results.MOMENT

                f.write("\nSCALARS CP float 1")
                f.write("\nLOOKUP_TABLE default")
                for i in range(n_cp):
                    new_CP = str(CP[i])
                    f.write("\n"+new_CP)
                    
    
                f.write("\nSCALARS CP_diff float 1")
                f.write("\nLOOKUP_TABLE default")
                for i in range(n_cp):
                    new_CP = str(CP[i] - CP_iso[i])
                    f.write("\n"+new_CP)
                                        
                #f.write("\nSCALARS LIFT float 1")
                #f.write("\nLOOKUP_TABLE default")
                #for i in range(n_cp):
                    #new_val = str(LIFT[i//n_cw])
                    #f.write("\n"+new_val)      
                
                #f.write("\nSCALARS DRAG float 1")
                #f.write("\nLOOKUP_TABLE default")
                #for i in range(n_cp):
                    #new_val = str(DRAG[i//n_cw])
                    #f.write("\n"+new_val)   
                    
                #f.write("\nSCALARS MOMENT float 1")
                #f.write("\nLOOKUP_TABLE default")
                #for i in range(n_cp):
                    #new_val = str(MOMENT[i//n_cw])
                    #f.write("\n"+new_val)   
                                
                

    f.close()
    return
