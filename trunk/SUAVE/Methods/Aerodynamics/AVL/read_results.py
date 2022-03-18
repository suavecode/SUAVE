## @ingroup Methods-Aerodynamics-AVL
#read_results.py
# 
# Created:  Mar 2015, T. Momose
# Modified: Jan 2016, E. Botero
#           Dec 2017, M. Clarke
#           Aug 2019, M. Clarke
#           Dec 2021, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Methods.Aerodynamics.AVL.Data.Wing import Control_Surface_Data ,  Control_Surface_Results 
import numpy as np 

## @ingroup Methods-Aerodynamics-AVL
def read_results(avl_object):
    """ This functions reads the results from the results text file created 
    at the end of an AVL function call

    Assumptions:
        None
        
    Source:
        Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avl

    Inputs:
        None

    Outputs:
        results     

    Properties Used:
        N/A
    """    
    # unpack
    aircraft = avl_object.geometry
    results  = Data()
    case_idx = 0  
    for case in avl_object.current_status.cases:
        num_ctrl =  case.stability_and_control.number_control_surfaces
        # open newly written result files and read in aerodynamic properties 
        with open(case.aero_result_filename_1,'r') as stab_der_vile:
            # Extract results from stability axis derivatives file                                                                
            case_res                                                        = Data()  
            case_res.aerodynamics                                           = Data()
            case_res.stability                                              = Data()
            case_res.stability.control_surfaces                             = Control_Surface_Data()   
            case_res.stability.alpha_derivatives                            = Data()
            case_res.stability.beta_derivatives                             = Data()   
                                                                            
            case_res.tag                                                    = case.tag 
            lines                                                           = stab_der_vile.readlines()
            case_res.S_ref                                                  = float(lines[8][10:16].strip())
            case_res.c_ref                                                  = float(lines[8][31:37].strip())
            case_res.b_ref                                                  = float(lines[8][52:58].strip())
            case_res.X_ref                                                  = float(lines[9][10:16].strip())
            case_res.Y_ref                                                  = float(lines[9][31:37].strip())
            case_res.Z_ref                                                  = float(lines[9][52:58].strip())  
                                                                            
            case_res.aerodynamics.AoA                                       = float(lines[15][10:19].strip())
            case_res.aerodynamics.CX                                        = float(lines[19][11:19].strip())
            case_res.aerodynamics.CY                                        = float(lines[20][11:19].strip()) 
            case_res.aerodynamics.CZ                                        = float(lines[21][11:19].strip())
                                                                            
            case_res.aerodynamics.Cltot                                     = float(lines[19][33:41].strip())
            case_res.aerodynamics.Cmtot                                     = float(lines[20][33:41].strip()) 
            case_res.aerodynamics.Cntot                                     = float(lines[21][33:41].strip())
                                                                            
            case_res.aerodynamics.roll_moment_coefficient                   = float(lines[19][32:42].strip())
            case_res.aerodynamics.pitch_moment_coefficient                  = float(lines[20][32:42].strip())
            case_res.aerodynamics.yaw_moment_coefficient                    = float(lines[21][32:42].strip())
            case_res.aerodynamics.total_lift_coefficient                    = float(lines[23][10:20].strip())
            case_res.aerodynamics.total_drag_coefficient                    = float(lines[24][10:20].strip())
            case_res.aerodynamics.viscous_drag_coefficient                  = float(lines[25][10:20].strip())
            case_res.aerodynamics.induced_drag_coefficient                  = float(lines[25][32:42].strip())
            case_res.aerodynamics.oswald_efficiency                         = float(lines[27][32:42].strip())
               
            case_res.stability.alpha_derivatives.lift_curve_slope           = float(lines[36+num_ctrl][24:34].strip()) # CL_a
            case_res.stability.alpha_derivatives.side_force_derivative      = float(lines[37+num_ctrl][24:34].strip()) # CY_a
            case_res.stability.alpha_derivatives.roll_moment_derivative     = float(lines[38+num_ctrl][24:34].strip()) # Cl_a
            case_res.stability.alpha_derivatives.pitch_moment_derivative    = float(lines[39+num_ctrl][24:34].strip()) # Cm_a
            case_res.stability.alpha_derivatives.yaw_moment_derivative      = float(lines[40+num_ctrl][24:34].strip()) # Cn_a
            case_res.stability.beta_derivatives.lift_coefficient_derivative = float(lines[36+num_ctrl][43:54].strip()) # CL_b
            case_res.stability.beta_derivatives.side_force_derivative       = float(lines[37+num_ctrl][43:54].strip()) # CY_b
            case_res.stability.beta_derivatives.roll_moment_derivative      = float(lines[38+num_ctrl][43:54].strip()) # Cl_b
            case_res.stability.beta_derivatives.pitch_moment_derivative     = float(lines[39+num_ctrl][43:54].strip()) # Cm_b
            case_res.stability.beta_derivatives.yaw_moment_derivative       = float(lines[40+num_ctrl][43:54].strip()) # Cn_b
            case_res.stability.CL_p                                         = float(lines[44+num_ctrl][24:34].strip())
            case_res.stability.CL_q                                         = float(lines[44+num_ctrl][43:54].strip())
            case_res.stability.CL_r                                         = float(lines[44+num_ctrl][65:74].strip())
            case_res.stability.CY_p                                         = float(lines[45+num_ctrl][24:34].strip())
            case_res.stability.CY_q                                         = float(lines[45+num_ctrl][43:54].strip())
            case_res.stability.CY_r                                         = float(lines[45+num_ctrl][65:74].strip())
            case_res.stability.Cl_p                                         = float(lines[46+num_ctrl][24:34].strip())
            case_res.stability.Cl_q                                         = float(lines[46+num_ctrl][43:54].strip())
            case_res.stability.Cl_r                                         = float(lines[46+num_ctrl][65:74].strip())
            case_res.stability.Cm_p                                         = float(lines[47+num_ctrl][24:34].strip())
            case_res.stability.Cm_q                                         = float(lines[47+num_ctrl][43:54].strip())
            case_res.stability.Cm_r                                         = float(lines[44+num_ctrl][65:74].strip())
            case_res.stability.Cn_p                                         = float(lines[48+num_ctrl][24:34].strip())
            case_res.stability.Cn_q                                         = float(lines[48+num_ctrl][43:54].strip())
            case_res.stability.Cn_r                                         = float(lines[48+num_ctrl][65:74].strip())
             
            # this block of text reads in aerodynamic results related to the defined control surfaces 
            if num_ctrl != 0: 
                for ctrl_idx in range(num_ctrl):
                    ctrl_surf = Control_Surface_Results()
                    ctrl_surf.tag                 = str(lines[29+ctrl_idx][2:11].strip())
                    ctrl_surf.deflection          = float(lines[29+ctrl_idx][21:32].strip())
                    ctrl_surf.CL                  = float(lines[52+num_ctrl][(20*ctrl_idx + 23):(20*ctrl_idx + 34)].strip())
                    ctrl_surf.CY                  = float(lines[53+num_ctrl][(20*ctrl_idx + 23):(20*ctrl_idx + 34)].strip())
                    ctrl_surf.Cl                  = float(lines[54+num_ctrl][(20*ctrl_idx + 23):(20*ctrl_idx + 34)].strip())
                    ctrl_surf.Cm                  = float(lines[55+num_ctrl][(20*ctrl_idx + 23):(20*ctrl_idx + 34)].strip())
                    ctrl_surf.Cn                  = float(lines[56+num_ctrl][(20*ctrl_idx + 23):(20*ctrl_idx + 34)].strip())
                    ctrl_surf.CDff                = float(lines[57+num_ctrl][(20*ctrl_idx + 23):(20*ctrl_idx + 34)].strip())
                    ctrl_surf.e                   = float(lines[58+num_ctrl][(20*ctrl_idx + 23):(20*ctrl_idx + 34)].strip())
                    case_res.stability.control_surfaces.append_control_surface_result(ctrl_surf)             
            case_res.stability.neutral_point      = float(lines[50+12*(num_ctrl>0)+num_ctrl][22:33].strip())    
            case_res.stability.spiral_criteria    = float(lines[52+12*(num_ctrl>0)+num_ctrl][22:33].strip())    
        
        # get number of wings, spanwise discretization for surface and strip force result extraction
        n_sw    = avl_object.settings.number_spanwise_vortices
        n_wings = 0 
        for wing in aircraft.wings:
            n_wings += 1
            if wing.symmetric:
                n_wings += 1   
        n_fus_sec = 0 
        for fuselage in aircraft.fuselages:
            n_fus_sec += 2
        
        wing_area            = np.zeros(n_wings)
        wing_CL              = np.zeros(n_wings)
        wing_CD              = np.zeros(n_wings)  
        wing_local_span      = np.zeros((n_wings,n_sw))
        wing_sectional_chord = np.zeros((n_wings,n_sw))
        wing_cl              = np.zeros((n_wings,n_sw))
        alpha_i              = np.zeros((n_wings,n_sw))
        wing_cd              = np.zeros((n_wings,n_sw))   
        
        # Extract resulst from surface forces result file
        with open(case.aero_result_filename_2,'r') as aero_res_file:
            aero_lines   = aero_res_file.readlines()
            line_idx     = 0
            header       = 12 + n_wings + n_fus_sec           
            for i in range(n_wings):
                wing_area[i] = float(aero_lines[header + line_idx][7:14].strip())
                wing_CL[i]   = float(aero_lines[header + line_idx][26:32].strip())
                wing_CD[i]   = float(aero_lines[header + line_idx][35:41].strip())
                line_idx += 1                   
            case_res.aerodynamics.wing_areas = wing_area 
            case_res.aerodynamics.wing_CLs   = wing_CL 
            case_res.aerodynamics.wing_CDs   = wing_CD
            
        # Extract resulst from  strip forces result file
        with open(case.aero_result_filename_3,'r') as aero_res_file_2:
            aero_lines_2     = aero_res_file_2.readlines()
            line_idx         = 0
            header           = 20
            divider_header   = 15    
            
            for i in range(n_wings): 
                for j in range(n_sw):
                    wing_local_span[i,j]      = float(aero_lines_2[header + j + line_idx][8:16].strip())
                    wing_sectional_chord[i,j] = float(aero_lines_2[header + j + line_idx][16:24].strip()) 
                    wing_cl[i,j]              = float(aero_lines_2[header + j + line_idx][61:69].strip())  
                    # At high angle of attacks, AVL does not give an answer 
                    try:
                        alpha_i[i,j]              = float(aero_lines_2[header + j + line_idx][43:51].strip())
                        wing_cd[i,j]              = float(aero_lines_2[header + j + line_idx][70:78].strip())
                    except:
                        alpha_i[i,j]              = 0.
                        wing_cd[i,j]              = 0.
                line_idx = divider_header +  n_sw + line_idx            
            case_res.aerodynamics.wing_local_spans         = wing_local_span
            case_res.aerodynamics.wing_section_chords      = wing_sectional_chord 
            case_res.aerodynamics.wing_section_cls         = wing_cl 
            case_res.aerodynamics.wing_section_aoa_i       = alpha_i 
            case_res.aerodynamics.wing_section_cds         = wing_cd 
  
        with open(case.aero_result_filename_4,'r') as bod_der_vile:
            # Extract results from body axis derivatives file                         
                                                                           
            lines_2                  = bod_der_vile.readlines() 
            case_res.stability.CX_u  = float(lines_2[36+num_ctrl][24:34].strip())
            case_res.stability.CX_v  = float(lines_2[36+num_ctrl][43:54].strip())
            case_res.stability.CX_w  = float(lines_2[36+num_ctrl][65:74].strip())
            case_res.stability.CY_u  = float(lines_2[37+num_ctrl][24:34].strip())
            case_res.stability.CY_v  = float(lines_2[37+num_ctrl][43:54].strip())
            case_res.stability.CY_w  = float(lines_2[37+num_ctrl][65:74].strip())
            case_res.stability.CZ_u  = float(lines_2[38+num_ctrl][24:34].strip())
            case_res.stability.CZ_v  = float(lines_2[38+num_ctrl][43:54].strip())
            case_res.stability.CZ_w  = float(lines_2[38+num_ctrl][65:74].strip())
            case_res.stability.Cl_u  = float(lines_2[39+num_ctrl][24:34].strip())
            case_res.stability.Cl_v  = float(lines_2[39+num_ctrl][43:54].strip())
            case_res.stability.Cl_w  = float(lines_2[39+num_ctrl][65:74].strip())
            case_res.stability.Cm_u  = float(lines_2[40+num_ctrl][24:34].strip())
            case_res.stability.Cm_v  = float(lines_2[40+num_ctrl][43:54].strip())
            case_res.stability.Cm_w  = float(lines_2[40+num_ctrl][65:74].strip())
            case_res.stability.Cn_u  = float(lines_2[41+num_ctrl][24:34].strip())
            case_res.stability.Cn_v  = float(lines_2[41+num_ctrl][43:54].strip())
            case_res.stability.Cn_w  = float(lines_2[41+num_ctrl][65:74].strip())
            
            case_res.stability.CX_p  = float(lines_2[45+num_ctrl][24:34].strip())
            case_res.stability.CX_q  = float(lines_2[45+num_ctrl][43:54].strip())
            case_res.stability.CX_r  = float(lines_2[45+num_ctrl][65:74].strip())
            case_res.stability.CY_p  = float(lines_2[46+num_ctrl][24:34].strip())
            case_res.stability.CY_q  = float(lines_2[46+num_ctrl][43:54].strip())
            case_res.stability.CY_r  = float(lines_2[46+num_ctrl][65:74].strip())
            case_res.stability.CZ_p  = float(lines_2[47+num_ctrl][24:34].strip())
            case_res.stability.CZ_q  = float(lines_2[47+num_ctrl][43:54].strip())
            case_res.stability.CZ_r  = float(lines_2[47+num_ctrl][65:74].strip())
            case_res.stability.Cl_p  = float(lines_2[48+num_ctrl][24:34].strip())
            case_res.stability.Cl_q  = float(lines_2[48+num_ctrl][43:54].strip())
            case_res.stability.Cl_r  = float(lines_2[48+num_ctrl][65:74].strip())
            case_res.stability.Cm_p  = float(lines_2[49+num_ctrl][24:34].strip())
            case_res.stability.Cm_q  = float(lines_2[49+num_ctrl][43:54].strip())
            case_res.stability.Cm_r  = float(lines_2[49+num_ctrl][65:74].strip())
            case_res.stability.Cn_p  = float(lines_2[50+num_ctrl][24:34].strip())
            case_res.stability.Cn_q  = float(lines_2[50+num_ctrl][43:54].strip())
            case_res.stability.Cn_r  = float(lines_2[50+num_ctrl][65:74].strip())
            
                
        results.append(case_res)
            
    return results
