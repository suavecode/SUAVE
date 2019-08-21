## @ingroup Methods-Aerodynamics-AVL
#translate_data.py
# 
# Created:  Mar 2015, T. Momose
# Modified: Jan 2016, E. Botero
#           Apr 2017, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

import SUAVE
from SUAVE.Core import Data, Units
from .Data.Cases import Run_Case

## @ingroup Methods-Aerodynamics-AVL
def translate_conditions_to_cases(avl,aircraft,conditions):
    """ Takes SUAVE Conditions() data structure and translates to a Container of
    avl Run_Case()s.

    Assumptions:
        None
        
    Source:
        Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avl

    Inputs:
        conditions.aerodynamics.angle_of_attack  [radians] 
        conditions.freestream.mach_number        [-]
        conditions.freestream.density            [kilograms per meters**3]
        conditions.freestream.gravity            [meters per second**2]

    Outputs:
        cases                                    [data structur]

    Properties Used:
        N/A
    """    
    # set up aerodynamic Conditions object
    cases = Run_Case.Container()
    for i in range(len(conditions.aerodynamics.angle_of_attack)):      
        case                                                  = Run_Case()
        case.tag                                              = avl.settings.filenames.case_template.format(avl.current_status.batch_index,i+1)
        case.mass                                             = conditions.weights.total_mass
        case.conditions.freestream.mach                       = conditions.freestream.mach_number
        case.conditions.freestream.density                    = conditions.freestream.density
        case.conditions.freestream.gravitational_acceleration = conditions.freestream.gravity      
        case.conditions.aerodynamics.angle_of_attack          = conditions.aerodynamics.angle_of_attack[i]/Units.deg
        case.conditions.aerodynamics.side_slip_angle          = conditions.aerodynamics.side_slip_angle  
        
        # determine the number of wings 
        n_wings = 0 
        for wing in aircraft.wings:
            n_wings += 1
            if wing.symmetric == True:
                n_wings += 1                
        case.num_wings                                        = n_wings
        case.n_sw                                             = avl.settings.spanwise_vortices  
                
        cases.append_case(case)
    
    return cases

def translate_results_to_conditions(cases,results,Eigen_Modes):
    """ Takes avl results structure containing the results of each run case stored
        each in its own Data() object. Translates into the Conditions() data structure.

    Assumptions:
        None
        
    Source:
        Drela, M. and Youngren, H., AVL, http://web.mit.edu/drela/Public/web/avl

    Inputs:
        case_res = results 
            
     Outputs:
        cases                        [data_structure]
   
    Properties Used:
        N/A
    """   
   
    num_wings = cases[0].num_wings 
    n_sw      = cases[0].n_sw
    dim       = len(cases)
        
    # set up aerodynamic Conditions object
    res                             = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics() 
    res.stability.static            = Data()
    res.stability.dynamic           = Data()
    
    # add missing entries 
    res.S_ref                                                = np.zeros((dim,1))
    res.c_ref                                                = np.zeros_like(res.S_ref)
    res.b_ref                                                = np.zeros_like(res.S_ref)
    res.X_ref                                                = np.zeros_like(res.S_ref)
    res.Y_ref                                                = np.zeros_like(res.S_ref)
    res.Z_ref                                                = np.zeros_like(res.S_ref)   
    res.aerodynamics.AoA                                     = np.zeros_like(res.S_ref)
    res.aerodynamics.CX                                      = np.zeros_like(res.S_ref)
    res.aerodynamics.CY                                      = np.zeros_like(res.S_ref)
    res.aerodynamics.CZ                                      = np.zeros_like(res.S_ref)                                                   
    res.aerodynamics.Cltot                                   = np.zeros_like(res.S_ref)
    res.aerodynamics.Cmtot                                   = np.zeros_like(res.S_ref) 
    res.aerodynamics.Cntot                                   = np.zeros_like(res.S_ref)    
    res.aerodynamics.roll_moment_coefficient                 = np.zeros_like(res.S_ref)
    res.aerodynamics.pitch_moment_coefficient                = np.zeros_like(res.S_ref)
    res.aerodynamics.yaw_moment_coefficient                  = np.zeros_like(res.S_ref)  
    res.aerodynamics.lift_coefficient                        = np.zeros_like(res.S_ref)
    res.aerodynamics.drag_breakdown.induced                  = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
    res.aerodynamics.drag_breakdown.induced.total            = np.zeros_like(res.S_ref)
    res.aerodynamics.drag_breakdown.induced.efficiency_factor= np.zeros_like(res.S_ref)
    
    res.stability.static.CL_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.Cy_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.Cl_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.Cm_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.Cn_alpha                            = np.zeros_like(res.S_ref)
    res.stability.static.CL_beta                             = np.zeros_like(res.S_ref)
    res.stability.static.Cy_beta                             = np.zeros_like(res.S_ref)
    res.stability.static.Cl_beta                             = np.zeros_like(res.S_ref)
    res.stability.static.Cm_beta                             = np.zeros_like(res.S_ref)
    res.stability.static.Cn_beta                             = np.zeros_like(res.S_ref)    
    res.stability.static.CL_p                                = np.zeros_like(res.S_ref)
    res.stability.static.CL_q                                = np.zeros_like(res.S_ref)
    res.stability.static.CL_r                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_p                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_q                                = np.zeros_like(res.S_ref)
    res.stability.static.CY_r                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_p                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_q                                = np.zeros_like(res.S_ref)
    res.stability.static.Cl_r                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_p                                = np.zeros_like(res.S_ref)
    res.stability.static.Cm_q                                = np.zeros_like(res.S_ref) 
    res.stability.static.Cm_r                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_p                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_q                                = np.zeros_like(res.S_ref)
    res.stability.static.Cn_r                                = np.zeros_like(res.S_ref)  
    
    #static stability
    res.stability.static.neutral_point                       = np.zeros_like(res.S_ref)    
    res.stability.static.spiral_stability_condition          = np.zeros_like(res.S_ref)    

    # dynamic stability
    res.stability.dynamic.roll_mode                 = np.zeros_like(res.S_ref)
    res.stability.dynamic.spiral_mode               = np.zeros_like(res.S_ref)           
    res.stability.dynamic.dutch_roll_mode_1_real    = np.zeros_like(res.S_ref)
    res.stability.dynamic.dutch_roll_mode_1_imag    = np.zeros_like(res.S_ref)
    res.stability.dynamic.dutch_roll_mode_2_real    = np.zeros_like(res.S_ref) 
    res.stability.dynamic.dutch_roll_mode_2_imag    = np.zeros_like(res.S_ref) 
    res.stability.dynamic.short_period_mode_1_real = np.zeros_like(res.S_ref)
    res.stability.dynamic.short_period_mode_1_imag = np.zeros_like(res.S_ref)
    res.stability.dynamic.short_period_mode_2_real = np.zeros_like(res.S_ref)
    res.stability.dynamic.short_period_mode_2_imag = np.zeros_like(res.S_ref)
    res.stability.dynamic.phugoid_mode_1_real      = np.zeros_like(res.S_ref)
    res.stability.dynamic.phugoid_mode_1_imag      = np.zeros_like(res.S_ref)
    res.stability.dynamic.phugoid_mode_2_real      = np.zeros_like(res.S_ref)
    res.stability.dynamic.phugoid_mode_2_imag      = np.zeros_like(res.S_ref) 
    
    # aero results 1: total surface forces and coefficeints 
    res.aerodynamics.wing_areas                    = np.zeros((dim,num_wings)) 
    res.aerodynamics.wing_CLs                      = np.zeros_like(res.aerodynamics.wing_areas) 
    res.aerodynamics.wing_CDs                      = np.zeros_like(res.aerodynamics.wing_areas) 

    # aero results 2 : sectional forces and coefficients 
    res.aerodynamics.wing_local_spans              = np.zeros((dim,num_wings,n_sw))
    res.aerodynamics.wing_section_chords           = np.zeros_like(res.aerodynamics.wing_local_spans)
    res.aerodynamics.wing_section_cls              = np.zeros_like(res.aerodynamics.wing_local_spans)
    res.aerodynamics.wing_section_cds              = np.zeros_like(res.aerodynamics.wing_local_spans)
    
    mach_case = list(results.keys())[0][5:8]   
    for i in range(len(results.keys())):
        aoa_case = '{:02d}'.format(i+1)
        tag = 'case_' + mach_case + '_' + aoa_case
        case_res = results[tag]       
        
        # stability file 
        res.S_ref[i][0]                                     = case_res.S_ref 
        res.c_ref[i][0]                                     = case_res.c_ref 
        res.b_ref[i][0]                                     = case_res.b_ref
        res.X_ref[i][0]                                     = case_res.X_ref 
        res.Y_ref[i][0]                                     = case_res.Y_ref 
        res.Z_ref[i][0]                                     = case_res.Z_ref       
        res.aerodynamics.AoA[i][0]                          = case_res.aerodynamics.AoA
        res.aerodynamics.CX[i][0]                           = case_res.aerodynamics.CX 
        res.aerodynamics.CY[i][0]                           = case_res.aerodynamics.CY  
        res.aerodynamics.CZ[i][0]                           = case_res.aerodynamics.CZ    
        res.aerodynamics.Cltot[i][0]                        = case_res.aerodynamics.Cltot 
        res.aerodynamics.Cmtot[i][0]                        = case_res.aerodynamics.Cmtot 
        res.aerodynamics.Cntot[i][0]                        = case_res.aerodynamics.Cntot      
        res.aerodynamics.roll_moment_coefficient[i][0]      = case_res.aerodynamics.roll_moment_coefficient
        res.aerodynamics.pitch_moment_coefficient[i][0]     = case_res.aerodynamics.pitch_moment_coefficient
        res.aerodynamics.yaw_moment_coefficient[i][0]       = case_res.aerodynamics.yaw_moment_coefficient
        res.aerodynamics.lift_coefficient[i][0]             = case_res.aerodynamics.total_lift_coefficient
        res.aerodynamics.drag_breakdown.induced.total[i][0] = case_res.aerodynamics.induced_drag_coefficient
        res.aerodynamics.drag_breakdown.induced.efficiency_factor[i][0]  = case_res.aerodynamics.span_efficiency_factor 
        res.stability.static.CL_alpha[i][0]                 = case_res.stability.alpha_derivatives.lift_curve_slope
        res.stability.static.Cy_alpha[i][0]                 = case_res.stability.alpha_derivatives.side_force_derivative
        res.stability.static.Cl_alpha[i][0]                 = case_res.stability.alpha_derivatives.roll_moment_derivative
        res.stability.static.Cm_alpha[i][0]                 = case_res.stability.alpha_derivatives.pitch_moment_derivative
        res.stability.static.Cn_alpha[i][0]                 = case_res.stability.alpha_derivatives.yaw_moment_derivative
        res.stability.static.CL_beta[i][0]                  = case_res.stability.beta_derivatives.lift_coefficient_derivative
        res.stability.static.Cy_beta[i][0]                  = case_res.stability.beta_derivatives.side_force_derivative
        res.stability.static.Cl_beta[i][0]                  = case_res.stability.beta_derivatives.roll_moment_derivative
        res.stability.static.Cm_beta[i][0]                  = case_res.stability.beta_derivatives.pitch_moment_derivative
        res.stability.static.Cn_beta[i][0]                  = case_res.stability.beta_derivatives.yaw_moment_derivative        
        res.stability.static.CL_p[i][0]                     = case_res.stability.CL_p 
        res.stability.static.CL_q[i][0]                     = case_res.stability.CL_q
        res.stability.static.CL_r[i][0]                     = case_res.stability.CL_r 
        res.stability.static.CY_p[i][0]                     = case_res.stability.CY_p 
        res.stability.static.CY_q[i][0]                     = case_res.stability.CY_q 
        res.stability.static.CY_r[i][0]                     = case_res.stability.CY_r
        res.stability.static.Cl_p[i][0]                     = case_res.stability.Cl_p 
        res.stability.static.Cl_q[i][0]                     = case_res.stability.Cl_q 
        res.stability.static.Cl_r[i][0]                     = case_res.stability.Cl_r 
        res.stability.static.Cm_p[i][0]                     = case_res.stability.Cm_p 
        res.stability.static.Cm_q[i][0]                     = case_res.stability.Cm_q 
        res.stability.static.Cm_r[i][0]                     = case_res.stability.Cm_r 
        res.stability.static.Cn_p[i][0]                     = case_res.stability.Cn_p 
        res.stability.static.Cn_q[i][0]                     = case_res.stability.Cn_q 
        res.stability.static.Cn_r[i][0]                     = case_res.stability.Cn_r         
        res.stability.static.neutral_point[i][0]            = case_res.stability.neutral_point
        
        # aero surface forces file 
        res.aerodynamics.wing_areas[i][:]                   = case_res.aerodynamics.wing_areas   
        res.aerodynamics.wing_CLs[i][:]                     = case_res.aerodynamics.wing_CLs    
        res.aerodynamics.wing_CDs[i][:]                     = case_res.aerodynamics.wing_CDs    
        
        # aero sectional forces file
        res.aerodynamics.wing_local_spans[i][:]             = case_res.aerodynamics.wing_local_spans
        res.aerodynamics.wing_section_chords[i][:]          = case_res.aerodynamics.wing_section_chords  
        res.aerodynamics.wing_section_cls[i][:]             = case_res.aerodynamics.wing_section_cls    
        res.aerodynamics.wing_section_cds [i][:]            = case_res.aerodynamics.wing_section_cds    
        
        # eigen mode results 
        if Eigen_Modes:
            # store eigen modes (poles of root locus)
            res.stability.dynamic.roll_mode[i][0]                 = case_res.stability.roll_mode_real        
            res.stability.dynamic.spiral_mode[i][0]               = case_res.stability.spiral_mode_real       
            res.stability.dynamic.dutch_roll_mode_1_real[i][0]    = case_res.stability.dutch_roll_mode_1_real
            res.stability.dynamic.dutch_roll_mode_1_imag[i][0]    = case_res.stability.dutch_roll_mode_1_imag 
            res.stability.dynamic.dutch_roll_mode_2_real[i][0]    = case_res.stability.dutch_roll_mode_2_real 
            res.stability.dynamic.dutch_roll_mode_2_imag[i][0]    = case_res.stability.dutch_roll_mode_2_imag 
            res.stability.dynamic.short_period_mode_1_real[i][0] = case_res.stability.short_period_mode_1_real 
            res.stability.dynamic.short_period_mode_1_imag[i][0] = case_res.stability.short_period_mode_1_imag
            res.stability.dynamic.short_period_mode_2_real[i][0] = case_res.stability.short_period_mode_2_real 
            res.stability.dynamic.short_period_mode_2_imag[i][0] = case_res.stability.short_period_mode_2_imag      
            res.stability.dynamic.phugoid_mode_1_real[i][0]      = case_res.stability.phugoid_mode_1_real
            res.stability.dynamic.phugoid_mode_1_imag[i][0]      = case_res.stability.phugoid_mode_1_imag
            res.stability.dynamic.phugoid_mode_2_real[i][0]      = case_res.stability.phugoid_mode_2_real 
            res.stability.dynamic.phugoid_mode_2_imag[i][0]      = case_res.stability.phugoid_mode_2_imag        

            # Set up A matrix       
            ctrl_surfs     = case_res.stability.A_B_matrix_headers[13:]
            A_matrix       = np.zeros((12,12))
            A_matrix[0,:]  = np.array(case_res.stability.A_B_matrix_1[:12])      
            A_matrix[1,:]  = np.array(case_res.stability.A_B_matrix_2[:12])      
            A_matrix[2,:]  = np.array(case_res.stability.A_B_matrix_3[:12])      
            A_matrix[3,:]  = np.array(case_res.stability.A_B_matrix_4[:12])      
            A_matrix[4,:]  = np.array(case_res.stability.A_B_matrix_5[:12])      
            A_matrix[5,:]  = np.array(case_res.stability.A_B_matrix_6[:12])      
            A_matrix[6,:]  = np.array(case_res.stability.A_B_matrix_7[:12])      
            A_matrix[7,:]  = np.array(case_res.stability.A_B_matrix_8[:12])      
            A_matrix[8,:]  = np.array(case_res.stability.A_B_matrix_9[:12])      
            A_matrix[9,:]  = np.array(case_res.stability.A_B_matrix_10[:12])      
            A_matrix[10,:] = np.array(case_res.stability.A_B_matrix_11[:12])      
            A_matrix[11,:] = np.array(case_res.stability.A_B_matrix_12[:12])        
            
            # Set up B matrix 
            B_matrix = np.zeros((12,len(ctrl_surfs)))
            B_matrix[0,:]  = np.array(case_res.stability.A_B_matrix_1[12:])      
            B_matrix[1,:]  = np.array(case_res.stability.A_B_matrix_2[12:])      
            B_matrix[2,:]  = np.array(case_res.stability.A_B_matrix_3[12:])      
            B_matrix[3,:]  = np.array(case_res.stability.A_B_matrix_4[12:])      
            B_matrix[4,:]  = np.array(case_res.stability.A_B_matrix_5[12:])      
            B_matrix[5,:]  = np.array(case_res.stability.A_B_matrix_6[12:])      
            B_matrix[6,:]  = np.array(case_res.stability.A_B_matrix_7[12:])      
            B_matrix[7,:]  = np.array(case_res.stability.A_B_matrix_8[12:])      
            B_matrix[8,:]  = np.array(case_res.stability.A_B_matrix_9[12:])      
            B_matrix[9,:]  = np.array(case_res.stability.A_B_matrix_10[12:])      
            B_matrix[10,:] = np.array(case_res.stability.A_B_matrix_11[12:])      
            B_matrix[11,:] = np.array(case_res.stability.A_B_matrix_12[12:]) 
            
            # Store results of A and B matrices          
            res.stability.dynamic.A_matrix_LongModes = A_matrix[np.ix_([0,1,2,3,8,10],[0,1,2,3,8,10])]
            res.stability.dynamic.A_matrix_LatModes   = A_matrix[np.ix_([4,5,6,7,9,11],[4,5,6,7,9,11])]
            res.stability.dynamic.A_matrix                     = A_matrix 
            res.stability.dynamic.B_matrix_LongModes  = B_matrix[np.ix_([0,1,2,3,8,10],cases[0].LongMode_CS_idxs )]
            res.stability.dynamic.B_matrix_LatModes  = B_matrix[np.ix_([4,5,6,7,9,11],cases[0].LatMode_CS_idxs)]
            res.stability.dynamic.B_matrix                     = B_matrix
            
    return res