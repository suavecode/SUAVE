# translate_data.py
# 
# Created:  Mar 2015, T. Momose
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

import SUAVE
from SUAVE.Core import Data, Units
from .Data.Cases import Run_Case
from copy import deepcopy

def translate_conditions_to_cases(avl,conditions):
    """ Takes SUAVE Conditions() data structure and translates to a Container of
        avl Run_Case()s.
    """
    # set up aerodynamic Conditions object
    cases = Run_Case.Container()
    for i in range(len(conditions.aerodynamics.angle_of_attack)):      
        case = Run_Case()
        case.tag  = avl.settings.filenames.case_template.format(avl.current_status.batch_index,i+1)
        case.mass = conditions.weights.total_mass
        case.conditions.freestream.mach     = conditions.freestream.mach_number
        case.conditions.freestream.velocity = conditions.freestream.velocity
        case.conditions.freestream.density  = conditions.freestream.density
        case.conditions.freestream.gravitational_acceleration = conditions.freestream.gravity
        case.conditions.aerodynamics.angle_of_attack = conditions.aerodynamics.angle_of_attack[i]/Units.deg
        case.conditions.aerodynamics.side_slip_angle = 0#conditions.aerodynamics.side_slip_angle[i][0]
        case.stability_and_control.control_deflections = np.array([[]]) # TODO How to do this from the SUAVE side?
        cases.append_case(case)
        
    return cases

def translate_results_to_conditions(cases,results):
    """ Takes avl results structure containing the results of each run case stored
        each in its own Data() object. Translates into the Conditions() data structure.
    """
    # set up aerodynamic Conditions object
    res = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    ones_1col = res.ones_row(1)
    # add missing entries
    res.aerodynamics.roll_moment_coefficient  = ones_1col * 0
    res.aerodynamics.pitch_moment_coefficient = ones_1col * 0
    res.aerodynamics.yaw_moment_coefficient   = ones_1col * 0
    res.aerodynamics.drag_breakdown.induced   = SUAVE.Analyses.Mission.Segments.Conditions.Conditions()
    res.aerodynamics.drag_breakdown.induced.total = ones_1col * 0
    res.aerodynamics.drag_breakdown.induced.efficiency_factor = ones_1col * 0
    res.aerodynamics.cz_alpha                 = ones_1col * 0
    res.aerodynamics.cy_alpha                 = ones_1col * 0
    res.aerodynamics.cl_alpha                 = ones_1col * 0
    res.aerodynamics.cm_alpha                 = ones_1col * 0
    res.aerodynamics.cn_alpha                 = ones_1col * 0
    res.aerodynamics.cz_beta                  = ones_1col * 0
    res.aerodynamics.cy_beta                  = ones_1col * 0
    res.aerodynamics.cl_beta                  = ones_1col * 0
    res.aerodynamics.cm_beta                  = ones_1col * 0
    res.aerodynamics.cn_beta                  = ones_1col * 0
    res.aerodynamics.neutral_point            = ones_1col * 0

    res.expand_rows(len(cases))

    mach_case = results.keys()[0][5:8]   
    for i in xrange(len(results.keys())):
        aoa_case = '{:02d}'.format(i+1)
        tag = 'case_' + mach_case + '_' + aoa_case
        case_res = results[tag]
       
        #res.freestream.velocity[i][0]          = cases[i].conditions.freestream.velocity
        #res.freestream.mach_number[i][0]       = cases[i].conditions.freestream.mach
        #res.freestream.gravity[i][0]           = cases[i].conditions.freestream.gravitational_acceleration
        #res.freestream.density[i][0]           = cases[i].conditions.freestream.density
        #res.aerodynamics.angle_of_attack[i][0] = cases[i].conditions.aerodynamics.angle_of_attack * Units.deg
        #res.aerodynamics.side_slip_angle[i][0] = cases[i].conditions.aerodynamics.side_slip_angle * Units.deg      
        #res.weights.total_mass[i][0]           = cases[i].mass

        res.aerodynamics.roll_moment_coefficient[i][0] = case_res.aerodynamics.roll_moment_coefficient
        res.aerodynamics.pitch_moment_coefficient[i][0] = case_res.aerodynamics.pitch_moment_coefficient
        res.aerodynamics.yaw_moment_coefficient[i][0] = case_res.aerodynamics.yaw_moment_coefficient
        res.aerodynamics.lift_coefficient[i][0] = case_res.aerodynamics.total_lift_coefficient
        res.aerodynamics.drag_breakdown.induced.total[i][0] = case_res.aerodynamics.induced_drag_coefficient
        res.aerodynamics.drag_breakdown.induced.efficiency_factor[i][0] = case_res.aerodynamics.span_efficiency_factor
        res.aerodynamics.cz_alpha[i][0] = -case_res.stability.alpha_derivatives.lift_curve_slope
        res.aerodynamics.cy_alpha[i][0] = case_res.stability.alpha_derivatives.side_force_derivative
        res.aerodynamics.cl_alpha[i][0] = case_res.stability.alpha_derivatives.roll_moment_derivative
        res.aerodynamics.cm_alpha[i][0] = case_res.stability.alpha_derivatives.pitch_moment_derivative
        res.aerodynamics.cn_alpha[i][0] = case_res.stability.alpha_derivatives.yaw_moment_derivative
        res.aerodynamics.cz_beta[i][0] = -case_res.stability.beta_derivatives.lift_coefficient_derivative
        res.aerodynamics.cy_beta[i][0] = case_res.stability.beta_derivatives.side_force_derivative
        res.aerodynamics.cl_beta[i][0] = case_res.stability.beta_derivatives.roll_moment_derivative
        res.aerodynamics.cm_beta[i][0] = case_res.stability.beta_derivatives.pitch_moment_derivative
        res.aerodynamics.cn_beta[i][0] = case_res.stability.beta_derivatives.yaw_moment_derivative
        res.aerodynamics.neutral_point[i][0] = case_res.stability.neutral_point

    return res