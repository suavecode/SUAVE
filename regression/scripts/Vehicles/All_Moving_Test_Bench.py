# All_Moving_Test_Bench.py
#
# Created:  Jul 2021 A. Blaufox
# Modified: 

""" setup file for the all-moving surface class test vehicle
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup(deflection_config=None):  
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'All_Moving_Test_Bench'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    vehicle.reference_area = 100
    
    # ------------------------------------------------------------------
    #   Wings
    # ------------------------------------------------------------------    
    stabilator   = make_stabilator()
    v_tail_right = make_all_moving_vert_tail()
    
    vehicle.append_component(stabilator)
    vehicle.append_component(v_tail_right)
    
    #deflection if sepcified-------------------
    if deflection_config is not None:
        #sign_duplicate
        stabilator  .sign_duplicate = deflection_config.  stabilator_sign_duplicate
        v_tail_right.sign_duplicate = deflection_config.v_tail_right_sign_duplicate
        
        #hinge_fraction
        stabilator  .hinge_fraction = deflection_config.  stabilator_hinge_fraction
        v_tail_right.hinge_fraction = deflection_config.v_tail_right_hinge_fraction
        
        #hinge_vector
        stabilator  .use_constant_hinge_fraction = deflection_config.  stabilator_use_constant_hinge_fraction
        v_tail_right.use_constant_hinge_fraction = deflection_config.v_tail_right_use_constant_hinge_fraction
        
        stabilator  .hinge_vector   = deflection_config.  stabilator_hinge_vector
        v_tail_right.hinge_vector   = deflection_config.v_tail_right_hinge_vector        
        
        #deflection
        deflection                  = deflection_config.deflection
        stab_def                    = deflection_config.stab_def
        vt_r_def                    = deflection_config.vt_r_def

        stabilator  .deflection     = (stab_def + deflection) *Units.degrees
        v_tail_right.deflection     = (vt_r_def + deflection) *Units.degrees

    #make left v-tail--------------------------
    v_tail_left  = v_tail_right.make_x_z_reflection()
    vehicle.append_component(v_tail_left)
    
    return vehicle    
    
# ------------------------------------------------------------------
#   Stabilator Construction Function
# ------------------------------------------------------------------ 
def make_stabilator():
    wing = SUAVE.Components.Wings.Stabilator()

    wing.spans.projected         = 7.
    
    wing.chords.root             = 4.23
    
    wing.chords.mean_aerodynamic = 8.0

    wing.areas.reference         = 100
    wing.areas.exposed           = 100    
    wing.areas.wetted            = 100

    wing.origin                  = [[0.,1.,0.]]
    wing.aerodynamic_center      =  [0.,0.,0.]

    wing.dynamic_pressure_ratio  = 0.9    
    
    # Wing Segments
    segment                        = SUAVE.Components.Wings.Segment()
    segment.tag                    = 'root_segment'
    segment.percent_span_location  = 0.0
    segment.twist                  = 0. * Units.deg
    segment.root_chord_percent     = 1.0
    segment.dihedral_outboard      = 4.6 * Units.degrees
    segment.sweeps.quarter_chord   = 28.2250  * Units.degrees 
    segment.thickness_to_chord     = .1
    wing.append_segment(segment)

    segment                        = SUAVE.Components.Wings.Segment()
    segment.tag                    = 'tip_segment'
    segment.percent_span_location  = 1.
    segment.twist                  = 0. * Units.deg
    segment.root_chord_percent     = 0.3333               
    segment.dihedral_outboard      = 0 * Units.degrees
    segment.sweeps.quarter_chord   = 0 * Units.degrees  
    segment.thickness_to_chord     = .1
    wing.append_segment(segment)   
    
    return wing

# ------------------------------------------------------------------
#   Stabilator Construction Function
# ------------------------------------------------------------------ 
def make_all_moving_vert_tail():
    wing = SUAVE.Components.Wings.Vertical_Tail_All_Moving()
    
    wing.spans.projected         = 7.
    
    wing.chords.root             = 4.23
    
    wing.chords.mean_aerodynamic = 8.0

    wing.areas.reference         = 100
    wing.areas.exposed           = 100    
    wing.areas.wetted            = 100

    wing.origin                  = [[7.,1.,0.]]
    wing.aerodynamic_center      =  [7.,0.,0.]

    wing.dynamic_pressure_ratio  = 0.9    
    
    # Wing Segments
    segment                        = SUAVE.Components.Wings.Segment()
    segment.tag                    = 'root_segment'
    segment.percent_span_location  = 0.0
    segment.twist                  = 0. * Units.deg
    segment.root_chord_percent     = 1.0
    segment.dihedral_outboard      = 23.6 * Units.degrees
    segment.sweeps.quarter_chord   = 28.2250  * Units.degrees 
    segment.thickness_to_chord     = .1
    wing.append_segment(segment)

    segment                        = SUAVE.Components.Wings.Segment()
    segment.tag                    = 'tip_segment'
    segment.percent_span_location  = 1.
    segment.twist                  = 0. * Units.deg
    segment.root_chord_percent     = 0.3333               
    segment.dihedral_outboard      = 0 * Units.degrees
    segment.sweeps.quarter_chord   = 0 * Units.degrees  
    segment.thickness_to_chord     = .1
    wing.append_segment(segment)   
    
    return wing
