## @ingroup Methods-Weights-Correlations-Tube_Wing
# tube.py
#
# Created:  Jan 2014, A. Wendorff
# Modified: Feb 2014, A. Wendorff
#           Feb 2016, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units

# ----------------------------------------------------------------------
#   Tube
# ----------------------------------------------------------------------

## @ingroup Methods-Weights-Correlations-Tube_Wing
def tube(vehicle, fuse, wt_wing, wt_propulsion):
    """ Calculate the weight of a fuselage in the state tube and wing configuration
    
    Assumptions:
        fuselage in a standard wing and tube configuration         
    
    Source: 
        N/A 
        
    Inputs:
        fuse.areas.wetted - fuselage wetted area                                                            [meters**2]
        fuse.differential_pressure- Maximum fuselage pressure differential                                  [Pascal]
        fuse.width - width of the fuselage                                                                  [meters]
        fuse.heights.maximum - height of the fuselage                                                       [meters]
        fuse.lengths.total - length of the fuselage                                                         [meters]
        vehicle.envelope.limit_load - limit load factor at zero fuel weight of the aircraft                 [dimensionless]
        vehicle.mass_properties.max_zero_fuel - zero fuel weight of the aircraft                            [kilograms]
        wt_wing - weight of the wing of the aircraft                           [kilograms]
        wt_propulsion - weight of the entire propulsion system of the aircraft                              [kilograms]
        vehicle.wings.main_wing.chords.root - wing root chord                                               [meters]
        
    Outputs:
        weight - weight of the fuselage                                                                     [kilograms]
            
    Properties Used:
        N/A
    """
    # unpack inputs

    diff_p  = fuse.differential_pressure / (Units.force_pound / Units.ft ** 2)  # Convert Pascals to lbs/ square ft
    width   = fuse.width / Units.ft  # Convert meters to ft
    height  = fuse.heights.maximum / Units.ft  # Convert meters to ft

    # setup
    length  = fuse.lengths.total - vehicle.wings.main_wing.chords.root / 2.
    length  = length / Units.ft  # Convert meters to ft
    weight  = (vehicle.mass_properties.max_zero_fuel - wt_wing - wt_propulsion) / Units.lb  # Convert kg to lbs
    area    = fuse.areas.wetted / Units.ft ** 2  # Convert square meters to square ft

    # process

    # Calculate fuselage indices
    I_p = 1.5 * 10 ** -3. * diff_p * width
    I_b = 1.91 * 10 ** -4. * vehicle.envelope.limit_load * weight * length / height ** 2.

    if I_p > I_b:
        I_f = I_p
    else:
        I_f = (I_p ** 2. + I_b ** 2.) / (2. * I_b)

    # Calculate weight of wing for traditional aircraft vertical tail without rudder
    fuselage_weight = ((1.051 + 0.102 * I_f) * area) * Units.lb  # Convert from lbs to kg

    return fuselage_weight
