## @ingroup Methods-Dynamo_Supply-dynamo_supply_mass_estimation
# dynamo_supply_mass_estimation.py
#
# Created:  Feb 2020,   K. Hamilton - Through New Zealand Ministry of Business Innovation and Employment Research Contract RTVU2004 
# Modified: Feb 2022,   S. Claridge 
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# import math
import numpy as np
import scipy.integrate as integrate

# ----------------------------------------------------------------------
#  Dynamo_Supply dynamo_supply_mass_estimation
# ----------------------------------------------------------------------    
    
    
    
def dynamo_supply_mass_estimation(HTS_Dynamo_Supply):
    """ Basic mass estimation for HTS Dynamo supply. This supply includes all elements required to create the required shaft power from supplied electricity, i.e. the esc, brushless motor, and gearbox.
    Assumptions:
        Mass scales linearly with power and current

    Source:
        Maxon Motor drivetrains

    Inputs:
        current             [A]
        power_out           [W]

    Outputs:
        mass                [kg]

    Properties Used:
        None
    """

    # unpack
    rated_power     = HTS_Dynamo_Supply.rated_power

    # Estimate mass of motor and gearbox. Source: Maxon EC-max 12V brushless motors under 100W.
    mass_motor      = 0.013 + 0.0046 * rated_power
    mass_gearbox    = 0.0109 + 0.0015 * rated_power

    # Estimate mass of motor driver (ESC). Source: Estimate
    mass_esc        = (5.0 + rated_power/50.0)/1000.0

    # Sum masses to give total mass
    mass            = mass_esc + mass_motor + mass_gearbox

    # Store results
    HTS_Dynamo_Supply.mass_properties.mass       = mass

    # Return results
    return mass