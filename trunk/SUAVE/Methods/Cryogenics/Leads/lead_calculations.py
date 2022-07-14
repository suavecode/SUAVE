## @ingroup Methods-Cryogenics-Leads
# lead-calculations.py
#
# Created:  Feb 2020, K. Hamilton - Through New Zealand Ministry of Business Innovation and Employment Research Contract RTVU2004 
# Modified: Nov 2021, S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Attributes.Solids.Solid import Solid
from scipy import integrate
from scipy import interpolate
from scipy.misc import derivative
import numpy as np

# ----------------------------------------------------------------------
#  Lead calculation functions
# ----------------------------------------------------------------------
## @ingroup Methods-Cryogenics-Leads
def Q_min(material, cold_temp, hot_temp, current):
    """Calculate minimum electrical power

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
        material
        cold_temp    [K]
        hot_temp     [K]
        current      [A]

    Outputs:
        power        [W]    

    Properties Used:
    N/A
    """    

    # Estimate the area under the thermal:electrical conductivity vs temperature plot for the temperature range of the current lead.
    integral = integrate.quad(lambda T: material.thermal_conductivity(T)/material.electrical_conductivity(T), cold_temp, hot_temp)

    # Estimate the average thermal:electrical conductivity for the lead.
    average_ratio = (1/(hot_temp-cold_temp)) * integral[0]

    # Solve the heat flux at the cold end. This is both the load on the cryocooler and the power loss in the current lead.
    minimum_Q = current * (2*average_ratio*(hot_temp-cold_temp))**0.5

    # This represents the special case where all the electrical power is delivered to the cryogenic environment as this optimised the lead for reduced cryogenic load. Q = electrical power
    power = minimum_Q

    return power

def LARatio( material, cold_temp, hot_temp, current, minimum_Q):
    """Calculate the optimum length to cross-sectional area ratio

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
        material
        cold_temp    [K]
        hot_temp     [K]
        current      [A]
        minimum_Q    [W]

    Outputs:
        la_ratio

    Properties Used:
    N/A
    """  
    # Calculate the optimum length to cross-sectional area ratio
    # Taken directly from McFee
    sigTL = material.electrical_conductivity(cold_temp)
    inte = integrate.quad(lambda T: Q_min(material,T,hot_temp,current)*derivative(material.electrical_conductivity,T), cold_temp, hot_temp)[0]
    la_ratio = (sigTL * minimum_Q + inte)/(current**2)

    return la_ratio

def calc_current(Cryogenic_Lead, current):

    """Estimates the heat flow into the cryogenic environment when a current other than the current the lead was optimised for is flowing. Assumes the temperature difference remains constant.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    Cryogenic_Lead. 
        outputs.optimum_current     [A]
        outputs.minimum_Q           [W]
        outputs.unpowered_Q         [W]
        cold_temp                   [K]
        hot_temp                    [K]
        outputs.cross_section       [m2]
        length                      [m]
        material

    Outputs:
        [lead cooling power, cryogenic loading due to lead]     [w,w]

    Properties Used:
    N/A
    """  

    c_l = Cryogenic_Lead

    design_current      = c_l.outputs.optimum_current
    design_Q            = c_l.outputs.minimum_Q
    zero_Q              = c_l.outputs.unpowered_Q
    cold_temp           = c_l.cold_temp
    hot_temp            = c_l.hot_temp
    cs_area             = c_l.outputs.cross_section
    length              = c_l.length
    material            = c_l.material

    # The thermal gradient along the lead is assumed to remain constant for all currents below the design current. The resistance remains constant if the temperature remains constant. The estimated heat flow is reduced in proportion with the carried current.
    if current <= design_current:
        proportion      = current/design_current
        R               = design_Q/(design_current**2.0)
        power           = R*current**2.0
        Q               = zero_Q + proportion * (design_Q - zero_Q)

    # If the supplied current is higher than the design current the maximum temperature in the lead will be higher than ambient. Solve by dividing the lead at the maximum temperature point.
    else:
        # Initial guess at max temp in lead
        max_temp        = 2 * hot_temp
        # Find actual maximum temperature by bisection, accept result within 1% of correct.
        error           = 1
        guess_over      = 0
        guess_diff      = hot_temp

        while error > 0.01:
            # Find length of warmer part of lead
            warm_Q          = Q_min(material, hot_temp, max_temp, current)

            warm_la         = LARatio(material, hot_temp, max_temp, current, warm_Q)
            warm_length     = cs_area * warm_la
            # Find length of cooler part of lead
            cool_Q          = Q_min(material, cold_temp, max_temp, current)
            cool_la         = LARatio(material, cold_temp, max_temp, current, cool_Q)
            cool_length     = cs_area * cool_la
            # compare lead length with known lead length as test of the max temp guess
            test_length     = warm_length + cool_length
            error           = abs((test_length-length)/length)
            # change the guessed max_temp
            # A max_temp too low will result in the test length being too long
            if test_length > length:
                if guess_over == 0:             # query whether solving by bisection yet
                    guess_diff  = max_temp      # if not, continue to double guess
                    max_temp    = 2*max_temp
                else:
                    max_temp    = max_temp + guess_diff
            else:
                guess_over  = 1              # set flag that bisection range found
                max_temp    = max_temp - guess_diff
            # Prepare guess difference for next iteration
            guess_diff  = 0.5*guess_diff
            # The cool_Q is the cryogenic heat load as warm_Q is sunk to ambient
            Q           = cool_Q
            # All Q is out of the lead, so the electrical power use in the lead is the sum of the Qs
            power       = warm_Q + cool_Q

    return [Q,power]