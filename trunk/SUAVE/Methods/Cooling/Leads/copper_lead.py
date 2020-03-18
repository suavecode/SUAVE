## @ingroup methods-cooling-Leads

## @ingroup Methods-cooling-leads
# copper_lead.py
# 
# Created:  Feb 2020, K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Attributes.Solids import Copper
from scipy import integrate
from scipy import interpolate
from scipy.misc import derivative


# ----------------------------------------------------------------------
#  Copper Lead
# ----------------------------------------------------------------------
## @ingroup Methods-Cooling-Leads
def initialize_copper_lead(lead):
    """
    Defines an optimum copper lead for supplying current to a cryogenic environment given the operating conditions and copper properties.
    
    Assumptions:
    None
    
    Inputs:
        lead.
            cold_temp           [K]
            hot_temp            [K]
            current             [A]
            length              [m]

    Outputs:      
        lead.     
            mass                [kg]
            cross_section       [m]
            optimum_current     [A]
            minimum_Q           [W]
    """

    # Unpack properties
    cold_temp   = lead.cold_temp
    hot_temp    = lead.hot_temp 
    current     = lead.current  
    length      = lead.length   
    
    # Instatiate the copper material
    copper = Copper()

    # Find the heat generated by the optimum lead
    minimum_Q = Q_min(copper, cold_temp, hot_temp, current)[0]

    # # Calculate the optimum length to cross-sectional area ratio
    la_ratio = LARatio(copper, cold_temp, hot_temp, current, minimum_Q)

    # Calculate the cross-sectional area
    cs_area = length/la_ratio
    # Apply the copper density to calculate the mass
    mass = cs_area*length*copper.density

    # 

    # Pack up results
    lead.mass               = mass
    lead.cross_section      = cs_area
    lead.optimum_current    = current
    lead.minimum_Q          = minimum_Q
    lead.material           = copper

    # find the heat conducted into the cryogenic environment if no current is flowing
    unpowered_Q             = Q_unpowered(lead)[0]

    # Pack up unpowered lead
    lead.unpowered_Q        = unpowered_Q


def LARatio(copper, cold_temp, hot_temp, current, minimum_Q):
    # Calculate the optimum length to cross-sectional area ratio
    # Taken directly from McFee
    sigTL = copper.electrical_conductivity(cold_temp)
    inte = integrate.quad(lambda T: Q_min(copper,T,hot_temp,current)[0]*derivative(copper.electrical_conductivity,T), cold_temp, hot_temp)[0]
    la_ratio = (sigTL * minimum_Q + inte)/(current**2)
    return la_ratio


def Q_min(copper, cold_temp, hot_temp, current):
    # Estimate the area under the thermal:electrical conductivity vs temperature plot for the temperature range of the current lead.
    integral = integrate.quad(lambda T: copper.thermal_conductivity(T)/copper.electrical_conductivity(T), cold_temp, hot_temp)

    # Estimate the average thermal:electrical conductivity for the lead.
    average_ratio = (1/(hot_temp-cold_temp)) * integral[0]

    # Solve the heat flux at the cold end. This is both the load on the cryocooler and the power loss in the current lead.
    minimum_Q = current * (2*average_ratio*(hot_temp-cold_temp))**0.5
    # This represents the special case where all the electrical power is delivered to the cryogenic environment as this optimised the lead for reduced cryogenic load. Q = electrical power
    power = minimum_Q

    return [minimum_Q, power]

def Q_unpowered(lead):
    # Estimates the heat flow into the cryogenic environment if no current is supplied to the lead.

    # unpack properties
    hot_temp        = lead.hot_temp     
    cold_temp       = lead.cold_temp    
    cross_section   = lead.cross_section
    length          = lead.length
    copper          = lead.material

    # Integrate the thermal conductivity across the relevant temperature range.
    integral = integrate.quad(lambda T: copper.thermal_conductivity(T), cold_temp, hot_temp)

    # Apply the conductivity to estimate the heat flow
    Q       = integral[0] * cross_section / length
    # Electrical power is obviously zero if no current is flowing
    power   = 0.0

    return [Q, power]

def Q_offdesign(lead, current):
    # Estimates the heat flow into the cryogenic environment when a current other than the current the lead was optimised for is flowing. Assumes the temperature difference remains constant.

    # unpack properties
    design_current      = lead.optimum_current
    design_Q            = lead.minimum_Q
    zero_Q              = lead.unpowered_Q
    cold_temp           = lead.cold_temp
    hot_temp            = lead.hot_temp
    cs_area             = lead.cross_section
    length              = lead.length
    copper              = lead.material

    # If the current is lower than the design current the heat flow will drop proportional to the supplied current. I.e. the heat from conduction remains the same while the joule heating in the lead reduces.
    if current <= design_current:
        proportion      = current/design_current
        power           = (design_Q-zero_Q)*proportion
        Q               = zero_Q + power

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
            warm_Q          = Q_min(copper, hot_temp, max_temp, current)
            warm_la         = LARatio(copper, hot_temp, max_temp, current, warm_Q)
            warm_length     = cs_area * warm_la
            # Find length of cooler part of lead
            cool_Q          = Q_min(copper, cold_temp, max_temp, current)
            cool_la         = LARatio(copper, cold_temp, max_temp, current, cool_Q)
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
    
    return [Q, power]