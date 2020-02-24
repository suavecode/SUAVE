## @ingroup methods-cooling-Leads

## @ingroup Methods-cooling-leads
# initialize_copper_leads.py
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
#  Initialize Copper Leads
# ----------------------------------------------------------------------
## @ingroup Methods-Cooling-Leads
def initialize_copper_leads(lead):
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
    minimum_Q = Qmin(copper, cold_temp, hot_temp, current)

    # Calculate the optimum length to cross-sectional area ratio
    sigTL = copper.electrical_conductivity(cold_temp)
    inte = integrate.quad(lambda T: Qmin(copper,T,hot_temp,current)*derivative(copper.electrical_conductivity,T), cold_temp, hot_temp)
    la_ratio = (sigTL * minimum_Q - inte)/(current**2)

    # Calculate the cross-sectional area
    cs_area = length/la_ratio
    # Apply the copper density to calculate the mass
    mass = cs_area*length*copper.density

    # Pack up results
    lead.mass               = mass
    lead.cross_section      = cs_area
    lead.optimum_current    = current
    lead.minimum_Q          = minimum_Q

    # find the heat conducted into the cryogenic environment if no current is flowing
    unpowered_Q             = Q_unpowered(lead)

    # Pack up unpowered lead
    lead.unpowered_Q        = unpowered_Q



def Qmin(copper, cold_temp, hot_temp, current):
    # Estimate the area under the thermal:electrical conductivity vs temperature plot for the temperature range of the current lead.
    integral = integrate.quad(lambda T: copper.thermal_conductivity(T)/copper.electrical_conductivity(T), cold_temp, hot_temp)

    # Estimate the average thermal:electrical conductivity for the lead.
    average_ratio = (1/(hot_temp-cold_temp)) * integral[0]

    # Solve the heat flux at the cold end. This is both the load on the cryocooler and the power loss in the current lead.
    minimum_Q = current * (2*average_ratio*(hot_temp-cold_temp))**0.5

    return minimum_Q

def Q_unpowered(lead):
    # Estimates the heat flow into the cryogenic environment if no current is supplied to the lead.

    # unpack properties
    hot_temp        = lead.hot_temp     
    cold_temp       = lead.cold_temp    
    cross_section   = lead.cross_section
    length          = lead.length
    copper          = Copper()

    # Sum the thermal conductivity across the relevant temperature range, then divide by temperature difference to get the average conductivity.
    average_conductivity = integrate.quad(lambda T: copper.thermal_conductivity(T), cold_temp, hot_temp)/(hot_temp-cold_temp)

    # Apply the average conductivity to estimate the heat flow
    Q = average_conductivity * (hot_temp-cold_temp) * cross_section / length

    return Q