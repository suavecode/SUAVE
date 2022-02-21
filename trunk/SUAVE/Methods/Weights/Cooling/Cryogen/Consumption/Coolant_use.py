## @ingroup Methods-Cooling-Cryogen-Consumption
# Coolant_use.py
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
#  Cryogen Coolant_use
# ----------------------------------------------------------------------

## @ingroup Methods-Cooling-cryogen-Consumption
def Coolant_use(cryogen,cryogen_temp,equipment_temp,cooling_required,pressure):
    """ Calculate the mass flow rate of cryogen required to maintain cryogenic equipment temperature

    Assumptions:
    constant pressure cryogen system
    perfect mixing and thermal conduction in cryogen
    
    Inputs:
        cryogen
        cryogen_temp                [K]
        equipment_temp              [K]
        cooling_power               [W]
        pressure                    [Pa]
    
    Outputs:
        mdot                        [kg/s]
    
    """
    
    # Method:
    # Given the cryogen pressure find the boiling temperature using the Antoine Equation.
    # T = B/(A-log10(P))-C
    boil_temp = (cryogen.antoine_B / (cryogen.antoine_A - np.log10(pressure))) - cryogen.antoine_C

    # Define cooling type flags and initial values
    Liquid_Cooled       = False
    Evap_Cooled         = False
    Gas_Cooled          = False
    liq_cooling         = 0.0
    gas_cooling         = 0.0
    evap_cooling        = 0.0

    # Calculate cooling type based on boiling temp and required temp
    # Then set the limits of each cooling type based on whether boiling occurs
    if boil_temp < equipment_temp:
        Gas_Cooled      = True
        gas_Ti          = cryogen_temp
        gas_Tf          = equipment_temp
    if boil_temp > cryogen_temp:
        Liquid_Cooled   = True
        liq_Ti          = cryogen_temp
        liq_Tf          = equipment_temp
    if Gas_Cooled and Liquid_Cooled:
        Evap_Cooled     = True
        liq_Tf          = boil_temp
        gas_Ti          = boil_temp

    # Calculate the cooling power per gram of coolant for each of the cooling modes
    # LIQUID COOLING
    # find the area under C_P vs temperature between the temperatures over which the cryogen is a liquid, as this will be the cooling available (in Joules) per gram, aka watts per gram per second.
    if Liquid_Cooled:
        liq_cooling = integrate.quad(lambda t: (cryogen.LCP_C3*t**3. + cryogen.LCP_C2*t**2. + cryogen.LCP_C1*t**1. + cryogen.LCP_C0*t**0.), liq_Ti, liq_Tf)[0]

    # GAS COOLING
    # find the area under C_P vs temperature between the temperatures over which the cryogen is a vapour, as this will be the cooling available (in Joules) per gram, aka watts per gram per second.
    if Gas_Cooled:
        gas_cooling = integrate.quad(lambda t: (cryogen.GCP_C3*t**3. + cryogen.GCP_C2*t**2. + cryogen.GCP_C1*t**1. + cryogen.GCP_C0*t**0.), gas_Ti, gas_Tf)[0]

    # EVAPORATION COOLING
    # Calculate the enthalpy using the polynomial fit to pressure. Enthalpy is in kJ/kg, i.e. J/g.
    if Evap_Cooled:
        evap_cooling = cryogen.H_C3*pressure**3. + cryogen.H_C2*pressure**2. + cryogen.H_C1*pressure**1. + cryogen.H_C0*pressure**0.

    # Sum the components of cooling to give the total cooling power per gram. X1000 to give per kg.
    cooling_power = 1000. * (liq_cooling + evap_cooling + gas_cooling)

    # Divide the cooling power by the per kg cooling power to calculate the coolant mass flow rate    
    mdot = cooling_required/cooling_power       # [kg/s]    

    # Return mass flow rate of the cryogen         
    return mdot