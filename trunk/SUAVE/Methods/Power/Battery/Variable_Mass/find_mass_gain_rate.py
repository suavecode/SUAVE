# find_mass_gain_rate.py
# 
# Created:  ### 2104, M. Vegh
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Find Mass Gain Rate
# ----------------------------------------------------------------------

def find_mass_gain_rate(battery,power):
    """finds the mass gain rate of the battery from the ambient air"""
    
    #weight gain of battery (positive means mass loss)
    mdot = -(power) *(battery.mass_gain_factor)  
                
    return mdot