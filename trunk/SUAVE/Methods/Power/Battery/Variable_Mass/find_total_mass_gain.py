# find_total_mass_gain.py
# 
# Created:  ### 2104, M. Vegh
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Find Total Mass Gain
# ----------------------------------------------------------------------

def find_total_mass_gain(battery):
    
    mgain=battery.max_energy*battery.mass_gain_factor
    
    return mgain