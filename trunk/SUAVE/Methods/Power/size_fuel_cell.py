"""Sizes a fuel cell based on the maximum power requirements of the fuel cell"""
#by M. Vegh


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def size_fuel_cell( fuel_cell,power ): #sizes a simple fuel cell based on the power requirements of the mission
    """
    Inputs:
    power=maximum power the fuel cell is sized for [W]

    fuelcell

    Reads:
    power
    fuelcell.SpecificPower= specific power of the fuel cell component [kW/kg]
    fuelcell.MassDensity =density of the fuel cell [kg/m^3]
    Returns:
    fuelcell.mass =mass of the fuel cell  [kg]
    """
    
    fuel_cell.MaxPower= power                                          #define maximum power output of fuel cell
    fuel_cell.Mass_Props.mass=(power/1000)/fuel_cell.SpecificPower
    fuel_cell.Volume=fuel_cell.Mass_Props.mass/fuel_cell.MassDensity
    
    return
