## @ingroup Components-Energy-Storages-Fuel_Tanks
# Fuel_Tank.py
# 
# Created:  Sep 2018, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
from SUAVE.Components.Energy.Energy_Component import Energy_Component

from jax.tree_util import register_pytree_node_class

# ----------------------------------------------------------------------
#  Fuel Tank
# ----------------------------------------------------------------------    

## @ingroup Components-Energy-Storages-Fuel_Tank
@register_pytree_node_class
class Fuel_Tank(Energy_Component):
    """
    Energy Component object that stores fuel. Contains values
    used to indicate its fuel type.
    """
    def __defaults__(self):
        self.mass_properties.empty_mass            = 0.0
        self.mass_properties.fuel_mass_when_full   = 0.0
        self.mass_properties.fuel_volume_when_full = 0.0
        self.fuel_type                             = None