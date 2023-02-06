## @ingroup Components-Energy-Converters
# Motor_HTS_Rotor.py
#
# Created:  Feb 2020,    K. Hamilton - Through New Zealand Ministry of Business Innovation and Employment Research Contract RTVU2004
# Modified: Nov 2021,   S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC imports
import MARC

# package imports
from MARC.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  HTS Rotor Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Motor_HTS_Rotor(Energy_Component):
    """This represents just the rotor of a HTS motor, i.e. the superconducting components.
    This is used to estimate the power and cooling required for the HTS components.
    The power used here could be considered the same as that used by the motor as inefficiency, however many publications consider the overall motor efficiency, i.e. including cryocooler and/or motor drive electronics.
    
    Assumptions:
    No ACLoss in the HTS,
    HTS is operated within the Ic and Tc limits.
    i.e the power used by the coil is only due to solder resistances only, and is not affected by the motor output power or speed.

    Source:
    None
    """      
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """              
        self.temperature         =   0.0     # Temperature inside of the rotor           [K]
        self.skin_temp           = 300.0     # Temperature of the outside of the rotor   [K]
        self.current             =   0.0     # HTS coil current.                         [A]
        self.resistance          =   0.0     # Resistance of the HTS oils.               [ohm]
        self.length              =   0.0     # Physical size of rotor exterior           [m]
        self.diameter            =   0.0     # Physical size of rotor exterior           [m]
        self.surface_area        =   0.0     # Surface area of the rotor.                [m2]
        self.R_value             = 125.0     # R_Value of the cryostat wall.             [K.m2/W]
        self.number_of_engines   =   2.0     # Number of rotors on the vehicle
        self.inputs.hts_current  =   0.0     #[A]
        self.inputs.ambient_temp =   0.0     #[K]

        self.outputs.coil_power     = 0.0    #[W]
        self.outputs.coil_voltage   = 0.0
        self.outputs.cryo_load      = 0.0    #[W]
    
    def power(self, conditions):
        """ Calculates the electrical power draw from the HTS coils, and the total heating load on the HTS rotor cryostat.

        Assumptions:
        No ACLoss in the HTS,
        HTS is operated within the Ic and Tc limits.
        i.e the power used by the coil is only due to solder resistances only.

        Source:
        N/A

        Inputs:
        self.inputs
            current             [A]
            ambient_temp        [K]

        Outputs:
        self.outputs
            coil_power          [W]
            cryogenic_load      [W]
            coil_voltage        [V]

        Properties Used:
        self.
            temperature     [K]
            resistance      [ohm]
            surface_area    [m2]
            R_value         [K.m2/W]

        """

        # unpack
        cryo_temp       = self.temperature
        coil_R          = self.resistance
        surface_area    = self.surface_area
        r_value         = self.R_value

        current         = self.inputs.hts_current
        ambient_temp    = self.inputs.ambient_temp


        # Calculate HTS coil power
        # This is both the electrical power required to operate the coil, and the thermal load imparted by the coil into the cryostat.
        coil_power      = coil_R * current**2
        coil_voltage    = current*coil_R

        # Estimate heating from external heat conducting through the cryostat wall.
        # Given the non-cryogenic armature coils are usually very close to this external wall it is likely the temperature at the wall is above ambient.
        Q = (ambient_temp - cryo_temp)/(r_value/surface_area)

        # Sum the heat loads to give total rotor heat load.
        cryo_load = coil_power + Q

        # Store the outputs.
        self.outputs.coil_power     = coil_power
        self.outputs.coil_voltage   = coil_voltage
        self.outputs.cryo_load      = cryo_load

        return coil_power