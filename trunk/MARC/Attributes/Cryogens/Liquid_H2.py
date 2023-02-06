## @ingroup Attributes-Cryogens
# Liquid H2
#
# Created:  Feb 2020,  K. Hamilton - Through New Zealand Ministry of Business Innovation and Employment Research Contract RTVU2004

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from .Cryogen import Cryogen

# ----------------------------------------------------------------------
#  Liquid H2 Cryogen Class
# ----------------------------------------------------------------------
## @ingroup Attributes-Cryogens
class Liquid_H2(Cryogen):
    """Holds values for this cryogen
    
    Assumptions:
    None
    
    Source:
    Ekin - Experimental Techniques for Low Temperature Measurements, ISBN 0-19-857054-6
    NIST Chemistry Webbook
    """

    def __defaults__(self):
        """This sets the default values.

        Assumptions:
        Ambient Pressure

        Source:

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """ 
        
        self.tag                        = 'Liquid_H2'
        self.density                    =    59.9            # [kg/m^3] 
        self.specific_energy            =   141.86e6         # [J/kg] 
        self.energy_density             =  8491.0e6          # [J/m^3]
        self.temperatures.freeze        =    13.99           # [K]
        self.temperatures.boiling       =    20.271          # [K]
        self.vaporization_enthalpy      =   461.             # [kJ/kg]
        self.specific_heat              =    10.67           # [kJ/kgK]

        # Coefficiencts for polynomial fit of Liquid Specific Heat Capacity (C_P) curve.
        # C_P = CP_C3*T^3 + CP_C2*T^2 + CP_C1*T^1 + CP_C0*T^0 where C_P is Specific Heat Capacity (J/gK) T is temperature (kelvin).
        # Data from NIST Chemistry Webbook. Pressure is 1.295MPa.
        self.LCP_C0                     =   -31.2
        self.LCP_C1                     =     5.56
        self.LCP_C2                     =    -0.272
        self.LCP_C3                     =     4.76E-03
        # Range for which this polynomial fit holds
        self.LCP_minT                   =    15.0              # [K]
        self.LCP_maxT                   =    30.0              # [K]

        # Coefficiencts for polynomial fit of Gas Specific Heat Capacity (C_P) curve.
        # C_P = CP_C3*T^3 + CP_C2*T^2 + CP_C1*T^1 + CP_C0*T^0 where C_P is Specific Heat Capacity (J/gK) T is temperature (kelvin).
        # Data from NIST Chemistry Webbook. Pressure is 0.01 MPa
        self.GCP_C0                     =   10.3
        self.GCP_C1                     =   -7.39E-03
        self.GCP_C2                     =    0.221E-03
        self.GCP_C3                     =   -0.516E-06
        # Range for which this polynomial fit holds
        self.GCP_minT                   =   20.0              # [K]
        self.GCP_maxT                   =  300.0              # [K]

        # Antoine Equation Coefficients for calculatating the evaporation temperature.
        # log10(P) = A - (B/(T+C)) where P is vapour pressure (Pa) and T temperature (kelvin).
        # Data from NIST Chemistry Webbook, coefficients converted so as to use pressure in Pa.
        self.antoine_A                  =    8.54314
        self.antoine_B                  =   99.395
        self.antoine_C                  =    7.726
        # Range for which Antoine Equation is referenced
        self.antoine_minT               =   21.01             # [K]
        self.antoine_maxT               =   32.27             # [K]

        # Coefficiencts for polynomial fit of vapourisation enthalpy
        # ΔH = H_C3*P^3 + H_C2*P^2 + H_C1*P^1 + H_C0*P^0 where ΔH is vapourisation enthalpy (kJ/kg), P is pressure (Pa).
        # Data from NIST Chemistry Webbook -1.00E-16	5.23E-11	-0.000176	464
        self.H_C0                       =   464.
        self.H_C1                       =    -0.000176
        self.H_C2                       =    52.3E-12
        self.H_C3                       =  -100.00E-18
        # Range for which this polynomial fit holds
        self.H_minP                     =     0.02E6         # [Pa]
        self.H_maxP                     =     1.20E6         # [Pa]

