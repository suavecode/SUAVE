## @ingroup Methods-Constraint_Analysis
# compute_climb_constraint.py
#
# Created:  Nov 2021, S. Karpuk, 
# Modified: 



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
import SUAVE
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Helper_Functions.oswald_efficiency  import oswald_efficiency                   as  oswald_efficiency
from SUAVE.Methods.Constraint_Analysis.compressibility_drag_constraint                   import compressibility_drag_constraint     as  compressibility_drag
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_turboprop_thrust
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_piston
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_power_electric
from SUAVE.Methods.Constraint_Analysis.normalize_propulsion                              import normalize_gasturbine_thrust

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Compute climb constraint
# ----------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def compute_climb_constraint(constraint_analysis):
    
    """Calculate thrust-to-weight ratios for the steady climb

        Assumptions:
            N/A

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            self.engine.type                    string
                 aerodynamics.cd_min_clean      [Unitless]
                 climb.altitude                 [m]
                       airspeed                 [m/s]
                       climb_rate               [m/s]
                       delta_ISA                [K]
                 geometry.aspect_ratio          [Unitless]
                 wing_loading                   [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:
    """  

    # Unpack inputs
    eng_type  = constraint_analysis.engine.type
    cd_min    = constraint_analysis.aerodynamics.cd_min_clean 
    altitude  = constraint_analysis.climb.altitude   
    Vx        = constraint_analysis.climb.airspeed   
    Vy        = constraint_analysis.climb.climb_rate 
    delta_ISA = constraint_analysis.climb.delta_ISA
    AR        = constraint_analysis.geometry.aspect_ratio 
    W_S       = constraint_analysis.wing_loading  
  

    # Determine atmospheric properties at the altitude
    planet      = SUAVE.Analyses.Planets.Planet()
    atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features

    atmo_values = atmosphere.compute_values(altitude,delta_ISA)
    rho         = atmo_values.density[0,0]
    a           = atmo_values.speed_of_sound[0,0]   

    T_W = np.zeros(len(W_S))
    if eng_type != 'turbofan' and eng_type != 'turbojet':
        P_W        = np.zeros(len(W_S))
        etap       = constraint_analysis.propeller.climb_efficiency
        if etap == 0:
            raise ValueError('Warning: Set the propeller efficiency during climb')

        for i in range(len(W_S)):
            error      = 1
            tollerance = 1e-6
            M          = 0.5                            # Initial Mach number
            q          = 0.5*rho*(M*a)**2               # Initial dynamic pressure
            # Iterate the best propeller climb speed estimation until converged 
            while abs(error) > tollerance:
                CL = W_S[i]/q

                # Calculate compressibility_drag
                cd_comp   = compressibility_drag(M,CL,constraint_analysis.geometry) 
                cd_comp_e = compressibility_drag(M,0,constraint_analysis.geometry)  
                cd0     = cd_min + cd_comp  

                # Calculate Oswald efficiency
                if constraint_analysis.aerodynamics.oswald_factor == 0:
                    e = oswald_efficiency(constraint_analysis,cd_min+cd_comp_e)

                k     = 1/(np.pi*e*AR)
                Vx    = np.sqrt(2/rho*W_S[i]*np.sqrt(k/(3*cd0)))
                Mnew  = Vx/a
                error = Mnew - M
                M     = Mnew
                q     = 0.5*rho*(M*a)**2 

            T_W[i] = Vy/Vx + q/W_S[i]*cd0 + k/q*W_S[i]
            P_W[i] = T_W[i]*Vx/etap

            if eng_type == 'turboprop':
                P_W[i] = P_W[i] / normalize_turboprop_thrust(atmo_values) 
            elif eng_type == 'piston':
                P_W[i] = P_W[i] / normalize_power_piston(rho)   
            elif eng_type == 'electric air-cooled':
                P_W[i] = P_W[i] / normalize_power_electric(rho)  
            elif eng_type == 'electric liquid-cooled':
                pass 

    else:
        M  = Vx/a                           
        q  = 0.5*rho*Vx**2      

        for i in range(len(W_S)):
            CL = W_S[i]/q

            # Calculate compressibility_drag
            cd_comp   = compressibility_drag(M,CL,constraint_analysis.geometry) 
            cd_comp_e = compressibility_drag(M,0,constraint_analysis.geometry)  
            cd0       = cd_min + cd_comp  

            # Calculate Oswald efficiency
            if constraint_analysis.aerodynamics.oswald_factor == 0:
                e = oswald_efficiency(constraint_analysis,cd_min+cd_comp_e)

            k      = 1/(np.pi*e*AR)
            T_W[i] = Vy/Vx + q/W_S[i]*cd0 + k/q*W_S[i]
            T_W[i] = T_W[i] / normalize_gasturbine_thrust(constraint_analysis,atmo_values,M,'climb')  

    # Pack outputs
    if eng_type != 'turbofan' and eng_type != 'turbojet':
        constraint = P_W         # convert to W/N
    else:
        constraint = T_W

    return constraint

