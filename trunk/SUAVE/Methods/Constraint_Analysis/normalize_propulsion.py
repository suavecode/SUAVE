## @ingroup Methods-Constraint_Analysis
# normalize_propulsion.py
# 
# Created:  Nov 2021, S. Karpuk
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE Imports
import SUAVE
import numpy as np

# ------------------------------------------------------------------------------------
#  Normalize engine power or thrust for the constraint analysis
# ------------------------------------------------------------------------------------

## @ingroup Methods-Constraint_Analysis
def normalize_power_piston(density):
    """Altitude correction for the piston engine

        Assumptions:
        None

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            density         [kg/m**3]

        Outputs:
            power_ratio     [Unitless]

            
        Properties Used:       
        """

    return 1.132*density/1.225-0.132

def normalize_power_electric(density):
    """Altitude correction for the electric engine

        Assumptions:
        None

        Source:
            Lusuardi L, Cavallini A (7-9 Nov 2018) 'The problem of altitude when qualifying the insulating system of actuators for more electrical aircraft',
            2018 IEEE International Conference on Electrical Systems for Aircraft,  Railway,  Ship  Propulsion and Road  Vehicles   International Transportation
            Electrification Conference (ESARS-ITEC) https://doi.org/10.1109/ESARS-ITEC.2018.8607370

        Inputs:
            density         [kg/m**3]

        Outputs:
            power_ratio     [Unitless]

        Properties Used:       
    """

    return 0.50987*density/1.225+0.4981

def normalize_gasturbine_thrust(ca,vehicle,atmo_properties,mach,seg_tag):

    """Altitude correction for engines that feature a gas turbine

        Assumptions:
            N/A

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082
            Clarkson Universitty AE429 Lecture notes (https://people.clarkson.edu/~pmarzocc/AE429/AE-429-6.pdf)


            For the Howe method, an afterburner factor of 1.2 is used
        Inputs:
            ca.analyses.takeoff.runway_elevation      [m]
               engine.throttle_ratio                  [Unitless]
                      afterburner                     [Unitless]
                      method                          [string]
            
            atmo_properties.pressure                  [Pa]
                            temperature               [K]
                            mach                      [Unitless]

        Outputs:
            thrust_ratio           [Unitless]

        Properties Used:       
    """        
    # Unpack inputs
    network_name = list(vehicle.networks.keys())[0]
    altitude     = ca.analyses.takeoff.runway_elevation
    TR           = ca.engine.throttle_ratio
    afterburner  = ca.engine.afterburner
    method       = ca.engine.method
    BPR          = vehicle.networks[network_name].bypass_ratio
    pressure     = atmo_properties.pressure[0,0]
    temperature  = atmo_properties.temperature[0,0]

    # Calculate atmospheric ratios
    theta = temperature/288 * (1+0.2*mach**2)
    delta = pressure/101325 * (1+0.2*mach**2)**3.5 

    Nets  = SUAVE.Components.Energy.Networks  

    for prop in vehicle.networks:     
        if isinstance(prop, Nets.Turbojet_Super):

            thrust_ratio = np.zeros(len(theta))
            for i in range(len(theta)):

                if afterburner == True:
                    if theta[i] <= TR:
                        thrust_ratio[i] = delta[i] * (1 - 0.3 * (theta[i] - 1) - 0.1 * np.sqrt(mach[i]))
                    else:
                        thrust_ratio[i] = delta[i] * (1 - 0.3 * (theta[i] - 1) - 0.1 * np.sqrt(mach[i]) - 1.5 * (theta[i] - TR) / theta[i])
                else:
                    if theta[i] <= TR:
                        thrust_ratio[i] =  delta[i] * 0.8 * (1 - 0.16 * np.sqrt(mach[i]))
                    else:
                        thrust_ratio[i] =  delta[i] * 0.8 * (1 - 0.16 * np.sqrt(mach[i]) - 24 * (theta[i] - TR) / ((9 + mach[i]) * theta[i]))

        elif isinstance(prop, Nets.Turbofan):

            thrust_ratio = np.zeros(len(theta))
            for i in range(len(theta)):

                if method == 'Mattingly':
                    if BPR < 1:
                        if afterburner == True:
                            if theta[i] <= TR:
                                thrust_ratio[i] =  delta[i]
                            else:
                                thrust_ratio[i] = delta[i] * (1 - 3.5 * (theta[i] - TR) / theta[i])    
                        else: 
                            if theta <= TR:
                                thrust_ratio[i] =  0.6 * delta[i]
                            else:
                                thrust_ratio[i] = 0.6 * delta[i] * (1 - 3.8 * (theta[i] - TR) / theta[i])
                    else:
                        if theta[i] <= TR:
                            thrust_ratio[i] =  delta[i] * (1 - 0.49 * np.sqrt(mach[i]))
                        else:
                            thrust_ratio[i] =  delta[i] * (1 - 0.49 * np.sqrt(mach[i]) - 3 * (theta[i] - TR)/(1.5 + mach[i]))
            
                elif method == 'Scholz':
                    if seg_tag != 'takeoff' and seg_tag != 'OEIclimb':
                        thrust_ratio[i] = (0.0013*BPR-0.0397)*altitude/1000-0.0248*BPR+0.7125
                    else:
                        thrust_ratio[i] = 1.0  
            
                elif method == 'Howe':
                    if BPR <= 1:
                        s = 0.8
                        if mach[i] < 0.4:
                            if afterburner == False:
                                K   = [1.0, 0.0, -0.2, 0.07]
                                kab = 1.0
                            else:
                                K   = [1.32, 0.062, -0.13, -0.27]  
                                kab = 1.2                    
                        elif mach[i] < 0.9:
                            if afterburner == False:     
                                K   = [0.856, 0.062, 0.16, -0.23]   
                                kab = 1.0
                            else:
                                K   = [1.17, -0.12, 0.25, -0.17]
                                kab = 1.2
                        elif mach[i] < 2.2:
                            if afterburner == False:  
                                K   = [1.0, -0.145, 0.5, -0.05]
                                kab = 1.0
                            else:
                                K   = [1.4, 0.03, 0.8, 0.4]
                                kab = 1.2
                        else:
                            raise ValueError("the Mach number is too high for the Howe method. The maximum possible value is 2.2\n")  
                
                    elif BPR > 3 and BPR < 6:
                        s = 0.7
                        if afterburner == False:
                            kab = 1.0
                        else:
                            kab = 1.2

                        if mach[i] < 0.4:
                            K = [1.0, 0.0, -0.6, -0.04]
                        elif mach[i] < 0.9:
                            K = [0.88, -0.016, -0.3, 0.0]
                        else:
                            raise ValueError("the Mach number is too high for the Howe method. The maximum possible value is 0.9\n")  
                       
                    else:
                        s = 0.7
                        if afterburner == False:
                            kab = 1.0
                        else:
                            kab = 1.2

                        if mach[i] < 0.4:
                            K = [1.0, 0.0, -0.595, -0.03]
                        elif mach[i] < 0.9:
                            K = [0.89, -0.014, -0.3, 0.005]
                        else:
                            raise ValueError("the Mach number is too high for the Howe method. The maximum possible value is 0.9\n")  
                                                                           
                    sigma        = (1 -2.2558e-5*altitude)**4.2561 
                    thrust_ratio[i] = kab * (K[0]+K[1]*BPR+(K[2]+K[3]*BPR)*mach[i])*sigma**s

                elif method == 'Bartel':
                    p_pSL = (1-2.2558e-5*altitude)**5.2461
                    A     = -0.4327*p_pSL**2+1.3855*p_pSL+0.0472
                    Z     = 0.9106*p_pSL**3-1.7736*p_pSL**2+1.8697*p_pSL
                    X     = 0.1377*p_pSL**3-0.4374*p_pSL**2+1.3003*p_pSL
                    G0    = 0.0603*BPR+0.6337

                    thrust_ratio[i] = A - (0.377*(1+BPR))/(((1+0.82*BPR)*G0)**0.5)*Z*mach[i]+(0.23+0.19*BPR**0.5)*X*mach[i]**2

        else:
            raise ValueError("Enter a correct thrust normalization method\n")  

    return thrust_ratio


def normalize_turboprop_thrust(atmo_properties):
    """Altitude correction for a turboprop engine

        Assumptions:
            N/A

        Source:
            Clarkson Universitty AE429 Lecture notes (https://people.clarkson.edu/~pmarzocc/AE429/AE-429-6.pdf)

        Inputs:
            atmo_properties.pressure                [Pa]
                            temperature             [K]


        Outputs:
            thrust_ratio           [Unitless]

        Properties Used:       
    """          
    # Unpack inputs
    pressure     = atmo_properties.pressure[0,0]
    temperature  = atmo_properties.temperature[0,0]

    density      = pressure/(287*temperature)
    thrust_ratio = (density/1.225)**0.7

    return thrust_ratio


    
