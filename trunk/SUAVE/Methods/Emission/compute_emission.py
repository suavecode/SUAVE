## @ingroup Methods-Emission
# compute_emission.py
#
# Created:  

# suave imports
import numpy as np
from SUAVE.Core import Units,Data

# ----------------------------------------------------------------------
#  Compute Emission
# ----------------------------------------------------------------------

## @ingroup Methods-Emission
def compute_emission(vehicle,mission):
    """ This function calculated Emission level for a given aircraft

        Assumptions:
            Constant relative humidity of 60% along the complete mission 

    
        Source:

            J. B. Caers, 'Conditions   for   Passenger   Aircraft   Minimum   Fuel Consumption, 
                          Direct  Operating  Costs  and  Environmental Impact'

            Terrance G. Tritz, Stephen C. HendersonDavid C. PickettSteven L. Baughcum,
                         'Scheduled civil aircraft emission inventories for 1992: Database
                          development and analysis'

            D. Scholz, 'Limits to Principles of Electric Flight'
 
        Inputs:
            vehicle, mission   
    
        Outputs:
            Equivalent CO2 mass,  mCO2eq
    
        Properties Used:
            N/A
        """    
    

    # Constants
    #---------------------------------------------------------------------------
    Phi = 0.6                                       # Relative humidity
    
    EI_CO2_combustion   = 3.16                      # direct combustion emission (kg CO2 / kg fuel)
    EI_CO2_production   = 0.5                       # production emission (kg CO2 / kg fuel)

    SGTP_CO2     = 3.58e-14                         # K/kg CO2
    SGTP_O3s     = 7.97e-12                         # K/kg NOx
    SGTP_O3l     = -9.14e-13                        # K/kg NOx
    SGTP_CH4     = -3.9e-12                         # K/kg NOx
    SGTP_contr   = 1.37e-13                         # K/km      
    SGTP_cirrus  = 4.12e-13                         # K/km      

    s_O3s_ref = np.array([[4876.8,	5334.6096,	5938.7232,	6552.2856,	7156.3992,	7780.02,	8383.8288,	8997.696,	9601.8096,	10205.9232,	10838.9928,	11452.86,	12066.7272,	12660.7824],
                          [0.469417,	0.469417,	0.55761,	0.620199,	0.711238,	0.711238,	0.813656,	0.930299,	1.00996,	1.13229,	1.42816,	1.62447,	1.8037,	1.93172]])
    s_O3l_ref = np.array([[4876.8,	5324.856,	5938.7232,	6552.2856,	7175.9064,	7780.02,	8393.5824,	8997.696,	9611.5632,	10215.6768,	10838.9928,	11443.1064,	12066.7272,	12670.536],
                          [0.86771,	0.86771,	0.924609,	0.955903,	0.961593,	0.944523,	0.927454,	0.927454,	0.941679,	0.975818,	1.14083,	1.21479,	1.20341,	1.20341]])
    s_CH4_ref = np.array([[4876.8,	5324.856,	5938.7232,	6552.2856,	7175.9064,	7780.02,	8393.5824,	8997.696,	9611.5632,	10215.6768,	10838.9928,	11443.1064,	12066.7272,	12670.536],
                          [0.86771,	0.86771,	0.924609,	0.955903,	0.961593,	0.944523,	0.927454,	0.927454,	0.941679,	0.975818,	1.14083,	1.21479,	1.20341,	1.20341]]) 
    s_AIC_ref = np.array([[4876.8,	5324.94744,	5958.19992,	6562.25256,	7166.27472,	7780.05048,	8384.07264,	8978.43264,	9631.13136,	10225.39992,	10829.45256,	11443.22832,	12057.00408,	12661.02624],
                          [0.0284495,	0.0284495,	0,	0,	0.173542,	0.395448,	0.799431,	1.25178,	1.70982,	2.10526,	1.82077,	1.53343,	0.967283,	0.793741]]) 


    k_RFI  = 4.7                                                                                                # the range [1.9 4.7] from Scholz
    kPEF_A = np.array([-3.1164e-9, 3.7595e-5, -1.8897e-1, 5.0657e2, -7.6385e5, 6.1428e8, -2.0583e11])


    # Unpack inputs
    #-----------------------------------------------------------------------------
    nPAX   = vehicle.passengers


    EI_CO2 = EI_CO2_combustion + EI_CO2_production              # Combined CO2 emission

    if (vehicle.propulsion_type == 'cryo-electric' or vehicle.propulsion_type == 'electric' or vehicle.propulsion_type == 'propeller electric' or vehicle.propulsion_type == 'propeller cryo-electric') :

        # Unpack inputs
        mbat  = vehicle.mass_properties.battery
        Cbat  = vehicle.propulsors.network.battery.specific_energy              # in J/kg
        Cfuel = vehicle.emission.reference_fuel.specific_energy                 # in J/kg
        year  = vehicle.emission.reference_year
        Xff   = vehicle.emission.fossil_fuel_share

        kPEF   = kPEF_A[0]*year**6 + kPEF_A[1]*year**5 + kPEF_A[2]*year**4 + kPEF_A[3]*year**3 + kPEF_A[4]*year**2 + kPEF_A[5]*year + kPEF_A[6] 

        mCO2eq_EM = EI_CO2*Xff*kPEF*mbat*Cbat/(0.9*Cfuel)
        mCO2eq    = mCO2eq_EM

        # compute equivalent emisson using a low-fidelity model (for comparison and validation)
        mCO2eq_fuel_LoFi = 0.0

    elif (vehicle.propulsion_type == 'battery-hybrid turbofan' or vehicle.propulsion_type == 'battery-hybrid propeller'):

        # Calculate the battery contribution
        mbat  = vehicle.mass_properties.battery
        Cbat  = vehicle.propulsors.network.battery.specific_energy              # in J/kg
        Cfuel = vehicle.emission.reference_fuel.specific_energy                 # in J/kg
        year  = vehicle.emission.reference_year
        Xff   = vehicle.emission.fossil_fuel_share

        kPEF   = kPEF_A[0]*year**6 + kPEF_A[1]*year**5 + kPEF_A[2]*year**4 + kPEF_A[3]*year**3 + kPEF_A[4]*year**2 + kPEF_A[5]*year + kPEF_A[6] 

        mCO2eq_EM = EI_CO2*Xff*kPEF*mbat*Cbat/(0.9*Cfuel)  

        # Calculate the fuel contribution
        mCO2eq_fuel = 0
        for i in range(Nsegm ):
            dTotalFuel     = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in kg
            dRange         = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in km
            dTime          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in s 
            f              = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in kg/km
            fuel_flow_rate = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in kg/s 
            s_O3s          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1) 
            s_O3l          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1) 
            s_CH4          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)   
            s_AIC          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)   
            CF_AIC         = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     
            CF_NOx         = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)
            EI_NOx         = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1) 

            emission_CO2  = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)
            emission_NOx  = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)
            emission_AIC  = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)
            

            # freestream data
            T     = mission.segments[i].conditions.freestream.temperature
            P     = mission.segments[i].conditions.freestream.pressure
            M     = mission.segments[i].conditions.freestream.mach_number 
            delta = P / 101325
            theta = T / 288.15

            for j in range(len(dTotalFuel)):
               dTotalFuel[j]     = np.abs(mission.segments[i].conditions.weights.total_mass[j] - mission.segments[i].conditions.weights.total_mass[j+1])                                            # Absolute value for RENOx safety 
               dRange[j]         = (mission.segments[i].conditions.frames.inertial.position_vector[j+1,0] - mission.segments[i].conditions.frames.inertial.position_vector[j,0]) / Units.km
               dTime[j]          = (mission.segments[i].conditions.frames.inertial.time[j+1,0] - mission.segments[i].conditions.frames.inertial.time[j,0]) 
               H1                = -mission.segments[i].conditions.frames.inertial.position_vector[j,2]
               H2                = -mission.segments[i].conditions.frames.inertial.position_vector[j+1,2]
               f[j]              = dTotalFuel[j]/dRange[j]
               fuel_flow_rate[j] = dTotalFuel[j]/(dTime[j] / Units.hour )

               # Compute forcing factors 
               s_O3s[j] = 0.5 * (np.interp(H1, s_O3s_ref[0,:], s_O3s_ref[1,:]) + np.interp(H2, s_O3s_ref[0,:], s_O3s_ref[1,:]))
               s_O3l[j] = 0.5 * (np.interp(H1, s_O3l_ref[0,:], s_O3l_ref[1,:]) + np.interp(H2, s_O3l_ref[0,:], s_O3l_ref[1,:]))
               s_CH4[j] = 0.5 * (np.interp(H1, s_CH4_ref[0,:], s_CH4_ref[1,:]) + np.interp(H2, s_CH4_ref[0,:], s_CH4_ref[1,:]))
               s_AIC[j] = 0.5 * (np.interp(H1, s_AIC_ref[0,:], s_AIC_ref[1,:]) + np.interp(H2, s_AIC_ref[0,:], s_AIC_ref[1,:])) 

               # Compute characterization factors
               CF_NOx[j] = SGTP_O3s/SGTP_CO2*s_O3s[j] + SGTP_O3l/SGTP_CO2*s_O3l[j] + SGTP_CH4/SGTP_CO2*s_CH4[j]
               CF_AIC[j] = SGTP_contr/SGTP_CO2*s_AIC[j] + SGTP_cirrus/SGTP_CO2*s_AIC[j] 

               # Compute EI_NOx
               Wff   = fuel_flow_rate[j] / delta[j] * (theta[j])**3.8 * np.exp(0.2*M[j]**2) * 2.205          # in lbm/hr
               RENOx = 0.0218*Wff**0.7268 / 1000                                                                     # Reference EI (lbm/1000lbm)

               k_T      = 373.16/T[j]
               beta     = 7.90298*(1-k_T) + 3.00571 + 5.02808*np.log(k_T) + \
                          1.3816e-7*(1-10**(11.344*(1-1/k_T)))+8.1328e-3*(10**(3.49149*(1-k_T))-1)
               Pv       = 0.014504*10**beta                                                                     # Saturated vapor pressure (psi)
               Pamb_psi = 0.000145038 * P[j]                                                                    # Ambient pressure (psi)
               omega    = (0.62198*Phi*Pv)/(Pamb_psi - Phi*Pv)                                                  # Specific humidity

               EI_NOx[j]    = RENOx * np.exp(-19*(omega-0.0063)) * np.sqrt(delta[j]**1.02 / theta[j]**3.3) 
               mCO2eq_fuel += (EI_CO2*f[j] + EI_NOx[j]*f[j]*CF_NOx[j] + CF_AIC[j])* dRange[j]

               emission_CO2[j] = EI_CO2*f[j]* dRange[j]
               emission_NOx[j] = EI_NOx[j]*f[j]*CF_NOx[j]* dRange[j]
               emission_AIC[j] = CF_AIC[j]* dRange[j]


            # Pack segment outputs
            mission.segments[i].emission.CO2 = emission_CO2
            mission.segments[i].emission.NOx = emission_NOx
            mission.segments[i].emission.AIC = emission_AIC

        mCO2eq = mCO2eq_fuel + mCO2eq_EM 

        # compute equivalent emisson using a low-fidelity model (for comparison and validation)
        Wf               = mission.segments[0].conditions.weights.total_mass[0] - mission.segments[-1].conditions.weights.total_mass[-1] 
        mCO2eq_fuel_LoFi = EI_CO2 * 1.1 * Wf * (k_RFI + 0.1)

    else:

        # Calculate the fuel contribution
        mCO2eq_EM      = 0
        mCO2eq_fuel    = 0
        Nsegm          = len(mission.segments)
        for i in range(Nsegm):

            mission.segments[i].emission     = Data() 
            mission.segments[i].emission.CO2 = Data()
            mission.segments[i].emission.NOx = Data()
            mission.segments[i].emission.AIC = Data()
            
            dTotalFuel     = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in kg
            dRange         = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in km
            dTime          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in s 
            f              = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in kg/km
            fuel_flow_rate = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     # in kg/s 
            s_O3s          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1) 
            s_O3l          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1) 
            s_CH4          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)   
            s_AIC          = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)   
            CF_AIC         = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)     
            CF_NOx         = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)
            EI_NOx         = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1) 
            
            emission_CO2  = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)
            emission_NOx  = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)
            emission_AIC  = np.zeros(len(mission.segments[i].conditions.weights.total_mass)-1)


            # freestream data
            T     = mission.segments[i].conditions.freestream.temperature
            P     = mission.segments[i].conditions.freestream.pressure
            M     = mission.segments[i].conditions.freestream.mach_number 
            delta = P / 101325
            theta = T / 288.15

            for j in range(len(dTotalFuel)):
               dTotalFuel[j]     = np.abs(mission.segments[i].conditions.weights.total_mass[j] - mission.segments[i].conditions.weights.total_mass[j+1])                                            # Absolute value for RENOx safety 
               dRange[j]         = (mission.segments[i].conditions.frames.inertial.position_vector[j+1,0] - mission.segments[i].conditions.frames.inertial.position_vector[j,0]) / Units.km
               dTime[j]          = (mission.segments[i].conditions.frames.inertial.time[j+1,0] - mission.segments[i].conditions.frames.inertial.time[j,0]) 
               H1                = -mission.segments[i].conditions.frames.inertial.position_vector[j,2]
               H2                = -mission.segments[i].conditions.frames.inertial.position_vector[j+1,2]
               f[j]              = dTotalFuel[j]/dRange[j]
               fuel_flow_rate[j] = dTotalFuel[j]/(dTime[j] / Units.hour )

               # Compute forcing factors 
               s_O3s[j] = 0.5 * (np.interp(H1, s_O3s_ref[0,:], s_O3s_ref[1,:]) + np.interp(H2, s_O3s_ref[0,:], s_O3s_ref[1,:]))
               s_O3l[j] = 0.5 * (np.interp(H1, s_O3l_ref[0,:], s_O3l_ref[1,:]) + np.interp(H2, s_O3l_ref[0,:], s_O3l_ref[1,:]))
               s_CH4[j] = 0.5 * (np.interp(H1, s_CH4_ref[0,:], s_CH4_ref[1,:]) + np.interp(H2, s_CH4_ref[0,:], s_CH4_ref[1,:]))
               s_AIC[j] = 0.5 * (np.interp(H1, s_AIC_ref[0,:], s_AIC_ref[1,:]) + np.interp(H2, s_AIC_ref[0,:], s_AIC_ref[1,:])) 

               # Compute characterization factors
               CF_NOx[j] = SGTP_O3s/SGTP_CO2*s_O3s[j] + SGTP_O3l/SGTP_CO2*s_O3l[j] + SGTP_CH4/SGTP_CO2*s_CH4[j]
               CF_AIC[j] = SGTP_contr/SGTP_CO2*s_AIC[j] + SGTP_cirrus/SGTP_CO2*s_AIC[j] 

               # Compute EI_NOx
               Wff   = fuel_flow_rate[j] / delta[j] * (theta[j])**3.8 * np.exp(0.2*M[j]**2) * 2.205          # in lbm/hr
               RENOx = 0.0218*Wff**0.7268 / 1000                                                                     # Reference EI (lbm/1000lbm)

               k_T      = 373.16/T[j]
               beta     = 7.90298*(1-k_T) + 3.00571 + 5.02808*np.log(k_T) + \
                          1.3816e-7*(1-10**(11.344*(1-1/k_T)))+8.1328e-3*(10**(3.49149*(1-k_T))-1)
               Pv       = 0.014504*10**beta                                                                     # Saturated vapor pressure (psi)
               Pamb_psi = 0.000145038 * P[j]                                                                    # Ambient pressure (psi)
               omega    = (0.62198*Phi*Pv)/(Pamb_psi - Phi*Pv)                                                  # Specific humidity

               EI_NOx[j]    = RENOx * np.exp(-19*(omega-0.0063)) * np.sqrt(delta[j]**1.02 / theta[j]**3.3) 
               mCO2eq_fuel += (EI_CO2*f[j] + EI_NOx[j]*f[j]*CF_NOx[j] + CF_AIC[j])* dRange[j]

               emission_CO2[j] = EI_CO2*f[j]* dRange[j]
               emission_NOx[j] = EI_NOx[j]*f[j]*CF_NOx[j]* dRange[j]
               emission_AIC[j] = CF_AIC[j]* dRange[j]


            # Pack segment outputs
            mission.segments[i].emission.CO2 = emission_CO2
            mission.segments[i].emission.NOx = emission_NOx
            mission.segments[i].emission.AIC = emission_AIC


        mCO2eq = mCO2eq_fuel + mCO2eq_EM 

        # compute equivalent emisson using a low-fidelity model (for comparison and validation)
        Wf               = mission.segments[0].conditions.weights.total_mass[0] - mission.segments[-1].conditions.weights.total_mass[-1] 
        mCO2eq_fuel_LoFi = EI_CO2 * 1.1 * Wf * (k_RFI + 0.1)


    # Pack outputs
    Emission = Data()
    Emission.electric_motor_equivalent   = mCO2eq_EM 
    Emission.gas_turbine_equivalent      = mCO2eq_fuel
    Emission.gas_turbine_equivalent_LoFi = mCO2eq_fuel_LoFi[0]
    Emission.Total_equivalent            = mCO2eq


   
    return Emission 
