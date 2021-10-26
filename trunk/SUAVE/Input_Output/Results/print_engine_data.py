## @ingroup Input_Output-Results
# print_engine_data.py

# Created:  SUAVE team
# Modified: Aug 2016, L. Kulik
#           Oct 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import SUAVE
from SUAVE.Core import Units, Data

import time  # importing library
import datetime  # importing library


# ----------------------------------------------------------------------
#  Print output file with compressibility drag
# ----------------------------------------------------------------------
## @ingroup Input_Output-Results
def print_engine_data(vehicle, filename='engine_data.dat', units="imperial"):
    """This creates a file showing engine information.

    Assumptions:
    One network (can be multiple engines) with 'turbofan' tag.

    Source:
    N/A

    Inputs:
    vehicle.
      tag
      turbofan()              Function to compute thrust and fuel burn rate
      networks.turbofan.
        design_thrust         [N]
        engine_length         [m]
        thrust.bypass_ratio   [-] 
    filename (optional)       <string> Determines the name of the saved file
    units (optional)          <string> Determines the type of units used in the output, options are imperial and si

    Outputs:
    filename                  Saved file with name as above

    Properties Used:
    N/A
    """       
    imperial = False
    SI = False

    if units.lower() == "imperial":
        imperial = True
    elif units.lower() == "si":
        SI = True
    else:
        print("Incorrect system of units selected - choose 'imperial' or 'SI'")
        return

    d_isa_vec = [0, 10]
    if imperial:
        speed_vec = np.linspace(0, 450, 10) * Units.knots
        hp_vec = np.linspace(0, 50000, 11) * Units.ft

    elif SI:
        speed_vec = np.linspace(0, 225, 10) * Units['m/s']
        hp_vec = np.linspace(0, 15000, 11) * Units.m

    speed_vec[0] = 1.

    thrust = np.zeros_like(speed_vec)
    mdot = np.zeros_like(speed_vec)

    # Determining vehicle number of engines
    engine_number = 0.
    for network in vehicle.networks:  # may have than one network
        engine_number += network.number_of_engines
    if engine_number == 0:
        raise ValueError("No engine found in the vehicle") 

    # Considering planet and atmosphere of 1st mission segment
    sea_level_gravity = SUAVE.Attributes.Planets.Earth().sea_level_gravity
    atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()  # mission.segments[0].atmosphere

    atmo_values     = atmo.compute_values(0.,0.)
    
    p0   = atmo_values.pressure
    T0   = atmo_values.temperature
    rho0 = atmo_values.density
    a0   = atmo_values.speed_of_sound
    mu0  = atmo_values.dynamic_viscosity    

    state = Data()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    state.numerics = SUAVE.Analyses.Mission.Segments.Conditions.Numerics()

    # write header of file
    fid = open(filename, 'w')  # Open output file
    fid.write('Output file with engine data \n\n')
    fid.write('  VEHICLE TAG : ' + vehicle.tag + '\n\n')

    for d_isa in d_isa_vec:
        fid.write('\n DELTA ISA: {:3.0f} degC'.format(d_isa) + '\n')

        if imperial:
            fid.write('\n' + 8 * ' ' + '|' + 40 * ' ' + ' SPEED [KTAS]' + 48 * ' ' + '|' + 27 * ' ' + ' SPEED [KTAS]' + 33 * ' ')
            fid.write('\n' + 8 * ' ' + '|')
            fid.write(str(np.transpose(list(map('{:10.0f}'.format, speed_vec / Units.knots)))))
            fid.write(' |' + 3 * ' ')
            fid.write(str(np.transpose(list(map('{:6.0f} '.format, speed_vec / Units.knots)))))
            fid.write( '\n HP[ft] |' + 40 * ' ' + ' THRUST [lbf]' + 48 * ' ' + '|' + 30 * ' ' + ' SFC [adm]' + 33 * ' ' + '|\n')

        elif SI:
            fid.write('\n' + 8 * ' ' + '|' + 40 * ' ' + ' SPEED [m/s]' + 48 * ' ' + '|' + 27 * ' ' + ' SPEED [m/s]' + 33 * ' ')
            fid.write('\n' + 8 * ' ' + '|')
            fid.write(str(np.transpose(list(map('{:10.0f}'.format, speed_vec / Units['m/s'])))))
            fid.write(' |' + 3 * ' ')
            fid.write(str(np.transpose(list(map('{:6.0f} '.format, speed_vec / Units['m/s'])))))
            fid.write('\n HP[m] |' + 40 * ' ' + ' THRUST [N]' + 48 * ' ' + '|' + 30 * ' ' + ' SFC * 10^6 [kg/Ns]' + 33 * ' ' + '|\n')

        for altitude in hp_vec:
            for idx, speed in enumerate(speed_vec):
                # Computing atmospheric conditions      
                atmo_values     = atmo.compute_values(altitude,0)
                
                p   = atmo_values.pressure
                T   = atmo_values.temperature
                rho = atmo_values.density
                a   = atmo_values.speed_of_sound
                mu  = atmo_values.dynamic_viscosity                
                
                T_delta_ISA = T + d_isa
                sigma_disa = (p / p0) / (T_delta_ISA / T0)
                rho_delta_ISA = rho0 * sigma_disa
                a_delta_ISA = atmo.fluid_properties.compute_speed_of_sound(T_delta_ISA)

                # Getting engine thrust
                state.conditions.freestream.dynamic_pressure = np.array(
                    np.atleast_1d(0.5 * rho_delta_ISA * speed * speed))
                state.conditions.freestream.gravity = np.array(np.atleast_1d(sea_level_gravity))
                state.conditions.freestream.velocity = np.array(np.atleast_1d(speed))
                state.conditions.freestream.mach_number = np.array(np.atleast_1d(speed / a_delta_ISA))
                state.conditions.freestream.temperature = np.array(np.atleast_1d(T_delta_ISA))
                state.conditions.freestream.pressure = np.array(np.atleast_1d(p))
                state.conditions.propulsion.throttle = np.array(np.atleast_1d(1.))

                results = vehicle.networks.turbofan(state)  # total thrust
                thrust[idx] = results.thrust_force_vector[0, 0]
                mdot[idx] = results.vehicle_mass_rate[0, 0]

            if imperial:
                scf = 3600. * mdot / 0.1019715 / thrust
                thrust = np.divide(thrust, engine_number) / Units.lbf
            elif SI:
                scf = mdot / thrust * 1e6
                thrust = np.divide(thrust, engine_number) / Units['N']


            if imperial:
                fid.write('{:7.0f} |'.format(float(altitude / Units.ft)))
            elif SI:
                fid.write('{:7.0f} |'.format(float(altitude / Units['m'])))

            fid.write(str(np.transpose(list(map('{:10.1f}'.format, thrust)))))

            fid.write(' |' + 3 * ' ')
            fid.write(str(np.transpose(list(map('{:6.3f} '.format, scf)))))
            fid.write('|')
            fid.write('\n')

    # close file
    fid.close
    # Print timestamp
    fid.write('\n\n' + 50 * '-' + '\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))

    # done!
    return


# ----------------------------------------------------------------------
#   Module Test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(' Error: No test defined ! ')
