# full_setup.py
#
# Created:  SUave Team, Aug 2014
# Modified:

""" setup file for a mission with a E190
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# the analysis functions
from the_aircraft_function import the_aircraft_function
from plot_mission import plot_mission

from test_mission_Embraer_E190 import vehicle_setup

from SUAVE.Methods.Performance  import payload_range
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # define the problem
    vehicle, mission = full_setup()

    # run the problem
    results = the_aircraft_function(vehicle,mission)

    # run payload diagram
    outputMission(results.mission_profile,'output.dat')
    cruise_segment_tag = "Cruise"
    payload_range_results = payload_range(vehicle,mission,cruise_segment_tag)

    # check the results
    check_results(results)

    # post process the results
    plot_mission(vehicle,mission,results)

    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    vehicle = vehicle_setup() # imported from E190 test script
    mission = mission_setup(vehicle)

    return vehicle, mission


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
def mission_setup(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Attributes.Missions.Mission()
    mission.tag = 'EMBRAER_E190AR test mission'

    # atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport


    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Throttle
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Throttle_Constant_Speed()
    segment.tag = "CLIMB_250KCAS"

    # connect vehicle configuration
    segment.config = vehicle.configs.takeoff

    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.048 * Units.km
    segment.air_speed      = 250.0 * Units.knots
    segment.throttle       = 1.0

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Throttle
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Throttle_Constant_Speed()
    segment.tag = "CLIMB_280KCAS"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 32000. * Units.ft
    segment.air_speed    = 350.0  * Units.knots
    segment.throttle     = 1.0

    # dummy for post process script
#    segment.climb_rate   = 0.1

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Climb.Constant_Throttle_Constant_Speed()
    segment.tag = "CLIMB_Final"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 37000. * Units.ft
    segment.air_speed    = 390.0  * Units.knots
    segment.throttle     = 1.0

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "Cruise"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet

    segment.air_speed  = 447. * Units.knots #230.  * Units['m/s']
    ## 35kft:
    # 415. => M = 0.72
    # 450. => M = 0.78
    # 461. => M = 0.80
    ## 37kft:
    # 447. => M = 0.78
    segment.distance   = 4000. * Units.nmi

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "DESCENT_M0.77"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 9.31  * Units.km
    segment.air_speed    = 440.0 * Units.knots
    segment.descent_rate = 2600. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "DESCENT_290KCAS"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 3.657 * Units.km
    segment.air_speed    = 365.0 * Units.knots
    segment.descent_rate = 2300. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = SUAVE.Attributes.Missions.Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "DESCENT_250KCAS"

    # connect vehicle configuration
    segment.config = vehicle.configs.cruise

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 250.0 * Units.knots
    segment.descent_rate = 1500. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Mission definition complete
    # ------------------------------------------------------------------

    return mission

#: def define_mission()


def check_results(new_results):

    # load old results
    old_results = load_results()

    # check segment values
    check_list = [
        'mission_profile.segments.Cruise.conditions.aerodynamics.angle_of_attack',
        'mission_profile.segments.Cruise.conditions.aerodynamics.drag_coefficient',
        'mission_profile.segments.Cruise.conditions.aerodynamics.lift_coefficient',
        'mission_profile.segments.Cruise.conditions.aerodynamics.cm_alpha',
        'mission_profile.segments.Cruise.conditions.aerodynamics.cn_beta',
        'mission_profile.segments.Cruise.conditions.propulsion.throttle',
        'mission_profile.segments.Cruise.conditions.propulsion.fuel_mass_rate',
    ]

    # gets a key recursively from a '.' string
    def get_key(data,keys):
        if isinstance(keys,str):
            keys = keys.split('.')
        k = keys.pop(0)
        if keys:
            return get_key(data[k],keys)
        else:
            return data[k]

    # do the check
    for k in check_list:
        print k

        old_val = np.max( get_key(old_results,k) )
        new_val = np.max( get_key(new_results,k) )
        err = (new_val-old_val)/old_val
        print 'Error at Max:' , err
        assert np.abs(err) < 1e-6 , 'Max Check Failed : %s' % k

        old_val = np.min( get_key(old_results,k) )
        new_val = np.min( get_key(new_results,k) )
        err = (new_val-old_val)/old_val
        print 'Error at Min:' , err
        assert np.abs(err) < 1e-6 , 'Min Check Failed : %s' % k

    # check high level outputs
    def check_vals(a,b):
        if isinstance(a,Data):
            for k in a.keys():
                err = check_vals(a[k],b[k])
                if err is None: continue
                print 'outputs' , k
                print 'Error:' , err
                assert np.abs(err) < 1e-6 , 'Outputs Check Failed : %s' % k
        else:
            return (a-b)/a

    # do the check
    check_vals(old_results.output,new_results.output)

    return

def load_results():
    return SUAVE.Plugins.VyPy.data.load('results_mission_E190_constThr.pkl')

def save_results(results):
    SUAVE.Plugins.VyPy.data.save(results,'results_mission_E190_constThr.pkl')





def outputMission(results,filename):

    import time                     # importing library
    import datetime                 # importing library

    fid = open(filename,'w')   # Open output file
    fid.write('Output file with mission profile breakdown\n\n') #Start output printing

    k1 = 1.727133242E-06                            # constant to airspeed conversion
    k2 = 0.2857142857                               # constant to airspeed conversion

    TotalRange = 0
    for i in range(len(results.segments)):          #loop for all segments
        segment = results.segments[i]

        HPf = -segment.conditions.frames.inertial.position_vector[-1,2] / Units.ft      #Final segment Altitude   [ft]
        HPi = -segment.conditions.frames.inertial.position_vector[0,2] / Units.ft       #Initial segment Altitude  [ft]

        CLf = segment.conditions.aerodynamics.lift_coefficient[-1]     #Final Segment CL [-]
        CLi = segment.conditions.aerodynamics.lift_coefficient[0]      #Initial Segment CL [-]
        Tf =  segment.conditions.frames.inertial.time[-1]/ Units.min   #Final Segment Time [min]
        Ti =  segment.conditions.frames.inertial.time[0] / Units.min   #Initial Segment Time [min]
        Wf =  segment.conditions.weights.total_mass[-1]                  #Final Segment weight [kg]
        Wi =  segment.conditions.weights.total_mass[0]                   #Initial Segment weight [kg]
        Dist = (segment.conditions.frames.inertial.position_vector[-1,0] - segment.conditions.frames.inertial.position_vector[0,0] ) / Units.nautical_miles #Distance [nm]
        TotalRange = TotalRange + Dist

        Mf = segment.conditions.freestream.mach_number[-1]          # Final segment mach number
        Mi = segment.conditions.freestream.mach_number[0]           # Initial segment mach number

#       Aispeed conversion: KTAS to  KCAS
        p0 = np.interp(0,segment.atmosphere.breaks.altitude,segment.atmosphere.breaks.pressure )
        deltai = segment.conditions.freestream.pressure[0] / p0
        deltaf = segment.conditions.freestream.pressure[-1]/ p0

        VEi = Mi*(340.294*np.sqrt(deltai))          #Equivalent airspeed [m/s]
        QCPOi = deltai*((1.+ k1*VEi**2/deltai)**3.5-1.) #
        VCi = np.sqrt(((QCPOi+1.)**k2-1.)/k1)       #Calibrated airspeed [m/s]
        KCASi = VCi / Units.knots                   #Calibrated airspeed [knots]

        VEf = Mf*(340.294*np.sqrt(deltaf))          #Equivalent airspeed [m/s]
        QCPOf = deltaf*((1.+ k1*VEf**2/deltaf)**3.5-1.)
        VCf = np.sqrt(((QCPOf+1.)**k2-1.)/k1) #m/s  #Calibrated airspeed [m/s]
        KCASf = VCf / Units.knots                   #Calibrated airspeed [knots]

#       String formatting
        CLf_str =   str('%15.3f'   % CLf)     + '|'
        CLi_str =   str('%15.3f'   % CLi)     + '|'
        HPf_str =   str('%7.0f'    % HPf)     + '|'
        HPi_str =   str('%7.0f'    % HPi)     + '|'
        Dist_str =  str('%9.0f'    % Dist)    + '|'
        Wf_str =    str('%8.0f'    % Wf)      + '|'
        Wi_str =    str('%8.0f'    % Wi)      + '|'
        T_str =     str('%7.1f'   % (Tf-Ti))  + '|'
        Fuel_str=   str('%8.0f'   % (Wi-Wf))  + '|'
        Mi_str =    str('%7.3f'   % Mi)       + '|'
        Mf_str =    str('%7.3f'   % Mf)       + '|'
        KCASi_str = str('%7.1f'   % KCASi)    + '|'
        KCASf_str = str('%7.1f'   % KCASf)    + '|'

        Segment_str = ' ' + segment.tag[0:31] + (31-len(segment.tag))*' ' + '|'

        if i == 0:  #Write header
            fid.write( '         FLIGHT PHASE           |   ALTITUDE    |     WEIGHT      |  DIST.  | TIME  |  FUEL  |            SPEED              |\n')
            fid.write( '                                | From  |   To  |Inicial | Final  |         |       |        |Inicial| Final |Inicial| Final |\n')
            fid.write( '                                |   ft  |   ft  |   kg   |   kg   |    nm   |  min  |   kg   | KCAS  | KCAS  |  Mach |  Mach |\n')
            fid.write( '                                |       |       |        |        |         |       |        |       |       |       |       |\n')
        # Print segment data
        fid.write( Segment_str+HPi_str+HPf_str+Wi_str+Wf_str+Dist_str+T_str+Fuel_str+KCASi_str+KCASf_str+Mi_str+Mf_str+'\n')

    #Summary of results [nm]
    TotalFuel = results.segments[0].conditions.weights.total_mass[0] - results.segments[-1].conditions.weights.total_mass[-1]   #[kg]
    TotalTime = (results.segments[-1].conditions.frames.inertial.time[-1] - results.segments[0].conditions.frames.inertial.time[0])  #[min]

    fid.write(2*'\n')
    fid.write(' Total Range (nm) ........... '+ str('%9.0f'   % TotalRange)+'\n')
    fid.write(' Total Fuel  (kg) ........... '+ str('%9.0f'   % TotalFuel)+'\n')
    fid.write(' Total Time  (hh:mm) ........ '+ time.strftime('    %H:%M', time.gmtime(TotalTime))+'\n')
    # Print timestamp
    fid.write(2*'\n'+ 43*'-'+ '\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))











if __name__ == '__main__':
    main()
    plt.show()