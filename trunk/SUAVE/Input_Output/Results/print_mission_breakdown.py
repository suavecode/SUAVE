## @ingroup Input_Output-Results
# print_mission_breakdown.py

# Created:  SUAVE team
# Modified: Aug 2016, L. Kulik
#           Mar 2020, K Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np

from SUAVE.Core import Units
import time                     # importing library
import datetime                 # importing library

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Input_Output-Results
def print_mission_breakdown(results,filename='mission_breakdown.dat', units="imperial"):
    """This creates a file showing mission information.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    results.segments.*.conditions.
      frames.
        inertial.position_vector     [m]
        inertial.time                [s]
      aerodynamics.lift_coefficient  [-]
      weights.total                  [kg]
      freestream.  
        mach_number                  [-]
        pressure                     [Pa]
    filename (optional)       <string> Determines the name of the saved file
    units (option)            <string> Determines the type of units used in the output, options are imperial and si

    Outputs:
    filename                  Saved file with name as above

    Properties Used:
    N/A
    """           
    imperial = False
    SI = False

    if units.lower()=="imperial":
        imperial = True
    elif units.lower()=="si":
        SI = True
    else:
        print("Incorrect system of units selected - choose 'imperial' or 'SI'")
        return

    fid = open(filename,'w')   # Open output file
    fid.write('Output file with mission profile breakdown\n\n') #Start output printing

    k1 = 1.727133242E-06                            # constant to airspeed conversion
    k2 = 0.2857142857                               # constant to airspeed conversion

    TotalRange      = 0.
    total_fuel      = 0.
    total_cryogen   = 0.
    i = 0
    for key in results.segments.keys():        #loop for all segments
        segment = results.segments[key]

        if imperial:
            HPf = -segment.conditions.frames.inertial.position_vector[-1,2] / Units.ft      #Final segment Altitude   [ft]
            HPi = -segment.conditions.frames.inertial.position_vector[0,2] / Units.ft       #Initial segment Altitude  [ft]
        elif SI:
            HPf = -segment.conditions.frames.inertial.position_vector[-1, 2] / Units.m      # Final segment Altitude   [m]
            HPi = -segment.conditions.frames.inertial.position_vector[0, 2] / Units.m       # Initial segment Altitude  [m]

        CLf = segment.conditions.aerodynamics.lift_coefficient[-1]          #Final Segment CL [-]
        CLi = segment.conditions.aerodynamics.lift_coefficient[0]           #Initial Segment CL [-]
        Tf  =  segment.conditions.frames.inertial.time[-1]/ Units.min       #Final Segment Time [min]
        Ti  =  segment.conditions.frames.inertial.time[0] / Units.min       #Initial Segment Time [min]
        Wf  =  segment.conditions.weights.total_mass[-1]                    #Final Segment weight [kg]
        Wi  =  segment.conditions.weights.total_mass[0]                     #Initial Segment weight [kg]

        WCf =  segment.conditions.weights.cryogen_mass[-1]                  #Final Segment Cryogen weight [kg]
        WCi =  segment.conditions.weights.cryogen_mass[0]                   #Initial Segment Cryogen weight [kg]
        WFf =  segment.conditions.weights.fuel_mass[-1]                     #Final Segment fuel (total - cryogen) weight [kg]
        WFi =  segment.conditions.weights.fuel_mass[0]                      #Initial Segment fuel (total - cryogen) weight [kg]

        # print(WCi)

        if imperial:
            Dist = (segment.conditions.frames.inertial.position_vector[-1,0] - segment.conditions.frames.inertial.position_vector[0,0] ) / Units.nautical_miles #Distance [nm]
        elif SI:
            Dist = (segment.conditions.frames.inertial.position_vector[-1, 0] -
                    segment.conditions.frames.inertial.position_vector[0, 0]) / Units.km  # Distance [km]

        TotalRange = TotalRange + Dist

        Mf = segment.conditions.freestream.mach_number[-1]          # Final segment mach number
        Mi = segment.conditions.freestream.mach_number[0]           # Initial segment mach number

        # Aispeed conversion: KTAS to  KCAS
        atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        p0 , dummy , dummy , dummy , dummy  = atmosphere.compute_values(0)
        deltai = segment.conditions.freestream.pressure[0] / p0
        deltaf = segment.conditions.freestream.pressure[-1]/ p0

        VEi = Mi*(340.294*np.sqrt(deltai))          #Equivalent airspeed [m/s]
        QCPOi = deltai*((1.+ k1*VEi**2/deltai)**3.5-1.) #
        VCi = np.sqrt(((QCPOi+1.)**k2-1.)/k1)       #Calibrated airspeed [m/s]
        if imperial:
            KCASi = VCi / Units.knots                   #Calibrated airspeed [knots]
        elif SI:
            KCASi = VCi #Calibrated airspeed [m/s]

        VEf = Mf*(340.294*np.sqrt(deltaf))          #Equivalent airspeed [m/s]
        QCPOf = deltaf*((1.+ k1*VEf**2/deltaf)**3.5-1.)
        VCf = np.sqrt(((QCPOf+1.)**k2-1.)/k1) #m/s  #Calibrated airspeed [m/s]
        if imperial:
            KCASf = VCf / Units.knots                   #Calibrated airspeed [knots]
        elif SI:
            KCASf = VCf

        # Total change in aircraft mass. Represents fuel in normal (non-cryogenic) case
        Fuel    = Wi-Wf

        # Cryogenic weight differences
        CRYO    = WCi - WCf
        FUEL_C  = WFi - WFf

        # Only show cryogen use if cryogen used
        # Strings for when there is no cryogenics
        cryogen_data = ""
        cryogen_unit = ""
        CRYO_str     = ""
        # Test if cryogen data exists
        if "vehicle_cryogen_rate" in results.segments[0].conditions.weights:
            # Modify header strings
            cryogen_data = " CRYOGEN "
            cryogen_unit = "   kg    "
            # Data for cyrogen column
            CRYO_str =  str('%8.2f'   % CRYO)     + '|'
            # Replace total mass difference with the fuel mass difference
            Fuel = FUEL_C

        # String formatting
        CLf_str =   str('%15.3f'   % CLf)     + '|'
        CLi_str =   str('%15.3f'   % CLi)     + '|'
        HPf_str =   str('%7.0f'    % HPf)     + '|'
        HPi_str =   str('%7.0f'    % HPi)     + '|'
        Dist_str =  str('%9.0f'    % Dist)    + '|'
        Wf_str =    str('%8.0f'    % Wf)      + '|'
        Wi_str =    str('%8.0f'    % Wi)      + '|'
        T_str =     str('%7.1f'   % (Tf-Ti))  + '|'
        Mi_str =    str('%7.3f'   % Mi)       + '|'
        Mf_str =    str('%7.3f'   % Mf)       + '|'
        KCASi_str = str('%7.1f'   % KCASi)    + '|'
        KCASf_str = str('%7.1f'   % KCASf)    + '|'
        Fuel_str=   str('%8.2f'   % Fuel)     + '|'

        Segment_str = '%- 31s |' % key 
        

        if i == 0:  # Write header
            if imperial:
                fid.write( '         FLIGHT PHASE           |   ALTITUDE    |     WEIGHT      |  DIST.  | TIME  |            SPEED              |  FUEL  |' + cryogen_data + '\n')
                fid.write( '                                | From  |   To  |Initial | Final  |         |       |Inicial| Final |Inicial| Final |        |\n')
                fid.write( '                                |   ft  |   ft  |   kg   |   kg   |    nm   |  min  | KCAS  | KCAS  |  Mach |  Mach |   kg   |' + cryogen_unit + '\n')
                fid.write( '                                |       |       |        |        |         |       |       |       |       |       |        |\n')
            elif SI:
                fid.write('         FLIGHT PHASE           |   ALTITUDE    |     WEIGHT      |  DIST.  | TIME  |            SPEED              |  FUEL  |' + cryogen_data + '\n')
                fid.write('                                | From  |   To  |Initial | Final  |         |       |Initial| Final |Initial| Final |        |\n')
                fid.write('                                |   m   |   m   |   kg   |   kg   |    km   |  min  | m/s   | m/s   |  Mach |  Mach |   kg   |' + cryogen_unit + '\n')
                fid.write('                                |       |       |        |        |         |       |       |       |       |       |        |\n')

        # Print segment data
        fid.write( Segment_str+HPi_str+HPf_str+Wi_str+Wf_str+Dist_str+T_str+KCASi_str+KCASf_str+Mi_str+Mf_str+Fuel_str+CRYO_str+'\n')

        # Sum fuel and cryogen usage for printing summary once mission complete
        total_fuel      = total_fuel + FUEL_C
        total_cryogen   = total_cryogen + CRYO

        i = i+1

    # Summary of results [nm]
    TotalFuel = results.segments[0].conditions.weights.total_mass[0] - results.segments[-1].conditions.weights.total_mass[-1]  # [kg]
    TotalTime = (results.segments[-1].conditions.frames.inertial.time[-1][0] - results.segments[0].conditions.frames.inertial.time[0][0])  # [min]

    # Summary for systems with cryogen mass usage. TotalFuel is modified to reflect this not being the only variable mass
    if "vehicle_cryogen_rate" in results.segments[0].conditions.weights:
        TotalConsumable = TotalFuel
        TotalFuel = total_fuel

    fid.write(2*'\n')
    if imperial:
        fid.write(' Total Range         (nm) ........... '+ str('%9.0f'   % TotalRange)+'\n')
    elif SI:
        fid.write(' Total Range         (km) ........... ' + str('%9.0f' % TotalRange) + '\n')
    fid.write(' Total Fuel          (kg) ........... '+ str(TotalFuel)+'\n')

    # Cryogen use results
    if "vehicle_cryogen_rate" in results.segments[0].conditions.weights:
        fid.write(' Total Cryogen       (kg) ........... '+ str(total_cryogen)+'\n')
        fid.write(' Total Consumables   (kg) ........... '+ str(TotalConsumable)+'\n')

    fid.write(' Total Time       (hh:mm) ........... '+ time.strftime('    %H:%M', time.gmtime(TotalTime))+'\n')
    # Print timestamp
    fid.write(2*'\n'+ 43*'-'+ '\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))
    
    fid.close

    # done!
    return

# ----------------------------------------------------------------------
#   Module Test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(' Error: No test defined ! ')    
