""" print_parasite_drag.py """

# Created: SUAVE team
# Updated: Carlos Ilario, Feb 2016

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units,Data
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Drag import miscellaneous_drag_aircraft_ESDU

from scipy.optimize import fsolve # for compatibility with scipy 0.10.0
import numpy as np


# ----------------------------------------------------------------------
#  Print output file with parasite drag breakdown
# ----------------------------------------------------------------------

def print_parasite_drag(ref_condition,vehicle,analyses,filename = 'parasite_drag.dat'):
    """ SUAVE.Methods.Results.print_parasite_drag(ref_condition,vehicle,filename = 'parasite_drag.dat'):
        
        Print output file with parasite drag breakdown
        
        Inputs:
            ref_condition   - data dictionary with fields:
                mach_number     - Mach number to be used in the drag estimation
                reynolds_number - Reynolds number to be used in the drag estimation
                
            vehicle         - SUave type vehicle

            filename [optional] - Name of the file to be created

        Outputs:
            output file

        Assumptions:
            Altitude to be estimated, in order to have a condition with the Reynolds 
            and Mach number required

    """
    
    # Imports
    import time                     # importing library
    import datetime                 # importing library

    # Unpack
    Mc                      = ref_condition.mach_number
    Rey                     = ref_condition.reynolds_number
    mean_aerodynamic_chord  = vehicle.wings['main_wing'].chords.mean_aerodynamic
    aspect_ratio            = vehicle.wings['main_wing'].aspect_ratio
    sweep                   = vehicle.wings['main_wing'].sweeps.quarter_chord  / Units.deg
    t_c                     = vehicle.wings['main_wing'].thickness_to_chord
    taper                   = vehicle.wings['main_wing'].taper
    sref                    = vehicle.reference_area
    
    settings                = analyses.configs.cruise.aerodynamics.settings

    # Defining conditions to solve for altitude that gives the input Rey and Mach
    alt_conditions = Data()
    alt_conditions.Mc  = Mc
    alt_conditions.Rey = Rey
    alt_conditions.mac = mean_aerodynamic_chord
    # solve system
    altitude = fsolve( func  = solve_altitude  ,
                        x0   = 0.              ,
                        args = alt_conditions           )

    # compute atmosphere
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmosphere.compute_values(altitude)
    
    p   = atmo_data.pressure
    T   = atmo_data.temperature
    rho = atmo_data.density
    a   = atmo_data.speed_of_sound
    mew = atmo_data.dynamic_viscosity
    
    # Find the dimensional RE, ie. Reynolds number/length
    re = rho*Mc*a/mew

    # Define variables needed in the aerodynamic method
    state = Data()
    state.conditions = Data()
    state.conditions.freestream = Data()
    state.conditions.freestream.mach_number       = np.atleast_1d(Mc)
    state.conditions.freestream.density           = np.atleast_1d(rho)
    state.conditions.freestream.dynamic_viscosity = np.atleast_1d(mew)
    state.conditions.freestream.reynolds_number   = np.atleast_1d(re)
    state.conditions.freestream.temperature       = np.atleast_1d(T)
    state.conditions.freestream.pressure          = np.atleast_1d(p)
    state.conditions.aerodynamics = Data()
    state.conditions.aerodynamics.drag_breakdown = Data()
    state.conditions.aerodynamics.drag_breakdown.parasite = Data()

    # Compute parasite drag of components    
    compute = analyses.configs.cruise.aerodynamics.process.compute.drag
    for wing in vehicle.wings:
        compute.parasite.wings.wing(state,settings,wing)
    
    for fuselage in vehicle.fuselages:
        compute.parasite.fuselages.fuselage(state,settings,fuselage)    
        
    for propulsor in vehicle.propulsors:
        compute.parasite.propulsors.propulsor(state,settings,propulsor) 
      
    compute.parasite.pylons(state,settings,vehicle) 
    compute.miscellaneous(state,settings,vehicle)
    compute.parasite.total(state,settings,vehicle)
    
    # getting induced drag efficiency factor
    state.conditions.aerodynamics.lift_coefficient = 0.5 # dummy value
    compute.induced(state,settings,vehicle)
    eff_fact = state.conditions.aerodynamics.drag_breakdown.induced.efficiency_factor
    # reynolds number
    
    Re_w = rho * Mc * a * mean_aerodynamic_chord/mew

    fid = open(filename,'w')   # Open output file
    fid.write('Output file with parasite drag breakdown\n\n') #Start output printing    
    fid.write( '  VEHICLE TAG : ' + vehicle.tag + '\n\n')
    fid.write( '  REFERENCE AREA .................. ' + str('%5.1f' %   sref               )   + ' m2 ' + '\n')
    fid.write( '  ASPECT RATIO .................... ' + str('%5.1f' %   aspect_ratio       )   + '    ' + '\n')
    fid.write( '  WING SWEEP ...................... ' + str('%5.1f' %   sweep              )   + ' deg' + '\n')
    fid.write( '  WING THICKNESS RATIO ............ ' + str('%5.2f' %   t_c                )   + '    ' + '\n')
    fid.write( '  INDUCED DRAG EFFICIENCY FACTOR .. ' + str('%5.3f' %   eff_fact             )   + '    '  + '\n')
    fid.write( '  MEAN AEROD. CHORD ............... ' + str('%5.3f' %mean_aerodynamic_chord)   + ' m ' + '\n')
    fid.write( '  REYNOLDS NUMBER ................. ' + str('%5.1f' %   (Re_w / (10**6))   )   + ' millions' + '\n')
    fid.write( '  MACH NUMBER ..................... ' + str('%5.3f' %   Mc                 )   + '    '  + '\n')

    fid.write( '\n\n' )
    fid.write( '            COMPONENT                 |      CDO      |  WETTED AREA  |  FORM FACTOR  | FLAT PLATE CF.| REYNOLDS FACT.| COMPRES. FACT.|\n')
    fid.write( '                                      |      [-]      |      [m2]     |      [-]      |      [-]      |      [-]      |      [-]      |\n')
    fid.write( '                                      |               |               |               |               |               |               |\n')

    drag_breakdown  = state.conditions.aerodynamics.drag_breakdown.parasite
    swet_tot        = 0.
    CD_p            = 0.
    for wing in vehicle.wings:
        drag_breakdown[wing.tag].tag = wing.tag
        swet_tot += drag_breakdown[wing.tag].wetted_area
        CD_p     += drag_breakdown[wing.tag].parasite_drag_coefficient

    for fuselage in vehicle.fuselages:
        drag_breakdown[fuselage.tag].tag = fuselage.tag
        swet_tot += drag_breakdown[fuselage.tag].wetted_area
        CD_p     += drag_breakdown[fuselage.tag].parasite_drag_coefficient

    for propulsor in vehicle.propulsors:
        drag_breakdown[propulsor.tag].tag = propulsor.tag[0:25] + '  (EACH)'
        CD_p     += drag_breakdown[propulsor.tag].parasite_drag_coefficient
        drag_breakdown[propulsor.tag].parasite_drag_coefficient /= propulsor.number_of_engines
        swet_tot += drag_breakdown[propulsor.tag].wetted_area * propulsor.number_of_engines

    drag_breakdown['pylon'].tag = 'Pylon (TOTAL)'
    CD_p     += drag_breakdown['pylon'].parasite_drag_coefficient
    swet_tot += drag_breakdown['pylon'].wetted_area

    for k in drag_breakdown:
        if isinstance(k,SUAVE.Analyses.Results):
            # String formatting
            component       =   ' ' + k.tag[0:37] + (37-len(k.tag))*' '         + '|'
            wetted_area     =   str('%11.1f'   % k.wetted_area)                 + '    |'
            form_factor     =   str('%11.3f'   % k.form_factor)                 + '    |'
            f_rey           =   str('%11.3f'   % k.reynolds_factor)             + '    |'
            cd_p            =   str('%11.5f'   % k.parasite_drag_coefficient)   + '    |'
            f_compress      =   str('%11.3f'   % k.compressibility_factor)      + '    |'
            cf              =   str('%11.5f'   % k.skin_friction_coefficient)   + '    |'

            # Print segment data
            fid.write(component + cd_p + wetted_area + form_factor + cf + f_rey + f_compress + '\n')

    # Print miscelllaneous drag
    component             =   ' Miscellaneous Drag' + 19*' ' +'|'
    cd_misc               =   str('%11.5f'   % state.conditions.aerodynamics.drag_breakdown.miscellaneous.total)   + '    |'
    CD_p                 +=  float(cd_misc[0:14])
    fid.write(component + cd_misc + 5*( 8*' '+'-'+6*' '+'|') + '\n')

    # String formatting for miscelllaneous drag
    component             =   ' Drag Coefficient Increment' + 11*' ' +'|'
    cd_increment          =   str('%11.5f'   % settings.drag_coefficient_increment)   + '    |'
    CD_p                 +=  float(cd_increment[0:14])
    fid.write(component + cd_increment + 5*( 8*' '+'-'+6*' '+'|') + '\n')
    
    # Print line with totals
    swet_tot     =  str('%11.1f'   % swet_tot)                 + '    |'
    CD_p         =  str('%11.5f'   % CD_p    )                 + '    |'
    fid.write(38*' ' + '|'+ 6*( 15*' '+'|') + '\n')
    fid.write(29*' ' + 'SUM 	  |' + CD_p + swet_tot + 4*( 15*' '+'|') + '\n')

    # Print timestamp
    fid.write(2*'\n'+ 43*'-'+ '\n' + datetime.datetime.now().strftime(" %A, %d. %B %Y %I:%M:%S %p"))
    # done
    fid.close
    
    
def solve_altitude(alt,alt_conditions):

    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmosphere.compute_values(alt)
    
    p   = atmo_data.pressure
    T   = atmo_data.temperature
    rho = atmo_data.density
    a   = atmo_data.speed_of_sound
    mew = atmo_data.dynamic_viscosity

    # conditions
    Mc  = alt_conditions.Mc
    mac = alt_conditions.mac
    Rey_ref = alt_conditions.Rey

    # reynolds number
    Rey = float( rho * Mc * a * (mac)/mew )

    # residual
    r = Rey - Rey_ref

    return r


# ----------------------------------------------------------------------
#   Module Test
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(' Error: No test defined ! ')    