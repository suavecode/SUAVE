## @ingroup Analyses-Constraint_Analysis
# Constraint_Analysis.py
#
# Created:  Oct 2020, S. Karpuk 
# Modified: 
#          

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
import SUAVE
from SUAVE.Core import Units, Data

# Package imports
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Class
# ----------------------------------------------------------------------
## @ingroup Analyses-Constraint_Analysis
class Constraint_Analysis():
    """Creates a constraint diagram using classical sizing methods

    Assumptions:
        None

    Source:
        S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082
        L. Loftin,Subsonic Aircraft: Evolution and the Matching of Size to Performance, NASA Ref-erence Publication 1060, August 1980
        M. Nita, D.Scholtz, 'Estimating the Oswald factor from basic aircraft geometrical parameters',Deutscher Luft- und Raumfahrtkongress 20121DocumentID: 281424

    
    """
    
    def __init__(self):
        """This sets the default values and methods for the analysis.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """          
        self.tag                = 'Constraint analysis'
        self.plot_tag           = False
        self.plot_units         = 'SI'

        # Defines default constraint analyses
        self.analyses = Data()
        self.analyses.takeoff   = True
        self.analyses.cruise    = True
        self.analyses.ma_cruise = False
        self.analyses.landing   = True
        self.analyses.OEI_climb = True
        self.analyses.turn      = False
        self.analyses.climb     = False
        self.analyses.ceiling   = False
        
        self.wing_loading      = np.arange(5, 200, 5) * Units['force_pound/foot**2']
        self.design_point_type = None

        # Default parameters for the constraint analysis
        # take-off
        self.takeoff = Data()
        self.takeoff.runway_elevation     = 0.0
        self.takeoff.ground_run           = 0.0
        self.takeoff.rolling_resistance   = 0.05
        self.takeoff.liftoff_speed_factor = 1.1
        self.takeoff.delta_ISA            = 0.0
        # climb
        self.climb = Data()
        self.climb.altitude   = 0.0
        self.climb.airspeed   = 0.0
        self.climb.climb_rate = 0.0
        self.climb.delta_ISA  = 0.0
        # OEI climb
        self.OEI_climb = Data()
        self.OEI_climb.climb_speed_factor = 1.2
        # cruise
        self.cruise = Data()
        self.cruise.altitude        = 0.0
        self.cruise.delta_ISA       = 0.0
        self.cruise.airspeed        = 0.0
        self.cruise.thrust_fraction = 0.0
        # max cruise
        self.max_cruise = Data()
        self.max_cruise.altitude        = 0.0
        self.max_cruise.delta_ISA       = 0.0
        self.max_cruise.mach            = 0.0
        self.max_cruise.thrust_fraction = 0.0
        # turn
        self.turn = Data()
        self.turn.angle           = 0.0
        self.turn.altitude        = 0.0
        self.turn.delta_ISA       = 0.0
        self.turn.airspeed        = 0.0
        self.turn.specific_energy = 0.0
        # ceiling
        self.ceiling = Data()
        self.ceiling.altitude  = 0.0
        self.ceiling.delta_ISA = 0.0
        self.ceiling.airspeed  = 0.0
        # landing
        self.landing = Data()
        self.landing.ground_roll           = 0.0
        self.landing.approach_speed_factor = 1.23
        self.landing.runway_elevation      = 0.0
        self.landing.delta_ISA             = 0.0 

        # Default aircraft properties
        # geometry
        self.geometry = Data()
        self.geometry.aspect_ratio           = 0.0 
        self.geometry.taper                  = 0.0
        self.geometry.thickness_to_chord     = 0.0
        self.geometry.sweep_quarter_chord    = 0.0
        self.geometry.high_lift_type_clean   = None
        self.geometry.high_lift_type_takeoff = None
        self.geometry.high_lift_type_landing = None 
        # engine
        self.engine = Data()
        self.engine.type             = None
        self.engine.number           = 0
        self.engine.bypass_ratio     = 0.0 
        self.engine.degree_of_hybridization = 0.0
        self.engine.throttle_ratio   = 1.0   
        self.engine.afterburner      = False
        self.engine.method           = 'Mattingly'
        # propeller
        self.propeller = Data()
        self.propeller.takeoff_efficiency   = 0.0
        self.propeller.climb_efficiency     = 0.0
        self.propeller.cruise_efficiency    = 0.0
        self.propeller.turn_efficiency      = 0.0
        self.propeller.ceiling_efficiency   = 0.0
        self.propeller.OEI_climb_efficiency = 0.0

        # Define aerodynamics
        self.aerodynamics = Data()
        self.aerodynamics.oswald_factor   = 0.0
        self.aerodynamics.cd_takeoff      = 0.0   
        self.aerodynamics.cl_takeoff      = 0.0   
        self.aerodynamics.cl_max_takeoff  = 0.0
        self.aerodynamics.cl_max_landing  = 0.0  
        self.aerodynamics.cl_max_clean    = 0.0
        self.aerodynamics.cd_min_clean    = 0.0
        self.aerodynamics.fuselage_factor = 0.974
        self.aerodynamics.viscous_factor  = 0.38
        

    def create_constraint_diagram(self,plot_figure = True): 
        """Creates a constraint diagram

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
            Constraint diagram data
            Design wing loading and thrust-to-weight ratio

        Properties Used:
        """

        # Go through each constraint estimation to find thrust-to-weight ratios
        analyses         = self.analyses
        design_point_tag = self.design_point_type

        fig = plt.figure()
        ax  = fig.add_subplot(1, 1, 1)

        all_constraints = []

        # Convert the wing loading depinting on the user input
        if self.plot_units == 'US':
            WS = self.wing_loading * 0.020885       # convert to lb/sq ft
        else:
            WS = self.wing_loading / 9.81           # convert to kg/sq m

        #take-off
        if analyses.takeoff == True:
            TW_TO = self.take_off_constraints()
            all_constraints.append(TW_TO)
            ax.plot(WS,TW_TO, label = 'Take-off')

        # turn
        if analyses.turn == True:
            TW_turn = self.turn_constraints()
            all_constraints.append(TW_turn)
            ax.plot(WS,TW_turn, label = 'Turn')

        # cruise
        if analyses.cruise == True:
            TW_cruise = self.cruise_constraints('normal cruise')
            all_constraints.append(TW_cruise)
            ax.plot(WS,TW_cruise, label = 'Cruise')

        # maximum cruise
        if analyses.max_cruise == True:
            TW_max_cruise = self.cruise_constraints('max cruise')
            all_constraints.append(TW_max_cruise)
            ax.plot(WS,TW_max_cruise, label = 'Max Cruise')  
        
        # climb
        if analyses.climb == True:
            TW_climb = self.climb_constraints()
            all_constraints.append(TW_climb)
            ax.plot(WS,TW_climb, label = 'Climb')

        # ceiling
        if analyses.ceiling == True:
            TW_ceiling = self.ceiling_constraints()
            all_constraints.append(TW_ceiling)
            ax.plot(WS,TW_ceiling, label = 'Ceiling')

        # landing
        if analyses.landing == True:
            WS_landing, TW_landing    = self.landing_constraints()
            ax.plot(WS_landing,TW_landing, label = 'Landing')

        # OEI 2nd segment climb
        if analyses.OEI_climb == True:
            TW_OEI_climb = self.OEI_climb_constraints()
            all_constraints.append(TW_OEI_climb)
            ax.plot(WS,TW_OEI_climb, label = '2nd segm-t OEI climb')

        # Find the design point
        combined_curve             = np.amax(all_constraints,0)
        self.combined_design_curve = combined_curve 

        if design_point_tag == 'minimum thrust-to-weight' or  design_point_tag == 'minimum power-to-weight':
            design_TW = min(combined_curve)                              
            design_WS = WS[np.argmin(combined_curve)] 
        elif design_point_tag == 'maximum wing loading':
            design_WS = WS_landing[0]
            design_TW = np.interp(design_WS,WS,combined_curve) 

        # Check the landing constraint
        if design_WS > WS_landing[0]:
            design_WS = WS_landing[0]
            design_TW = np.interp(design_WS, WS, combined_curve) 

        ax.plot(WS,combined_curve, label = 'Combined', color = 'r')
        ax.scatter(design_WS,design_TW, label = 'Design point')
        ax.set_xlim(0, WS_landing[0]+10)

        # Plot the constraint diagram and store the design point
        ax.legend(loc=2,)
        ax.grid(True)
        if self.plot_units == 'US':
            design_WS  = 4.88243 * design_WS 
            landing_WS = 4.88243 * WS_landing
            plt.ylim(0, 0.5)
            if self.engine.type != 'turbofan' and self.engine.type != 'turbojet':
                ax.set_ylabel('P/W, hp/lb')
                design_TW = 1.64399 * design_TW      # convert from hp/lb to kW/kg 
                plt.ylim(0, 0.3)
            else:
                ax.set_ylabel('T/W, lb/lb')
                design_TW = 9.81 * design_TW         # convert from lb/lb to N/kg
            ax.set_xlabel('W/S, lb/sq ft')  
            plt.ylim(0, 0.2)
        else:
            landing_WS = WS_landing
            
            if self.engine.type != 'turbofan' and self.engine.type != 'turbojet':
                ax.set_ylabel('P/W, kW/kg')
                plt.ylim(0, 0.3)
            else:
                ax.set_ylabel('T/W, N/kg')  
                plt.ylim(0, 6)
            ax.set_xlabel('W/S, kg/sq m')    

        if plot_figure == True:         
            plt.savefig("Constarint_diagram.png", dpi=150)

        # Write the constrain diagram design point into a file
        f = open("Constraint_analysis_data.dat", "w")
        f.write('Output file with the constraint analysis design point\n\n')           
        f.write("Design point :\n")
        f.write('     Wing loading = ' + str(design_WS) + ' kg/sq m\n')
        f.write('                    ' + str(design_WS*0.204816) + ' lb/sq ft\n')  
        if self.engine.type != 'turbofan' and self.engine.type != 'turbojet':
            f.write('     Power-to-weight ratio = ' + str(design_TW) + ' kW/kg\n')    
            f.write('                             ' + str(design_TW*0.608277388) + ' hp/lb\n')
        else:
            f.write('     Thrust-to-weight ratio = ' + str(design_TW) + ' N/kg\n')
            f.write('                              ' + str(design_TW/9.81) + ' lb/lb\n')    
        f.close()
        
        # Pack outputs
        self.des_wing_loading     = design_WS
        self.landing_wing_loading = landing_WS
        self.des_thrust_to_weight = design_TW

        return


    # Auxilary functions to create the constraint diagram
    #-----------------------------------------------------------------------------------
    def take_off_constraints(self):
        """Calculate thrust-to-weight ratios at take-off

        Assumptions:
            Maximum take-off lift coefficient is 85% of the maximum landing lift coefficient
            
        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            self.engine.type                        string
                 aerodynamics.cd_takeoff            [Unitless]
                              cl_takeoff            [Unitless]
                              cl_max_takeoff        [Unitless]
                takeoff.ground_run                  [m]      
                        liftoff_speed_factor        [Unitless]
                        rolling_resistance
                        runway_elevation            [m]
                        delta_ISA                   [K]
                        
                wing_loading                        [N/m**2]
                
        Outputs:
            constraints.T_W                         [Unitless]
                        W_S                         [N/m**2]

        Properties Used:

        """  
        # Unpack inputs
        eng_type  = self.engine.type
        cd_TO     = self.aerodynamics.cd_takeoff 
        cl_TO     = self.aerodynamics.cl_takeoff 
        cl_max_TO = self.aerodynamics.cl_max_takeoff
        Sg        = self.takeoff.ground_run
        eps       = self.takeoff.liftoff_speed_factor
        miu       = self.takeoff.rolling_resistance
        altitude  = self.takeoff.runway_elevation
        delta_ISA = self.takeoff.delta_ISA
        W_S       = self.wing_loading

        # Set take-off aerodynamic properties
        if cl_TO == 0 or cd_TO == 0:
            raise ValueError("Define cl_takeoff or cd_takeoff\n")

        if cl_max_TO == 0:
            cl_max_LD = self.estimate_max_lift(self.geometry.high_lift_type_takeoff)     # Landing maximum lift coefficient     
            cl_max_TO = 0.85 * cl_max_LD                                                 # Take-off flaps settings

        # Check if the take-off distance was input
        if Sg == 0:
            raise ValueError("Input the ground_run distance\n")

        # Determine atmospheric properties at the altitude
        planet      = SUAVE.Analyses.Planets.Planet()
        atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmosphere.features.planet = planet.features

        atmo_values = atmosphere.compute_values(altitude,delta_ISA)
        rho         = atmo_values.density[0,0]
        p           = atmo_values.pressure[0,0]
        T           = atmo_values.temperature[0,0]
        g           = atmosphere.planet.sea_level_gravity

        T_W = np.zeros(len(W_S))
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            P_W  = np.zeros(len(W_S))
            etap = self.propeller.takeoff_efficiency
            if etap == 0:
                raise ValueError('Warning: Set the propeller efficiency during take-off')
        for i in range(len(W_S)):
            Vlof   = eps*np.sqrt(2*W_S[i]/(rho*cl_max_TO))
            Mlof   = Vlof/atmo_values.speed_of_sound[0,0]
            T_W[i] = eps**2*W_S[i] / (g*Sg*rho*cl_max_TO) + eps*cd_TO/(2*cl_max_TO) + miu * (1-eps*cl_TO/(2*cl_max_TO))
            
            # Convert thrust to power (for propeller-driven engines) and normalize the results wrt the Sea-level
            if eng_type != 'turbofan' and eng_type != 'turbojet':
                
                P_W[i] = T_W[i]*Vlof/etap

                if eng_type == 'turboprop':
                    P_W[i] = P_W[i] / self.normalize_turboprop_thrust(p,T) 
                elif eng_type == 'piston':
                    P_W[i] = P_W[i] / self.normalize_power_piston(rho)
                elif eng_type == 'electric':
                    P_W[i] = P_W[i] / self.normalize_power_electric(rho)  
            else:
               T_W[i] = T_W[i] / self.normalize_gasturbine_thrust(eng_type,p,T,Mlof,altitude,'takeoff')  
     

        # Pack outputs
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            if self.plot_units == 'US':
                constraint = 0.00608 * P_W       # convert to hp/lb
            else:
                constraint = 9.81* P_W/1000    # convert to kW/kg
        else:
            if self.plot_units != 'US':
                constraint = 9.81* T_W         # convert to N/kg
            else:
                constraint = T_W

        return constraint
        
    def turn_constraints(self):
        """Calculate thrust-to-weight ratios for the turn maneuver

        Assumptions:
            Minimum drag coefficient is independent on the altitude and airspeed

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            self.aerodynamics.cd_min_clean      [Unitless]
                 engine.type                    string
                 turn.angle                     [radians]
                      altitude                  [m]
                      delta_ISA                 [K]
                      airspeed                  [m/s]
                      specific_energy
                      thrust_fraction           [Unitless]
                 geometry.aspect_ratio          [Unitless]
                 wing_loading                   [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]
                        
        Properties Used:
        """  

        # Unpack inputs
        cd_min    = self.aerodynamics.cd_min_clean 
        eng_type  = self.engine.type
        phi       = self.turn.angle         
        altitude  = self.turn.altitude        
        delta_ISA = self.turn.delta_ISA       
        M         = self.turn.mach       
        Ps        = self.turn.specific_energy 
        throttle  = self.turn.thrust_fraction   
        AR        = self.geometry.aspect_ratio
        W_S       = self.wing_loading

        # Determine atmospheric properties at the altitude
        planet      = SUAVE.Analyses.Planets.Planet()
        atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmosphere.features.planet = planet.features

        atmo_values = atmosphere.compute_values(altitude,delta_ISA)
        rho         = atmo_values.density[0,0]
        p           = atmo_values.pressure[0,0]
        T           = atmo_values.temperature[0,0]
        Vturn       = M * atmo_values.speed_of_sound[0,0]
        q           = 0.5*rho*Vturn**2
        g           = atmosphere.planet.sea_level_gravity   

        T_W = np.zeros(len(W_S))
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            P_W  = np.zeros(len(W_S))
            etap = self.propeller.turn_efficiency
            if etap == 0:
                raise ValueError('Warning: Set the propeller efficiency during turn')

        for i in range(len(W_S)):
            CL = W_S[i]/q

            # Calculate compressibility_drag
            cd_comp   = self.compressibility_drag(M,CL) 
            cd_comp_e = self.compressibility_drag(M,0)  
            cd0     = cd_min + cd_comp  

            # Calculate Oswald efficiency
            if self.aerodynamics.oswald_factor == 0:
                e = self.oswald_efficiency(cd_min+cd_comp_e,M)

            k      = 1/(np.pi*e*AR)
            T_W[i] = q * (cd0/W_S[i] + k/(q*np.cos(phi))**2*W_S[i]) + Ps/Vturn

        # Convert thrust to power (for propeller-driven engines) and normalize the results wrt the Sea-level
        if eng_type != 'turbofan' and eng_type != 'turbojet':

            P_W = T_W*Vturn/etap

            if eng_type == 'turboprop':
                P_W = P_W / self.normalize_turboprop_thrust(p,T) 
            elif eng_type == 'piston':
                P_W = P_W / self.normalize_power_piston(rho) 
            elif eng_type == 'electric':
                P_W = P_W / self.normalize_power_electric(rho) 
        else:
            T_W = T_W / self.normalize_gasturbine_thrust(eng_type,p,T,M,altitude,'turn')          

        # Pack outputs normalized by throttle
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            if self.plot_units == 'US':
                constraint = 0.006 * P_W / throttle      # convert to hp/lb
            else:
                constraint = 9.81 * P_W/1000 / throttle    # convert to kW/kg
        else:
            if self.plot_units != 'US':
                constraint = 9.81 * T_W / throttle         # convert to N/kg
            else:
                constraint = T_W / throttle

        return constraint

    def cruise_constraints(self,cruise_tag):
        """Calculate thrust-to-weight ratios for the cruise

        Assumptions:
           N/A 

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            self.engine.type                    string
            aerodynamics.cd_min_clean           [Unitless]
            cruise.altitude                     [m]
                   delta_ISA                    [K]
                   airspeed                     [m/s]
                   thrust_fraction              [Unitless]
            geometry.aspect_ratio               [Unitless]
            wing_loading                        [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:
        """  

        # Unpack inputs
        if cruise_tag == 'max cruise':
            altitude    = self.max_cruise.altitude 
            delta_ISA   = self.max_cruise.delta_ISA
            M           = self.max_cruise.mach 
            throttle    = self.max_cruise.thrust_fraction 
        else:
            altitude    = self.cruise.altitude 
            delta_ISA   = self.cruise.delta_ISA
            M           = self.cruise.mach 
            throttle    = self.cruise.thrust_fraction

        eng_type  = self.engine.type
        cd_min    = self.aerodynamics.cd_min_clean 
        AR        = self.geometry.aspect_ratio
        W_S       = self.wing_loading
              
        # Determine atmospheric properties at the altitude
        planet      = SUAVE.Analyses.Planets.Planet()
        atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmosphere.features.planet = planet.features

        atmo_values = atmosphere.compute_values(altitude,delta_ISA)
        rho         = atmo_values.density[0,0]
        p           = atmo_values.pressure[0,0]
        T           = atmo_values.temperature[0,0]
        Vcr         = M * atmo_values.speed_of_sound[0,0]
        q           = 0.5*rho*Vcr**2
        g           = atmosphere.planet.sea_level_gravity
   

        T_W = np.zeros(len(W_S))
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            P_W  = np.zeros(len(W_S))
            etap = self.propeller.cruise_efficiency
            if etap == 0:
                raise ValueError('Warning: Set the propeller efficiency during cruise')
        for i in range(len(W_S)):
            CL = W_S[i]/q

            # Calculate compressibility_drag
            cd_comp   = self.compressibility_drag(M,CL) 
            cd_comp_e = self.compressibility_drag(M,0)  
            cd0       = cd_min + cd_comp  

            # Calculate Oswald efficiency
            if self.aerodynamics.oswald_factor == 0:
                e = self.oswald_efficiency(cd_min+cd_comp_e,M)

            k = 1/(np.pi*e*AR)
            T_W[i] = q*cd0/W_S[i]+k*W_S[i]/q

        # Convert thrust to power (for propeller-driven engines) and normalize the results wrt the Sea-level
        if eng_type != 'turbofan' and eng_type != 'turbojet':

            P_W = T_W*Vcr/etap

            if eng_type == 'turboprop':
                P_W = P_W / self.normalize_turboprop_thrust(p,T)
            elif eng_type == 'piston':
                P_W = P_W / self.normalize_power_piston(rho)
            elif eng_type == 'electric':
                P_W = P_W / self.normalize_power_electric(rho)
        else:
            T_W = T_W / self.normalize_gasturbine_thrust(eng_type,p,T,M,altitude,'cruise')     


        # Pack outputs
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            if self.plot_units == 'US':
                constraint = 0.006 * P_W / throttle      # convert to hp/lb
            else:
                constraint = 9.81 * P_W/1000 / throttle   # convert to kW/kg
        else:
            if self.plot_units != 'US':
                constraint = 9.81 * T_W / throttle        # convert to N/kg
            else:
                constraint = T_W / throttle

        return constraint

    def climb_constraints(self):
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
        eng_type  = self.engine.type
        cd_min    = self.aerodynamics.cd_min_clean 
        altitude  = self.climb.altitude   
        Vx        = self.climb.airspeed   
        Vy        = self.climb.climb_rate 
        delta_ISA = self.climb.delta_ISA
        AR        = self.geometry.aspect_ratio 
        W_S       = self.wing_loading  
  

        # Determine atmospheric properties at the altitude
        planet      = SUAVE.Analyses.Planets.Planet()
        atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmosphere.features.planet = planet.features

        atmo_values = atmosphere.compute_values(altitude,delta_ISA)
        rho         = atmo_values.density[0,0]
        p           = atmo_values.pressure[0,0]
        T           = atmo_values.temperature[0,0]
        a           = atmo_values.speed_of_sound[0,0]   

        T_W = np.zeros(len(W_S))
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            P_W        = np.zeros(len(W_S))
            etap       = self.propeller.climb_efficiency
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
                    cd_comp   = self.compressibility_drag(M,CL) 
                    cd_comp_e = self.compressibility_drag(M,0)  
                    cd0     = cd_min + cd_comp  

                    # Calculate Oswald efficiency
                    if self.aerodynamics.oswald_factor == 0:
                        e = self.oswald_efficiency(cd_min+cd_comp_e,M)

                    k     = 1/(np.pi*e*AR)
                    Vx    = np.sqrt(2/rho*W_S[i]*np.sqrt(k/(3*cd0)))
                    Mnew  = Vx/a
                    error = Mnew - M
                    M     = Mnew
                    q     = 0.5*rho*(M*a)**2 

                T_W[i] = Vy/Vx + q/W_S[i]*cd0 + k/q*W_S[i]
                P_W[i] = T_W[i]*Vx/etap

                if eng_type == 'turboprop':
                    P_W[i] = P_W[i] / self.normalize_turboprop_thrust(p,T) 
                elif eng_type == 'piston':
                    P_W[i] = P_W[i] / self.normalize_power_piston(rho)   
                elif eng_type == 'electric':
                    P_W[i] = P_W[i] / self.normalize_power_electric(rho)  

        else:
            M  = Vx/a                           
            q  = 0.5*rho*Vx**2      

            for i in range(len(W_S)):
                CL = W_S[i]/q

                # Calculate compressibility_drag
                cd_comp   = self.compressibility_drag(M,CL) 
                cd_comp_e = self.compressibility_drag(M,0)  
                cd0     = cd_min + cd_comp  

                # Calculate Oswald efficiency
                if self.aerodynamics.oswald_factor == 0:
                    e = self.oswald_efficiency(cd_min+cd_comp_e,M)

                k      = 1/(np.pi*e*AR)
                T_W[i] = Vy/Vx + q/W_S[i]*cd0 + k/q*W_S[i]
                T_W[i] = T_W[i] / self.normalize_gasturbine_thrust(eng_type,p,T,M,altitude,'climb')  

        # Pack outputs
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            if self.plot_units == 'US':
                constraint = 0.006 * P_W       # convert to hp/lb
            else:
                constraint = 9.81 * P_W/1000   # convert to kW/kg
        else:
            if self.plot_units != 'US':
                constraint = 9.81 * T_W         # convert to N/kg
            else:
                constraint = T_W 

        return constraint

    def ceiling_constraints(self):
        """Calculate thrust-to-weight ratios for the service ceiling

        Assumptions:
            Ceiling climb rate of 100 fpm 

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082

        Inputs:
            self.engine.type                    string
                 aerodynamics.cd_min_clean      [Unitless]
                 ceiling.altitude               [m]
                         airspeed               [m/s]
                 geometry.aspect_ratio          [Unitless]
                 wing_loading                   [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/sq m]

        Properties Used:

        """  

        # Unpack inputs
        eng_type  = self.engine.type
        cd_min    = self.aerodynamics.cd_min_clean
        altitude  = self.ceiling.altitude  
        delta_ISA = self.ceiling.delta_ISA 
        M         = self.ceiling.mach 
        AR        = self.geometry.aspect_ratio
        W_S       = self.wing_loading  

        # Determine atmospheric properties at the altitude
        planet      = SUAVE.Analyses.Planets.Planet()
        atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmosphere.features.planet = planet.features

        atmo_values = atmosphere.compute_values(altitude,delta_ISA)
        rho         = atmo_values.density[0,0]
        p           = atmo_values.pressure[0,0]
        T           = atmo_values.temperature[0,0]
        a           = atmo_values.speed_of_sound[0,0] 
        Vceil       = M * a
        q           = 0.5 * rho * Vceil**2 

        T_W = np.zeros(len(W_S))
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            P_W  = np.zeros(len(W_S))
            etap = self.propeller.cruise_efficiency
            if etap == 0:
                raise ValueError('Warning: Set the propeller efficiency at the ceiling')
        for i in range(len(W_S)):
            CL = W_S[i]/q

            # Calculate compressibility_drag
            cd_comp   = self.compressibility_drag(M,CL) 
            cd_comp_e = self.compressibility_drag(M,0)  
            cd0     = cd_min + cd_comp  

            # Calculate Oswald efficiency
            if self.aerodynamics.oswald_factor == 0:
                e = self.oswald_efficiency(cd_min+cd_comp_e,M)

            k = 1/(np.pi*e*AR)

            T_W[i] = 0.508/(np.sqrt(2/rho*W_S[i]*np.sqrt(k/(3*cd0)))) + 4*np.sqrt(k*cd0/3)

        # Convert thrust to power (for propeller-driven engines) and normalize the results wrt the Sea-level       
        if eng_type != 'turbofan' and eng_type != 'turbojet':

            P_W = T_W*Vceil/etap

            if eng_type   == 'turboprop':
                P_W = P_W / self.normalize_turboprop_thrust(p,T) 
            elif eng_type == 'piston':
                P_W = P_W / self.normalize_power_piston(rho) 
            elif eng_type == 'electric':
                P_W = P_W / self.normalize_power_electric(rho)  
        else:
            T_W = T_W / self.normalize_gasturbine_thrust(eng_type,p,T,M,altitude,'ceiling')           

        # Pack outputs
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            if self.plot_units == 'US':
                constraint = 0.006 * P_W       # convert to hp/lb
            else:
                constraint = 9.81 * P_W/1000    # convert to kW/kg
        else:
            if self.plot_units != 'US':
                constraint = 9.81 * T_W         # convert to N/kg
            else:
                constraint = T_W 

        return constraint

    def OEI_climb_constraints(self):
        """Calculate thrust-to-weight ratios for the 2nd segment OEI climb

        Assumptions:
            Climb CL and CD are similar to the take-off CL and CD
            Based on FAR Part 25

        Source:
            L. Loftin,Subsonic Aircraft: Evolution and the Matching of Size to Performance, NASA Ref-erence Publication 1060, August 1980

        Inputs:
            self.engine.type                    string
                 takeoff.runway_elevation       [m]
                         delta_ISA              [K]
                 aerodynamics.cd_takeoff        [Unitless]
                              cl_takeoff        [Unitless]
                OEI_climb.climb_speed_factor    [Unitless]
                engine.number                   [Unitless]
                wing_loading                    [N/m**2]

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:

        """  
        # Unpack inputs
        eng_type  = self.engine.type
        altitude  = self.takeoff.runway_elevation     
        delta_ISA = self.takeoff.delta_ISA   
        cd_TO     = self.aerodynamics.cd_takeoff 
        cl_TO     = self.aerodynamics.cl_takeoff
        eps       = self.OEI_climb.climb_speed_factor 
        Ne        = self.engine.number
        W_S       = self.wing_loading 

        # determine the flight path angle
        if Ne == 2:
            gamma = 1.203 * Units.degrees
        elif Ne == 3:
            gamma = 1.3748 * Units.degrees
        elif Ne == 4:
            gamma = 1.5466 * Units.degrees
        else:
            gamma = 0.0

        # Determine atmospheric properties at the altitude
        planet      = SUAVE.Analyses.Planets.Planet()
        atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmosphere.features.planet = planet.features

        atmo_values = atmosphere.compute_values(altitude,delta_ISA)
        rho         = atmo_values.density[0,0]
        p           = atmo_values.pressure[0,0]
        T           = atmo_values.temperature[0,0]
        a           = atmo_values.speed_of_sound[0,0] 
        Vcl         = eps * np.sqrt(2*W_S/(rho*cl_TO))
        M           = Vcl/a

        L_D = cl_TO/cd_TO

        T_W    = np.zeros(len(W_S))
        T_W[:] = Ne/((Ne-1)*L_D)+gamma

        if eng_type != 'turbofan' and eng_type != 'turbojet':
            P_W  = np.zeros(len(W_S))
            etap = self.propeller.cruise_efficiency
            if etap == 0:
                raise ValueError('Warning: Set the propeller efficiency during the OEI 2nd climb segment')

            for i in range(len(W_S)):
                P_W[i] = T_W[i]*Vcl[i]/etap

                if eng_type   == 'turboprop':
                    P_W[i] = P_W[i] / self.normalize_turboprop_thrust(p,T)
                elif eng_type == 'piston':
                    P_W[i] = P_W[i] / self.normalize_power_piston(rho) 
                elif eng_type == 'electric':
                    P_W[i] = P_W[i] / self.normalize_power_electric(rho) 
        else:
            for i in range(len(W_S)):
                T_W[i] = T_W[i] / self.normalize_gasturbine_thrust(eng_type,p,T,M[i],altitude,'OEIclimb')  

        # Pack outputs
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            if self.plot_units == 'US':
                constraint = 0.006 * P_W       # convert to hp/lb
            else:
                constraint = 9.81 * P_W/1000    # convert to kW/kg
        else:
            if self.plot_units != 'US':
                constraint = 9.81 * T_W         # convert to N/kg
            else:
                constraint = T_W

        return constraint

    def landing_constraints(self):
        """Calculate the landing wing loading 

        Assumptions:
        None

        Source:
            L. Loftin,Subsonic Aircraft: Evolution and the Matching of Size to Performance, NASA Ref-erence Publication 1060, August 1980

        Inputs:
            self.aerodynamics.cl_max_landing    [Unitless]
                 landing.ground_roll            [m]
                         runway_elevation       [m]
                         approach_speed_factor  [Unitless]
                         delta_ISA              [K]
                 engine.type                    string

        Outputs:
            constraints.T_W                     [Unitless]
                        W_S                     [N/m**2]

        Properties Used:

        """  

        # Unpack inputs
        cl_max    = self.aerodynamics.cl_max_landing
        Sg        = self.landing.ground_roll
        altitude  = self.landing.runway_elevation     
        eps       = self.landing.approach_speed_factor
        delta_ISA = self.landing.delta_ISA
        eng_type  = self.engine.type

        # Estimate maximum lift coefficient
        if cl_max == 0:
            cl_max = self.estimate_max_lift(self.geometry.high_lift_type_takeoff)

        # Estimate the approach speed
        if eng_type != 'turbofan' and eng_type != 'turbojet':
            kappa = 0.7
            Vapp  = np.sqrt(3.8217*Sg/kappa+49.488)
        else:
            kappa = 0.6
            Vapp  = np.sqrt(2.6319*Sg/kappa+458.8)

        # Determine atmospheric properties at the altitude
        planet      = SUAVE.Analyses.Planets.Planet()
        atmosphere  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
        atmosphere.features.planet = planet.features

        atmo_values = atmosphere.compute_values(altitude,delta_ISA)
        rho         = atmo_values.density[0,0]

        W_S        = np.zeros(2)
        
        cl_app     = cl_max / eps**2
        W_S[:]     = 0.5*rho*Vapp**2*cl_app
        constraint = np.array([0,30])

        # Convert the wing loading depinting on the user input
        if self.plot_units == 'US':
            W_S = W_S * 0.020885              # convert to lb/sq ft
        else:
            W_S = W_S/9.81                    # convert to kg/sq m

        return W_S,constraint



    def oswald_efficiency(self,cdmin,mach):
        """Calculate an average Oswald efficiencies based on different methods

        Assumptions:
        None

        Source:
           M. Nita, D.Scholtz, 'Estimating the Oswald factor from basic aircraft geometrical parameters',
            Deutscher Luft- und Raumfahrtkongress 20121DocumentID: 281424

        Inputs:
            self.geometry.taper                 [Unitless]
                 aspect_ratio                   [Unitless]
                 sweep_quarter_chord            [radians]
                 aerodynamics.fuselage_factor   [Unitless]
                              viscous_factor    [Unitless]

        Outputs:
            e          [Unitless]

        Properties Used:

        """  

        # Unpack inputs
        taper = self.geometry.taper
        AR    = self.geometry.aspect_ratio
        sweep = self.geometry.sweep_quarter_chord / Units.degrees
        kf    = self.aerodynamics.fuselage_factor 
        K     = self.aerodynamics.viscous_factor 

        dtaper    = -0.357+0.45*np.exp(-0.0375*np.abs(sweep))
        eff_taper = taper - dtaper
        f_taper   = 0.0524*eff_taper**4-0.15*eff_taper**3+0.1659*eff_taper**2-0.0706*eff_taper+0.0119
        u         = 1 / (1+f_taper*AR)
        P         = K*cdmin
        Q         = 1/(kf*u)

        #if mach > 0.3:
        #    keM = -0.001521*(mach/0.3-1)**10.82+1
        #else:
        keM = 1
        
        e = keM/(Q+P*np.pi*AR)


        return e


    def estimate_max_lift(self,highlift_type):
        """Estimates the wing maximum lift coefficient

        Assumptions:
        None

        Source:
            D. Scholz, "Aircraft Design lecture notes", https://www.fzt.haw-hamburg.de/pers/Scholz/HOOU/AircraftDesign_5_PreliminarySizing.pdf

        Inputs:
            self.geometry.sweep_quarter_chord       [degrees]

        Outputs:
            CLmax       [Unitless]

        Properties Used:

        """  

        # Unpack inputs
        sweep = self.geometry.sweep_quarter_chord / Units.degrees

        if highlift_type == None: # No Flaps
            return -0.0002602 * sweep**2 -0.0008614 * sweep + 1.51    
            
        elif highlift_type == 'plain':  # Plain 
            return -0.0002823 * sweep**2 -0.000141 * sweep + 1.81  
            
        elif highlift_type == 'single-slotted': # Single-Slotted 
            return -0.0002599 * sweep**2 -0.002727 * sweep + 2.205      
      
        elif highlift_type == 'single-slotted Fowler':   # Fowler
            return -0.000283 * sweep**2 -0.003897 * sweep + 2.501  
      
        elif highlift_type == 'double-slotted fixed vane':    # Double-Slotted            
            return -0.0002574 * sweep**2 -0.007129 * sweep + 2.735  
            
        elif highlift_type == 'double-slotted Fowler':    # Double-slotted Fowler with Slats
            return -0.0002953 * sweep**2 -0.006719 * sweep + 3.014 
            
        elif highlift_type == 'triple-slotted Fowler':    # Triple-slotted Fowler with Slats
            return -0.0003137 * sweep**2 -0.008903 * sweep + 3.416 
              
        else:
            highliftmsg = "High-lift device type must be specified in the design dictionary."
            raise ValueError(highliftmsg)


    def compressibility_drag(self,mach,cl):
        """Estimates drag due to compressibility

        Assumptions:
            Subsonic to low transonic
            Supercritical airfoil

        Source:
            adg.stanford.edu (Stanford AA241 A/B Course Notes)

        Inputs:
            mach                                    [Unitless]
            cl                                      [Unitless]
            self.geometry.sweep_quarter_chord       [radians]
                          thickness_to_chord        [Unitless]

        Outputs:
            cd_c           [Unitless]

        Properties Used:

        """  

        # Unpack inputs
        sweep = self.geometry.sweep_quarter_chord
        t_c   = self.geometry.thickness_to_chord

        cos_sweep = np.cos(sweep)

        # get effective Cl and sweep
        tc = t_c /(cos_sweep)
        cl = cl / (cos_sweep*cos_sweep)

        # compressibility drag based on regressed fits from AA241
        mcc_cos_ws = 0.922321524499352       \
                    - 1.153885166170620*tc    \
                    - 0.304541067183461*cl    \
                    + 0.332881324404729*tc*tc \
                    + 0.467317361111105*tc*cl \
                    + 0.087490431201549*cl*cl
        
        # crest-critical mach number, corrected for wing sweep
        mcc = mcc_cos_ws / cos_sweep

        # divergence ratio
        mo_mc = mach/mcc
    
        # compressibility correlation, Shevell
        dcdc_cos3g = 0.0019*mo_mc**14.641
    
        # compressibility drag
        cd_c = dcdc_cos3g * cos_sweep*cos_sweep*cos_sweep


        return cd_c
    

    def normalize_power_piston(self,density):
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

    def normalize_power_electric(self,density):
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

    def normalize_gasturbine_thrust(self,engine_type,pressure,temperature,mach,altitude,seg_tag):
        """Altitude correction for engines that feature a gas turbine

        Assumptions:
            N/A

        Source:
            S. Gudmundsson "General Aviation aircraft Design. Applied methods and procedures", Butterworth-Heinemann (6 Nov. 2013), ISBN - 0123973082
            Clarkson Universitty AE429 Lecture notes (https://people.clarkson.edu/~pmarzocc/AE429/AE-429-6.pdf)


            For the Howe method, an afterburner factor of 1.2 is used
        Inputs:
            engine_type             string
            pressure                [Pa]
            temperature             [K]
            mach                    [Unitless]

        Outputs:
            thrust_ratio           [Unitless]

        Properties Used:       
        """        
        # Unpack inputs
        TR = self.engine.throttle_ratio

        # Calculate atmospheric ratios
        theta = temperature/288 * (1+0.2*mach**2)
        delta = pressure/101325 * (1+0.2*mach**2)**3.5 
        
        if engine_type == 'turbojet':

            if self.engine.afterburner == True:
                if theta <= TR:
                    thrust_ratio = delta * (1 - 0.3 * (theta - 1) - 0.1 * np.sqrt(mach))
                else:
                    thrust_ratio = delta * (1 - 0.3 * (theta - 1) - 0.1 * np.sqrt(mach) - 1.5 * (theta - TR) / theta)
            else:
                if theta <= TR:
                    thrust_ratio =  delta * 0.8 * (1 - 0.16 * np.sqrt(mach))
                else:
                    thrust_ratio =  delta * 0.8 * (1 - 0.16 * np.sqrt(mach) - 24 * (theta - TR) / ((9 + mach) * theta))

        elif engine_type == 'turbofan':
            
            if self.engine.method == 'Mattingly':
                if self.engine.bypass_ratio < 1:
                    if self.engine.afterburner == True:
                        if theta <= TR:
                            thrust_ratio =  delta
                        else:
                            thrust_ratio = delta * (1 - 3.5 * (theta - TR) / theta)    
                    else: 
                        if theta <= TR:
                            thrust_ratio =  0.6 * delta
                        else:
                            thrust_ratio = 0.6 * delta * (1 - 3.8 * (theta - TR) / theta)
                else:
                    if theta <= TR:
                        thrust_ratio =  delta * (1 - 0.49 * np.sqrt(mach))
                    else:
                        thrust_ratio =  delta * (1 - 0.49 * np.sqrt(mach) - 3 * (theta - TR)/(1.5 + mach))
            
            elif self.engine.method == 'Scholz':
                if seg_tag != 'takeoff' and seg_tag != 'OEIclimb':
                    BPR          = self.engine.bypass_ratio
                    thrust_ratio = (0.0013*BPR-0.0397)*altitude/1000-0.0248*BPR+0.7125
                else:
                    thrust_ratio = 1.0  
            
            elif self.engine.method == 'Howe':
                BPR = self.engine.bypass_ratio
                if self.engine.bypass_ratio <= 1:
                    s = 0.8
                    if mach < 0.4:
                        if self.engine.afterburner == False:
                            K   = [1.0, 0.0, -0.2, 0.07]
                            kab = 1.0
                        else:
                            K   = [1.32, 0.062, -0.13, -0.27]  
                            kab = 1.2                    
                    elif mach < 0.9:
                        if self.engine.afterburner == False:     
                            K   = [0.856, 0.062, 0.16, -0.23]   
                            kab = 1.0
                        else:
                            K   = [1.17, -0.12, 0.25, -0.17]
                            kab = 1.2
                    elif mach < 2.2:
                        if self.engine.afterburner == False:  
                            K   = [1.0, -0.145, 0.5, -0.05]
                            kab = 1.0
                        else:
                            K   = [1.4, 0.03, 0.8, 0.4]
                            kab = 1.2
                    else:
                        raise ValueError("the Mach number is too high for the Howe method. The maximum possible value is 2.2\n")  
                
                elif self.engine.bypass_ratio > 3 and self.engine.bypass_ratio < 6:
                    s = 0.7
                    if self.engine.afterburner == False:
                        kab = 1.0
                    else:
                        kab = 1.2

                    if mach < 0.4:
                        K   = [1.0, 0.0, -0.6, -0.04]
                    elif mach < 0.9:
                        K = [0.88, -0.016, -0.3, 0.0]
                    else:
                        raise ValueError("the Mach number is too high for the Howe method. The maximum possible value is 0.9\n")  
                       
                else:
                    s = 0.7
                    if self.engine.afterurner == False:
                        kab = 1.0
                    else:
                        kab = 1.2

                    if mach < 0.4:
                        K = [1.0, 0.0, -0.595, -0.03]
                    elif mach < 0.9:
                        K = [0.89, -0.014, -0.3, 0.005]
                    else:
                        raise ValueError("the Mach number is too high for the Howe method. The maximum possible value is 0.9\n")  
                                                                           
                sigma        = (1 -2.2558e-5*altitude)**4.2561 
                thrust_ratio = kab * (K[0]+K[1]*BPR+(K[2]+K[3]*BPR)*mach)*sigma**s

            elif self.engine.method == 'Bartel':
                BPR   = self.engine.bypass_ratio
                p_pSL = (1-2.2558e-5*altitude)**5.2461
                A     = -0.4327*p_pSL**2+1.3855*p_pSL+0.0472
                Z     = 0.9106*p_pSL**3-1.7736*p_pSL**2+1.8697*p_pSL
                X     = 0.1377*p_pSL**3-0.4374*p_pSL**2+1.3003*p_pSL
                G0    = 0.0603*BPR+0.6337

                thrust_ratio = A - (0.377*(1+BPR))/(((1+0.82*BPR)*G0)**0.5)*Z*mach+(0.23+0.19*BPR**0.5)*X*mach**2

            else:
                raise ValueError("Enter a correct thrust normalization method\n")  

        return thrust_ratio


    def normalize_turboprop_thrust(self,pressure,temperature):
        """Altitude correction for a turboprop engine

        Assumptions:
            N/A

        Source:
            Clarkson Universitty AE429 Lecture notes (https://people.clarkson.edu/~pmarzocc/AE429/AE-429-6.pdf)

        Inputs:
            engine_type             string
            pressure                [Pa]
            temperature             [K]
            mach                    [Unitless]

        Outputs:
            thrust_ratio           [Unitless]

        Properties Used:       
        """          

        density      = pressure/(287*temperature)
        thrust_ratio = (density/1.225)**0.7

        return thrust_ratio
