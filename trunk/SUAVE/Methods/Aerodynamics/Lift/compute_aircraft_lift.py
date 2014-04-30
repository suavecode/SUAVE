# compute_aircraft_lift.py
# 
# Created:  Anil V., Dec 2013
# Modified: Anil, Trent, Tarik, Feb 2014 
# Modified: Anil  April 2014 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
# suave imports

from SUAVE.Structure import Data
from SUAVE.Attributes import Units

from SUAVE.Attributes.Results import Result
from SUAVE.Attributes.Missions.Segments import Segment
#from SUAVE import Vehicle
from SUAVE.Components.Wings import Wing
from SUAVE.Components.Fuselages import Fuselage
from SUAVE.Components.Propulsors import Turbofan
from SUAVE.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Geometry.Two_Dimensional.Planform import fuselage_planform

#from SUAVE.Attributes.Aerodynamics.Aerodynamics_Surrogate import Aerodynamics_Surrogate
#from SUAVE.Attributes.Aerodynamics.Aerodynamics_Surrogate import Interpolation
from SUAVE.Attributes.Aerodynamics.Aerodynamics_1d_Surrogate import Aerodynamics_1d_Surrogate
from SUAVE.Methods.Aerodynamics.Drag import compute_aircraft_drag



from SUAVE.Attributes.Aerodynamics.Configuration   import Configuration
from SUAVE.Attributes.Aerodynamics.Conditions      import Conditions
from SUAVE.Attributes.Aerodynamics.Geometry        import Geometry


from SUAVE.Methods.Aerodynamics.Lift.weissenger_vortex_lattice import weissinger_vortex_lattice
#from SUAVE.Methods.Aerodynamics.Lift import compute_aircraft_lift
#from SUAVE.Methods.Aerodynamics.Drag import compute_aircraft_drag


# python imports
import os, sys, shutil
from copy import deepcopy
from warnings import warn

# package imports
import numpy as np
import scipy as sp

#def main():
    
    #test()
    
    #return
# ----------------------------------------------------------------------
#  The Function
# ----------------------------------------------------------------------

def compute_aircraft_lift(conditions,configuration,geometry=None):
    """ SUAVE.Methods.Aerodynamics.compute_aircraft_lift(conditions,configuration,geometry)
        computes the lift associated with an aircraft 
        
        Inputs:
            conditions - data dictionary with fields:
                mach_number - float or 1D array of freestream mach numbers
                angle_of_attack - floar or 1D array of angle of attacks
                
            configuration - data dictionary with fields:
                surrogate_models.lift_coefficient - a callable function or class 
                    with inputs of angle of attack and outputs of lift coefficent
                fuselage_lift_correction - the correction to fuselage contribution to lift
                    
            geometry - Not used
            
        
        Outputs:
            CL - float or 1D array of lift coefficients of the total aircraft
        
        Updates:
            conditions.lift_breakdown - stores results here
            
        Assumptions:
            surrogate model returns total incompressible lift due to wings
            prandtl-glaurert compressibility correction on this
            fuselage contribution to lift correction as a factor
        
    """    
   
    # unpack
    fus_correction = configuration.fuselage_lift_correction
    Mc             = conditions.mach_number
    AoA            = conditions.angle_of_attack
    
    # the lift surrogate model for wings only
    wings_lift_model = configuration.surrogate_models.lift_coefficient
    
    # pack for interpolate
    X_interp = np.array([AoA]).T
    #X_interp = AoA
    
    # interpolate
    wings_lift = wings_lift_model(X_interp)  
    
    
    
    # compressibility correction
    compress_corr = 1./(np.sqrt(1.-Mc**2.))
    
    # correct lift
    wings_lift_comp = wings_lift * compress_corr
    
    # total lift, accounting one fuselage
    aircraft_lift_total = wings_lift_comp * fus_correction 
    
    #print aircraft_lift_total
    # store to results
    lift_results = Result(
        total                = aircraft_lift_total ,
        incompressible_wings = wings_lift          ,
        compressible_wings   = wings_lift_comp     ,
        compressibility_correction_factor = compress_corr  ,
        fuselage_correction_factor        = fus_correction ,
    )
    #conditions.lift_breakdown.update( lift_results )    #update
    
    conditions.lift_coefficient= aircraft_lift_total
    conditions.clean_wing_lift[0] = wings_lift_comp
    #conditions.clean_wing_lift[1] = 0.0
    #conditions.clean_wing_lift[2] = 0.0
    


    return aircraft_lift_total


# ----------------------------------------------------------------------
#   Module Tests
# ----------------------------------------------------------------------
# this will run from command line, put simple tests for your code here
#def test():
    
    #Seg=Segment()
    #aircraft=SUAVE.Vehicle()
    #aircraft.tag="aero_test"
    #aircraft.Sref=125.00
    
    #Wing1=Wing(tag = 'Wing1')
    #Wing2=Wing(tag = 'Wing2')
    #Wing3=Wing(tag = 'Wing3')

    
    
    #Seg.p = 23900.0            # Pa
    #Seg.T = 218.0            # K
    #Seg.rho = 0.41          # kg/m^3
    #Seg.mew = 1.8*10**-5          # Ps-s
    #Seg.M = 0.8            # dimensionless
   
    #Wing1.sref=124.862
    #Wing1.ar = 9
    #Wing1.span =35.66
    #Wing1.sweep=25*np.pi/180
    #Wing1.symmetric = True
    #Wing1.t_c = 0.1
    #Wing1.taper= 0.16
    
     
    #wing_planform(Wing1)
    #Wing1.chord_mac= 12.5
    #Wing1.S_exposed=0.8*Wing1.area_wetted
    #Wing1.S_affected=0.6*Wing1.area_wetted 
    #Wing1.Cl = 0.3
    #Wing1.e = 0.9
    #Wing1.twist_rc =3.0*np.pi/180
    #Wing1.twist_tc =-1.0*np.pi/180
 
   
    #aircraft.append_component(Wing1)
  
    
    #Wing2.sref=32.488
    #Wing2.ar = 6.16
    #Wing2.span =100
    #Wing2.sweep=30*np.pi/180
    #Wing2.symmetric = True
    #Wing2.t_c = 0.08
    #Wing2.taper= 0.4
    
    
    
    
    #wing_planform(Wing2)
    ##Wing2.chord_mac = 12.5
    #Wing2.chord_mac= 8.0
    #Wing2.S_exposed=0.8*Wing2.area_wetted
    #Wing2.S_affected=0.6*Wing2.area_wetted     
    #Wing2.Cl = 0.2
    #Wing2.e = 0.9
    #Wing2.twist_rc =3.0*np.pi/180
    #Wing2.twist_tc =-1.0*np.pi/180   
  
    #aircraft.append_component(Wing2)
    
 
    
    
    #Wing3.sref=32.488
    #Wing3.ar = 1.91
    #Wing3.span =100
    #Wing3.sweep=25*np.pi/180
    #Wing3.symmetric = False
    #Wing3.t_c = 0.08
    #Wing3.taper= 0.25
       
    #wing_planform(Wing3)
    #Wing3.chord_mac= 8.0
    #Wing3.S_exposed=0.8*Wing3.area_wetted
    #Wing3.S_affected=0.6*Wing3.area_wetted     
    #Wing3.Cl = 0.002  
    #Wing3.e = 0.9
    #Wing3.twist_rc =0.0*np.pi/180
    #Wing3.twist_tc =0.0*np.pi/180       
        
    #aircraft.append_component(Wing3)
    

   
    #fus=Fuselage(tag = 'fuselage1')
    
    #fus.num_coach_seats = 200
    #fus.seat_pitch = 1
    #fus.seats_abreast = 6
    #fus.fineness_nose = 1.6
    #fus.fineness_tail =  2
    #fus.fwdspace = 6
    #fus.aftspace = 5
    #fus.width  =4
    #fus.height =4   
    
    #fuselage_planform(fus)
    #aircraft.append_component(fus)


    #turbofan=Turbofan()
    #turbofan.nacelle_dia= 4.0
    #aircraft.append_component(turbofan)


    ##wing_aero = SUAVE.Attributes.Aerodynamics.Fidelity_Zero()
    
    ##self.tag = 'Fidelity_Zero'
      
    #geometry      = Geometry()
    
    #configuration = Configuration()
    
    #conditions =Conditions()
    
    ## correction factors
    #configuration.fuselage_lift_correction           = 1.14
    #configuration.trim_drag_correction_factor        = 1.02
    #configuration.wing_parasite_drag_form_factor     = 1.1
    #configuration.fuselage_parasite_drag_form_factor = 2.3
    #configuration.aircraft_span_efficiency_factor    = 0.78
    
    ## vortex lattice configurations
    #configuration.number_panels_spanwise  = 5
    #configuration.number_panels_chordwise = 1
    
    #Seg.p = 23900.0            # Pa
    #Seg.T = 218.0            # K
    #Seg.rho = 0.41          # kg/m^3
    #Seg.mew = 1.8*10**-5          # Ps-s
    #Seg.M = 0.8            # dimensionless
    
    #conditions.mach_number=Seg.M
    #conditions.density=Seg.rho
    #conditions.viscosity=Seg.mew
    #conditions.temperature = Seg.T  
    #conditions.pressure=Seg.p
    ##conditions.angle_of_attack    
    
    #conditions_table = Conditions(
        #angle_of_attack = np.linspace(-10., 10., 5) * Units.deg ,
    #)
    
    #models = Data()    
    
    
    ##wing_aero.initialize(aircraft)

    ##conditions_table = self.conditions_table
    ##geometry         = self.geometry
    ##configuration    = self.configuration
    ##
    #AoA = conditions_table.angle_of_attack
    #n_conditions = len(AoA)
    
    #vehicle=aircraft
    
    #print 'nacelle dia', vehicle.Propulsors[0].nacelle_dia

    ## copy geometry
    #for k in ['Fuselages','Wings','Propulsors']:
        #geometry[k] = deepcopy(vehicle[k])
        

    
    ## reference area
    #geometry.reference_area = vehicle.Sref
    
    ## arrays
    #CL  = np.zeros_like(AoA)
    
    ## condition input, local, do not keep
    #konditions = Conditions()
    
    #konditions=conditions    
    
    ## calculate aerodynamics for table
    #for i in xrange(n_conditions):
        
        ## overriding conditions, thus the name mangling
        #konditions.angle_of_attack = AoA[i]
        
        ## these functions are inherited from Aerodynamics() or overridden
        #CL[i] = calculate_lift_vortex_lattice(konditions, configuration, geometry)    
    
    #conditions_table.lift_coefficient = CL
    
    ##conditions_table = self.conditions_table
    #AoA_data = conditions_table.angle_of_attack
    ##
    #CL_data  = conditions_table.lift_coefficient
    
    ## pack for surrogate
    #X_data = np.array([AoA_data]).T        
    
    ## assign models
    #Interpolation = Aerodynamics_1d_Surrogate.Interpolation(X_data,CL_data)
    
    ##print len(CL_data)
    ##models.lift_coefficient = Interpolation(X_data,CL_data)
    #models.lift_coefficient = Interpolation
    
    ## assign to configuration
    #configuration.surrogate_models = models    
    
    #CL = compute_aircraft_lift(conditions,configuration,geometry)
            
            ## drag computes second
    #CD = compute_aircraft_drag(conditions,configuration,geometry)

    
    ##aircraft.Aerodynamics = wing_aero 

    
    ##[Cd,Cl]=aircraft.Aerodynamics(0,Seg)
  
    #print 'Aerodynamics module test script'
    #print 'aircraft Cl' , CL
    #print 'aircraft Cd' , CD
  
  
    #return



    
#def calculate_lift_vortex_lattice(conditions,configuration,geometry):
    #""" calculate total vehicle lift coefficient by vortex lattice
    #"""
    
    ## unpack
    ##vehicle_reference_area = geometry.Sref
    #vehicle_reference_area = geometry.reference_area
    #total_lift = 0.0
    
    #for wing in geometry.Wings.values():
        ##print 'wing span' , wing.sweep
        #[wing_lift,wing_induced_drag] = weissinger_vortex_lattice(conditions,configuration,wing)
        

        #total_lift += wing_lift * wing.sref / vehicle_reference_area
    
    #return total_lift
    
    
    
####: class Aerodynamics_Surrogate()

###test()

if __name__ == '__main__':   
    #test()
    raise RuntimeError , 'module test failed, not implemented'


#-------runn this caase  - have a local test case---------------------

