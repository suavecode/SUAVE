# test_aerodynamics
#
# Created:  Tim MacDonald - 09/09/14
# Modified: Tim MacDonald - 09/10/14
# Modified: Anil Variyar - February 2015

import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data

import numpy as np
import pylab as plt

from test_mission_B737 import vehicle_setup


def main():    
    '''
    
    Rough Idea - 
    
    initialize a vehicle
    
    initialize a SUAVE.Analyses.Aerodynamics.Markup()
       only a linear lift slope model, no surrogate
    
    run an array of angle of attacks
    plot the polars
    
    '''

    # initialize the vehicle
    vehicle = vehicle_setup() 
    
    # initalize the aero model
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.initialize()
    
    n_control_points = 11
    angle_of_attacks = np.linspace(-10,10,n_control_points) * Units.deg
    
    state = SUAVE.Analyses.Missions.Segments.Conditions.State()
    state.conditions = SUAVE.Analyses.Missions.Segments.Conditions.Aerodynamics()
    
    
    state.expand_rows(n_control_points)
    
    state.conditions.aerodynamics.angle_of_attack[:,0] = angle_of_attacks
    state.conditions.freestream.velocity[:,0] = 10.0
    # populate as needed for model
    
    results = aerodynamics.evaluate(state)
    
    print results
    
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #print 'Aero regression test passed!'
    
    ## --------------------------------------------------------------------
    ## Drag Polar
    ## --------------------------------------------------------------------  
    
    ## Cruise conditions (except Mach number)
    #conditions.freestream.mach_number = np.array([0.2]*test_num)
    #conditions.freestream.density = np.array([0.3804534]*test_num)
    #conditions.freestream.viscosity = np.array([1.43408227e-05]*test_num)
    #conditions.freestream.temperature = np.array([218.92391647]*test_num)
    #conditions.freestream.pressure = np.array([23908.73408391]*test_num)
    
    #compute_aircraft_lift(conditions, configuration, geometry) # geometry is third variable, not used
    #CL = conditions.aerodynamics.lift_breakdown.total    
    
    #compute_aircraft_drag(conditions, configuration, geometry)
    #CD = conditions.aerodynamics.drag_breakdown.total
    
    #plt.figure("Drag Polar")
    #axes = plt.gca()     
    #axes.plot(CD,CL,'bo-')
    #axes.set_xlabel('$C_D$')
    #axes.set_ylabel('$C_L$')
    
    
    #plt.show(block=True) # here so as to not block the regression test
    
    
    
    
    
    
    
        #lift_model = vehicle.configs.cruise.aerodynamics_model.configuration.surrogate_models.lift_coefficient
        
        #wing_lift = lift_model(AoA)
        
        #wing_lift_r = np.array([-0.79420805, -0.56732369, -0.34043933, -0.11355497,  0.11332939,
                                #0.34021374,  0.5670981 ,  0.79398246,  1.02086682,  1.24775117,
                                #1.47463553])
        
        #surg_test = np.abs((wing_lift-wing_lift_r)/wing_lift)
        
        #print 'Surrogate Test Results \n'
        #print surg_test
        
        #assert(np.max(surg_test)<1e-4), 'Aero regression failed at surrogate test'
    
        
        ## --------------------------------------------------------------------
        ## Initialize variables needed for CL and CD calculations
        ## Use a seeded random order for values
        ## --------------------------------------------------------------------
        
        #random.seed(1)
        #Mc = np.linspace(0.05,0.9,test_num)
        #random.shuffle(Mc)
        #rho = np.linspace(0.3,1.3,test_num)
        #random.shuffle(rho)
        #mu = np.linspace(5*10**-6,20*10**-6,test_num)
        #random.shuffle(mu)
        #T = np.linspace(200,300,test_num)
        #random.shuffle(T)
        #pressure = np.linspace(10**5,10**6,test_num)
    
        
        #conditions = Data()
        
        #conditions.freestream = Data()
        #conditions.freestream.mach_number = Mc
        #conditions.freestream.density = rho
        #conditions.freestream.viscosity = mu
        #conditions.freestream.temperature = T
        #conditions.freestream.pressure = pressure
        
        #conditions.aerodynamics = Data()
        #conditions.aerodynamics.angle_of_attack = AoA
        #conditions.aerodynamics.lift_breakdown = Data()
        
        #configuration = vehicle.configs.cruise.aerodynamics_model.configuration
        
        #conditions.aerodynamics.drag_breakdown = Data()
    
        #geometry = Data()
        #for k in ['fuselages','wings','propulsors']:
            #geometry[k] = deepcopy(vehicle[k])    
        #geometry.reference_area = vehicle.reference_area  
        ##geometry.wings[0] = Data()
        ##geometry.wings[0].vortex_lift = False
        
        ## --------------------------------------------------------------------
        ## Test compute Lift
        ## --------------------------------------------------------------------
        
        #compute_aircraft_lift(conditions, configuration, geometry) 
        
        #lift = conditions.aerodynamics.lift_breakdown.total
        #lift_r = np.array([-2.07712357, -0.73495391, -0.38858687, -0.1405849 ,  0.22295808,
                           #0.5075275 ,  0.67883681,  0.92787301,  1.40470556,  2.08126751,
                           #1.69661601])
        
        #lift_test = np.abs((lift-lift_r)/lift)
        
        #print '\nCompute Lift Test Results\n'
        #print lift_test
            
        #assert(np.max(lift_test)<1e-4), 'Aero regression failed at compute lift test'    
        
        
        ## --------------------------------------------------------------------
        ## Test compute drag 
        ## --------------------------------------------------------------------
        
        #compute_aircraft_drag(conditions, configuration, geometry)
        
        ## Pull calculated values
        #drag_breakdown = conditions.aerodynamics.drag_breakdown
        
        ## Only one wing is evaluated since they rely on the same function
        #cd_c           = drag_breakdown.compressible['main_wing'].compressibility_drag
        #cd_i           = drag_breakdown.induced.total
        #cd_m           = drag_breakdown.miscellaneous.total
        #cd_m_fuse_base = drag_breakdown.miscellaneous.fuselage_base
        #cd_m_fuse_up   = drag_breakdown.miscellaneous.fuselage_upsweep
        #cd_m_nac_base  = drag_breakdown.miscellaneous.nacelle_base['turbo_fan']
        #cd_m_ctrl      = drag_breakdown.miscellaneous.control_gaps
        #cd_p_fuse      = drag_breakdown.parasite['fuselage'].parasite_drag_coefficient
        #cd_p_wing      = drag_breakdown.parasite['main_wing'].parasite_drag_coefficient
        #cd_tot         = drag_breakdown.total
        
        #(cd_c_r, cd_i_r, cd_m_r, cd_m_fuse_base_r, cd_m_fuse_up_r, cd_m_nac_base_r, cd_m_ctrl_r, cd_p_fuse_r, cd_p_wing_r, cd_tot_r) = reg_values()
        
        #drag_tests = Data()
        #drag_tests.cd_c = np.abs((cd_c-cd_c_r)/cd_c)
        #drag_tests.cd_i = np.abs((cd_i-cd_i_r)/cd_i)
        #drag_tests.cd_m = np.abs((cd_m-cd_m_r)/cd_m)
        ## Line below is not normalized since regression values are 0, insert commented line if this changes
        #drag_tests.cd_m_fuse_base = np.abs((cd_m_fuse_base-cd_m_fuse_base_r)) # np.abs((cd_m_fuse_base-cd_m_fuse_base_r)/cd_m_fuse_base)
        #drag_tests.cd_m_fuse_up   = np.abs((cd_m_fuse_up - cd_m_fuse_up_r)/cd_m_fuse_up)
        #drag_tests.cd_m_ctrl      = np.abs((cd_m_ctrl - cd_m_ctrl_r)/cd_m_ctrl)
        #drag_tests.cd_p_fuse      = np.abs((cd_p_fuse - cd_p_fuse_r)/cd_p_fuse)
        #drag_tests.cd_p_wing      = np.abs((cd_p_wing - cd_p_wing_r)/cd_p_wing)
        #drag_tests.cd_tot         = np.abs((cd_tot - cd_tot_r)/cd_tot)
        
        #print '\nCompute Drag Test Results\n'
        #print drag_tests
        
        #for i, tests in drag_tests.items():
            #assert(np.max(tests)<1e-4),'Aero regression test failed at ' + i
        
        #return conditions, configuration, geometry, test_num    