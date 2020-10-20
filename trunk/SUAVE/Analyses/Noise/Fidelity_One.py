## @ingroup Analyses-Noise
# Fidelity_One.py
#
# Created:  
# Modified: Feb 2016, A. Wendorff
# Modified: Oct 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE 
from SUAVE.Core import Data , Units
from .Noise     import Noise 

# noise imports 
from SUAVE.Methods.Noise.Fidelity_One.Airframe    import noise_airframe_Fink
from SUAVE.Methods.Noise.Fidelity_One.Engine      import noise_SAE 
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import pnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_tone_correction
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import epnl_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_certification_limits
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import noise_geometric
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools import SPL_arithmetic

from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_low_fidelity import propeller_low_fidelity
from SUAVE.Methods.Noise.Fidelity_One.Propeller.propeller_noise_sae    import propeller_noise_sae 

# package imports
import numpy as np

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Noise
class Fidelity_One(Noise):
    
    """ SUAVE.Analyses.Noise.Fidelity_One()
    
        The Fidelity One Noise Analysis Class
        
            Assumptions:
            None
            
            Source:
            N/A
    """
    
    def __defaults__(self):
        
        """ This sets the default values for the analysis.
        
                Assumptions:
                Ground microphone angles start in front of the aircraft (0 deg) and sweep in a lateral direction 
                to the starboard wing and around to the tail (180 deg)
                
                Source:
                N/A
                
                Inputs:
                None
                
                Output:
                None
                
                Properties Used:
                N/A
        """
        
        # Initialize quantities    
        self.harmonics                     = np.empty(shape=[0, 1])
        
        settings                           = self.settings
        settings.propeller_SAE_noise_model = False 
        settings.flyover                   = False    
        settings.approach                  = False
        settings.sideline                  = False
        settings.mic_x_position            = 0    
        settings.ground_microphone_angles  = np.array([45. , 30. , 15. , 10. , 5. , 0.1 , -0.1 , -5 , -10 , -15, -30 , -45])*Units.degrees
        #np.array([0.1,15.,30.,45.,60.,75.,90.1,105.,120.,135.,150.,165., 179.9])*Units.degrees       
        
        return
            
    def evaluate_noise(self,segment):
        """ Process vehicle to setup geometry, condititon and configuration
    
        Assumptions:
        None
    
        Source:
        N/4
    
        Inputs:
        conditions - DataDict() of aerodynamic conditions
        results    - DataDict() of moment coeffients and stability and body axis derivatives
    
        Outputs:
        None
    
        Properties Used:
        self.geometry
        """         
    
        # unpack
        harmonics     = self.harmonics
        config        = segment.analyses.noise.geometry
        analyses      = segment.analyses
        settings      = self.settings 
        conditions    = segment.state.conditions
        
    
        # unpack 
        alt         = -conditions.frames.inertial.position_vector[:,2]  
        dist        = conditions.frames.inertial.position_vector[:,0] 
        gma         = settings.ground_microphone_angles 
        
        dim_alt = len(alt)
        num_mic = len(gma)  
        
        angles   = np.repeat(np.atleast_2d(gma), dim_alt, axis = 0)
        altitude = np.repeat(np.atleast_2d(alt).T, num_mic, axis = 1)
        
        mic_locations        = np.zeros((dim_alt,num_mic,3)) 
        mic_locations[:,:,1] = np.tan(angles)*altitude  
        mic_locations[:,:,2] = altitude 
        
        conditions.noise.microphone_angles     = gma
        conditions.noise.microphone_locations  = mic_locations
        conditions.noise.number_of_microphones = num_mic
         
        ctrl_pts = len(altitude) 
        
        # create empty arrays for results  
        num_src            = len(config.propulsors) + 1 
        source_SPLs_dBA    = np.zeros((ctrl_pts,num_src,num_mic)) 
        source_EPNLs       = np.zeros((ctrl_pts,num_src,num_mic))
        source_SENELs      = np.zeros((ctrl_pts,num_src,num_mic))        
        total_SPL_dBA      = np.zeros((ctrl_pts,num_mic))  
        
        # loop for microphone locations 
        for mic_loc in range(num_mic):  
            # calcuate location and geometric angles of noise sources 
            geometric          = noise_geometric(segment,analyses,config,mic_loc)
            
            # make flag to skip if flaps not present        
            if 'flap' in config.wings.main_wing.control_surfaces:            
                airframe_noise     = noise_airframe_Fink(segment,analyses,config,mic_loc )  # (EPNL_total,SPL_total_history,SENEL_total)
                #source_EPNLs[:,0,mic_loc]  = airframe_noise[0]            
                #source_SPLs_dBA[:,0,mic_loc]   = airframe_noise[1]  # histroy in 0.5 time steps at 25 differnt frequencies  
                #source_SENELs[:,0,mic_loc] = airframe_noise[2]        
             
            # iterate through sources 
            si = 1 
            for source in conditions.noise.sources.keys():
                 
                if source  == 'turbofan':   
                    if bool(conditions.noise.sources[source].fan) and bool(conditions.noise.sources[source].core): 
                        config.propulsors[source].fan.rotation            = 0 # NEED TO UPDATE ENGINE MODEL WITH FAN SPEED in RPM
                        config.propulsors[source].fan_nozzle.noise_speed  = conditions.noise.sources.turbofan.fan.exit_velocity 
                        config.propulsors[source].core_nozzle.noise_speed = conditions.noise.sources.turbofan.core.exit_velocity 
                        engine_noise   = noise_SAE(source,segment,analyses,config)  # EPNL_total,SPL_total_history,SENEL_total
                        source_EPNLs[:,si,mic_loc]  = engine_noise[0]
                        #source_SPLs_dBA[:,si,mic_loc]   = engine_noise[1]
                        #source_SENELs[:,si,mic_loc] = engine_noise[2]                    
                        
                elif (source  == 'propeller') or (source   == 'rotor'): 
                    if bool(conditions.noise.sources[source]) == True : 
                        # Compute Propeller Noise 
                        if settings.propeller_SAE_noise_model:
                            propeller_noise = propeller_noise_sae(source,segment, ioprint = 0) #  (np.max(PNL_dBA), EPNdB_takeoff, EPNdB_landing, OASPL ,PNL )
                            source_EPNLs[:,si,mic_loc]  = propeller_noise[0]
                            #source_SPLs_dBA[:,si,mic_loc]   = propeller_noise[3]   
                            #source_SENELs[:,si,mic_loc] = propeller_noise[4]      
                        else:
                            propeller_noise = propeller_low_fidelity(source,segment,mic_loc,harmonics) 
                            #source_EPNLs[:,si,mic_loc]  = propeller_noise[0]
                            source_SPLs_dBA[:,si,mic_loc]   = propeller_noise.SPL_Hv_dBA
                            #source_SENELs[:,si,mic_loc] = propeller_noise[2]      
                        
                si += 1
                
            total_SPL_dBA[:,mic_loc]  = SPL_arithmetic(source_SPLs_dBA[:,:,mic_loc])
        
        conditions.noise.total_SPL_dBA = total_SPL_dBA
        return   

