## @ingroup Analyses-Energy
# Fidelity_Zero.py
#
# Created:   
# Modified: Oct 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from .Energy      import Energy 

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Energy
class Fidelity_Zero(Energy):
    
    """ SUAVE.Analyses.Energy.Fidelity_Zero()
    
        The Fidelity Zero Energy Analysis Class
        
            Assumptions:
            None
            
            Source:
            N/A
    """
    
    def __defaults__(self):
        
        """ This sets the default values for the analysis.
        
            Assumptions:
            None
            
            Source:
            N/A
            
            Inputs:
            None
            
            Output:
            None
            
            Properties Used:
            N/A
        """ 
        return 
    
    def evaluate_battery_state_of_health(self,segment):
        """ Updates battery age based on operating conditions, cell temperature and time of operation.
    
        Assumptions:
        None 
        
        Source: 
        Cell specific. See individual battery cell for more details
          
        Assumptions:
        Cell specific. See individual battery cell for more details
       
        Inputs: 
        segment.
            conditions                    - conditions of battery at each segment  [unitless]
            increment_battery_cycle_day   - flag to increment battery cycle day    [boolean]
        
        Outputs:
        N/A  
             
        Properties Used:
        N/A 
        
        """          
         
        increment_day = segment.increment_battery_cycle_day 
            
        for network in segment.analyses.energy.network: 
            battery = network.battery
            battery.update_battery_state_of_health(segment,increment_battery_cycle_day = increment_day) 
        
                
        return    
