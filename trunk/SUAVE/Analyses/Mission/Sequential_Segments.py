## @ingroup Analyses-Mission
# Sequential_Segments.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import ContainerOrdered as ContainerBase
from SUAVE.Methods import Missions as Methods
from Mission import Mission

# ----------------------------------------------------------------------
#   Class
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission
class Sequential_Segments(Mission):
    """ Solves each segment one at time
    
        Assumptions:
        None
        
        Source:
        None
    """
    
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        
        self.tag = 'mission'
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize
        # --------------------------------------------------------------
        self.process.initialize = Methods.Segments.Common.Sub_Segments.expand_sub_segments

        # --------------------------------------------------------------
        #   Converge
        # --------------------------------------------------------------
        self.process.converge = Methods.Segments.Common.Sub_Segments.sequential_sub_segments
        
        # --------------------------------------------------------------
        #   Iterate
        # --------------------------------------------------------------        
        del self.process.iterate

        # --------------------------------------------------------------
        #   Finalize
        # --------------------------------------------------------------        
        self.process.finalize.sub_segments = Methods.Segments.Common.Sub_Segments.finalize_sub_segments
        
        return
    
    def finalize(self):
        """ Stub
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """           
        pass

# ----------------------------------------------------------------------
#   Cotnainer Class
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission
class Container(ContainerBase):
    """ Container for mission
    
        Assumptions:
        None
        
        Source:
        None
    """      
    
    def evaluate(self,state=None):
        """ Go through the missions, run through them, save the results
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state   [Data()]
    
            Outputs:
            Results [Results()]
    
            Properties Used:
            None
        """          
        results = SUAVE.Analyses.Results()
        
        for key,mission in self.items():
            result = mission.evaluate(state)
            results[key] = result
            
        return results
    
    def finalize(self):
        """ Stub
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
            """          
        pass

# Link container
Mission.Container = Container
