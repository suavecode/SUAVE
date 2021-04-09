## @ingroup Analyses-Energy
# Energy.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Analyses import Analysis


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses-Energy
class Energy(Analysis):
    """ SUAVE.Analyses.Energy.Energy()
    """
    def __defaults__(self):
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
        self.tag     = 'energy'
        self.network = None
        
    def evaluate_thrust(self,state):
        
        """Evaluate the thrust produced by the energy network.
    
                Assumptions:
                Network has an "evaluate_thrust" method.
    
                Source:
                N/A
    
                Inputs:
                State data container
    
                Outputs:
                Results of the thrust evaluation method.
    
                Properties Used:
                N/A                
            """
                
            
        network = self.network
        results = network.evaluate_thrust(state) 
        
        return results
    