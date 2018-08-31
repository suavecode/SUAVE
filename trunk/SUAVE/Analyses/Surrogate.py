## @ingroup Analyses
# Surrogate.py
#
# Created:  Trent Lukaczyk, March 2015 
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# imports

from SUAVE.Core import Data
from .Analysis import Analysis


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses
class Surrogate(Analysis):
    ''' SUAVE.Analyses.Surrogate()
    
        The Top Level Surrogate Class
        
            Assumptions:
            None
            
            Source:
            N/A
    '''
    
    def __defaults__(self):
        """This sets the default values for the surrogate.
        
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
        self.training = Data()
        self.surrogates = Data()
        return

    def finalize(self):
        """This is used to finalize the surrogate's training and build
           the final surrogate.
        
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
        self.sample_training()
        self.build_surrogate()
        return

    def sample_training(self):
        """This is used to train the surrogate on the sampled data
           according the optimization plan.
               
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
        return

    def build_surrogate(self):
        """This is used to build the surrogate based on previous training.
                       
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
        return

    def evaluate(self,state):
        """This is used to evaluate the surrogate.
                               
                Assumptions:
                None
                               
                Source:
                N/A
                               
                Inputs:
                None
                               
                Outputs:
                Results of the surrogate model
                               
                Properties Used:
                N/A
            """                
        results = None
        raise NotImplementedError
        return results

