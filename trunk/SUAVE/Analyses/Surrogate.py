# Surrogate.py
#
# Created:  Trent Lukaczyk, March 2015 
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# imports

from SUAVE.Core import Data
from Analysis import Analysis


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Surrogate(Analysis):
    ''' Surrogate Base Class
    '''
    
    def __defaults__(self):
        self.training = Data()
        self.surrogates = Data()
        return

    def finalize(self):
        self.sample_training()
        self.build_surrogate()
        return

    def sample_training(self):
        return

    def build_surrogate(self):
        return

    def evaluate(self,state):
        results = None
        raise NotImplementedError
        return results

