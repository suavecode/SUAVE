## @ingroup Analyses
# Settings.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Core import Container as ContainerBase


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses
class Settings(Data):
    """ SUAVE.Analyses.Settings()
    
        The Top Level Settings Class
        
            Assumptions:
            None
            
            Source:
            N/A
    """
    def __defaults__(self):
        """This sets the default values and methods for the settings.
        
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
        self.tag    = 'settings'
        
        self.verbose_process = False
        

## ----------------------------------------------------------------------
##  Config Container
## ----------------------------------------------------------------------

### @ingroup Analyses
#class Container(ContainerBase):
    #""" SUAVE.Analyses.Settings.Container()
    
        #The Top Level Settings Container Class
        
            #Assumptions:
            #None
            
            #Source:
            #N/A
    #"""
    #pass

## ------------------------------------------------------------
##  Handle Linking
## ------------------------------------------------------------

#Settings.Container = Container