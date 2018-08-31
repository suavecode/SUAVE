## @ingroup Analyses
# Vehicle.py
#
# Created:
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from .Analysis import Analysis


# ----------------------------------------------------------------------
#  Vehicle Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses
class Vehicle(Analysis.Container):
    """ SUAVE.Analyses.Vehicle()
    
        The Vehicle Analyses Container Class
        
            Assumptions:
            None
            
            Source:
            N/A
    """
    def __defaults__(self):
        """This sets the default analyses to be applied to the vehicle.
        
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
        self.sizing       = None
        self.weights      = None
        self.aerodynamics = None
        self.stability    = None
        self.energy       = None
        self.atmosphere   = None
        self.planet       = None
        self.noise        = None
        self.costs        = None

    def append(self,analysis):
        """This is used to add new analyses to the container.
        
                Assumptions:
                None
                
                Source:
                N/A
                
                Inputs:
                Analysis to be added
                
                Outputs:
                None
                
                Properties Used:
                N/A
        """

        key = self.get_root(analysis)

        self[key] = analysis


    _analyses_map = None

    def __init__(self,*args,**kwarg):
        """This sets the initialization behavior of the vehicle analysis
           container. Maps analysis paths to string keys.
           
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

        Analysis.Container.__init__(self,*args,**kwarg)

        self._analyses_map = {
            SUAVE.Analyses.Sizing.Sizing             : 'sizing'       ,
            SUAVE.Analyses.Weights.Weights           : 'weights'      ,
            SUAVE.Analyses.Aerodynamics.Aerodynamics : 'aerodynamics' ,
            SUAVE.Analyses.Stability.Stability       : 'stability'    ,
            SUAVE.Analyses.Energy.Energy             : 'energy'       ,
            SUAVE.Analyses.Atmospheric.Atmospheric   : 'atmosphere'   ,
            SUAVE.Analyses.Planets.Planet            : 'planet'       ,
            SUAVE.Analyses.Noise.Noise               : 'noise'        ,
            SUAVE.Analyses.Costs.Costs               : 'costs'        ,
        }

    def get_root(self,analysis):

        """ This is used to determine the root of the analysis path associated
            with a particular analysis key by the analysis map.
            
                Assumptions:
                None
                
                Source:
                N/A
                
                Inputs:
                Analysis key to be checked
                
                Outputs:
                Path root of analysis
                
                Properties Used:
                N/A
                
                
        """
        for analysis_type, analysis_root in self._analyses_map.items():
            if isinstance(analysis,analysis_type):
                break
        else:
            raise Exception("Unable to place analysis type %s" % analysis.typestring())

        return analysis_root


