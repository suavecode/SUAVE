

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data, Data_Exception, Data_Warning
from Analysis import Analysis
from Results import Results


# ----------------------------------------------------------------------
#  Vehicle Analysis
# ----------------------------------------------------------------------

class Vehicle(Analysis.Container):
    """ SUAVE.Analyses.Vehicle()
    """
    def __defaults__(self):
        self.sizing       = None
        self.weights      = None
        self.aerodynamics = None
        self.stability    = None
        self.energy       = None
        self.propulsion   = None
        self.atmosphere   = None
        self.planet       = None


    def append(self,analysis):
        
        key = self.get_root(analysis)
        
        self[key] = analysis
        

    _analyses_map = None
    
    def __init__(self,*args,**kwarg):
        
        Analysis.Container.__init__(self,*args,**kwarg)
        
        self._analyses_map = {
            SUAVE.Analyses.Sizing.Sizing             : 'sizing'       ,
            SUAVE.Analyses.Weights.Weights           : 'weights'      ,
            SUAVE.Analyses.Aerodynamics.Aerodynamics : 'aerodynamics' ,
            SUAVE.Analyses.Stability.Stability       : 'stability'    ,
            SUAVE.Analyses.Energy.Propulsion         : 'propulsion'   ,
            SUAVE.Analyses.Energy.Energy             : 'energy'       ,
            SUAVE.Analyses.Atmospheres.Atmosphere    : 'atmosphere'   ,
            SUAVE.Analyses.Planets.Planet            : 'planet'       ,
        }

    def get_root(self,analysis):

        # find analysis root by type, allow subclasses
        for analysis_type, analysis_root in self._analyses_map.iteritems():
            if isinstance(analysis,analysis_type):
                break
        else:
            raise Exception , "Unable to place analysis type %s" % analysis.typestring()

        return analysis_root        
        
        
   