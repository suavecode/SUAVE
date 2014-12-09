

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Structure import Data, Data_Exception, Data_Warning
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
        
        if self[key] is None:
            self[key] = analysis
            
        else:
            raise Exception, 'Analysis already has key %s' % key
        
        

    _analyses_map = None
    
    def __init__(self,*args,**kwarg):
        
        Analysis.Container.__init__(self,*args,**kwarg)
        
        from SUAVE import Analyses as Analyses_
        
        self._analyses_map = {
            Analyses_.Sizing.Sizing             : 'sizing'       ,
            Analyses_.Weights.Weights           : 'weights'      ,
            Analyses_.Aerodynamics.Aerodynamics : 'aerodynamics' ,
            Analyses_.Stability.Stability       : 'stability'    ,
            Analyses_.Energy.Propulsion         : 'propulsion'   ,
            Analyses_.Energy.Energy             : 'energy'       ,
            Analyses_.Atmospheres.Atmosphere    : 'atmosphere'   ,
            Analyses_.Planets.Planet            : 'planet'       ,
        }

    def get_root(self,analysis):

        # find analysis root by type, allow subclasses
        for analysis_type, analysis_root in self._analyses_map.iteritems():
            if isinstance(analysis,analysis_type):
                break
        else:
            raise Exception , "Unable to place analysis type %s" % analysis.typestring()

        return analysis_root        
        
        
   