


import SUAVE



def main():
    
    return
    
def setup_interface(configs,analyses):
    
    configs.origin
    configs.takeoff
    configs.cruise
    configs.landing
    
    analyses.mission.analyses.takeoff.aerodynamics
    analyses.mission.analyses.takeoff.aerodynamics.config
    
    analyses.mission.analyses.cruise.aerodynamics
    analyses.mission.analyses.landing.aerodynamics
    
    analyses.performance
    analyses.weights
    analyses.missions
    analyses.missions.fuel
    analyses.missions.short_field
    analyses.missions.payload
    analyses.field_length
    analyses.noise
    
    interface = Interface()
    interface.tag = 'optimization_outer_loop'
    
    interface.vehicle = vehicle
    interface.configs = configs
    interface.analyses = analyses
    
    
class Interface(object):
    
    def __init__(self):
        
        self.__cache__ = Hashed_Dict()
        
        self.inputs = None
        
        self.configs  = Data()
        self.analyses = Data()
        
        self.results = None
        
    def update(self):
        
        # unpack
        inputs   = self.inputs
        configs  = self.configs
        analyses = self.analyses
        
        # update the configs
        configs.inputs = inputs
        configs.update()
        
        # update the analyses
        analyses.inputs = inputs
        analyses.update()
        analyses.finalize()
        
    def evaluate(self):
        """ assumes self.update() was called
        """
        
        # unpack
        inputs   = self.inputs
        analyses = self.analyses
        
        # caching
        try:
            # pull from cache
            outputs = self.__cache__[inputs]
        
        # evaluate the problem
        except KeyError:
            
            # evaluate the analyses
            analyses.evaluate()
            outputs = analyses.outputs
             
            # store outputs to cache
            self.__cache__[inputs] = outputs
            
        # update outputs
        self.outputs = outputs
        
        # return
        return outputs
    
    def clear_cache(self):
        self.__cache__.clear()
        
    # make this interface callable
    def __call__(self,inputs):
        self.inputs = inputs
        self.update()
        self.evaluate()
        outputs = self.outputs
        return outputs
        
    
    
    