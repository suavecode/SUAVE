# surrogate_problem.py
#
# Created:  May 206, M. Vegh
# Modified:


from SUAVE.Core import Data
import numpy as np

class Surrogate_Problem(Data):
    def __defaults__(self):
        self.obj_surrogate = None
        self.constraints_surrogates = None
    
    def compute(self, x):
        f = self.obj_surrogate.predict(x)
        g = []
        for j in range(len(self.constraints_surrogates)):
            g.append(self.constraints_surrogates[j].predict(x))
          
        #g = np.array(g)
        fail  = np.array(np.isnan(f.tolist()) or np.isnan(np.array(g).any())).astype(int)


        return f, g, fail
        
    __call__ = compute