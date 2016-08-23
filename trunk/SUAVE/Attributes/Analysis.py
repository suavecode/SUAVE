# Analysis.py
# 
# Created:  Unk,     , T. Lukaczyk
# Modified: Jan, 2016, M. Vegh



# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Analysis Data Class
# ----------------------------------------------------------------------

class Analysis(Data):
    """ SUAVE Data Class for Analysis
        used for 
    """
    def __defaults__(self):
        self.Vehicle = None
        self.Mission = None
        self.Procedure = AnalysisMap()
        
    def solve(self):
        procedure = self.procedure
        for segment,configuration in procedure.items():
            results = segment.solve(configuration)
        
    def __str__(self):
        args = ''
        args += self.dataname() + '\n'
        args += 'Vehicle = %s\n' % self.Vehicle.tag
        args += 'Mission = %s\n' % self.Mission.tag
        args += 'Procedure =\n'
        for step in self.Procedure.values():
            seg = step[0]
            con = step[1]
            args += '  %s : %s\n' % (seg.tag,con.tag)
        return args
        
class AnalysisMap(Data):
    pass