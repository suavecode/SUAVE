
# ----------------------------------------------------------------------
#   Exceptions
# ----------------------------------------------------------------------

class EvaluationFailure(Exception):
    pass

class Infeasible(EvaluationFailure):
    pass
    
class ResourceException(Exception):
    pass


# ----------------------------------------------------------------------
#   Warnings
# ----------------------------------------------------------------------

class ResourceWarning(Warning):
    pass

class FailedTask(Warning):
    pass

