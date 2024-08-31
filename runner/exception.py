class RunnerException(Exception):
    pass

class SkipStep(RunnerException):
    pass

class SkipEpoch(RunnerException):
    pass

class StopRun(RunnerException):
    pass
