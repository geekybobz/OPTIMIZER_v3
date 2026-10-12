"""Delayed independently verified shooting system for process-boundary tests."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).parent))
from shooting_systems import CubicRootShootingSystem

class DelayedCubic(CubicRootShootingSystem):
    def __init__(self, delay=.05):
        super().__init__()
        self.delay=delay
    def system_spec(self):
        spec=super().system_spec()
        spec['parameters']['delay']=self.delay
        return spec
    def residuals(self,variables,*,options=None):
        time.sleep(self.delay)
        return super().residuals(variables,options=options)
    def with_secondary(self,**updates):
        if updates: raise KeyError(updates)
        return DelayedCubic(self.delay)
def system(**parameters):
    return DelayedCubic(**parameters)
