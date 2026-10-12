"""Neutral delay/failure/source-freeze fixture, not a scientific benchmark."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).parent))
from quadratic_system import QuadraticVectorSystem

SOURCE_OFFSET = 0.0


class JobQuadraticSystem(QuadraticVectorSystem):
    def __init__(self, *, N=5, delay=0.0, marker=None, crash=False, **parameters):
        super().__init__(N=N, **parameters)
        self.delay=float(delay); self.marker=marker; self.crash=bool(crash)
        self.calls=0

    def with_secondary(self, **updates):
        parameters = {**self.secondary, **updates}
        return self.__class__(N=self.N, tau=self.primary['tau'], delay=self.delay,
                              marker=self.marker, crash=self.crash, **parameters)

    def _target_matrix(self):
        return super()._target_matrix()+SOURCE_OFFSET

    def system_spec(self):
        spec=super().system_spec()
        spec["parameters"]["primary"].update(delay=self.delay,marker=self.marker,crash=self.crash)
        return spec

    def evaluate(self, controls):
        self.calls+=1
        if self.marker:
            with Path(self.marker).open('a') as output:
                output.write(json.dumps({"pid":os.getpid(),"threads":os.environ.get("OPENBLAS_NUM_THREADS")})+'\n')
        if self.crash: os._exit(19)
        time.sleep(self.delay)
        return super().evaluate(controls)


def system(**parameters):
    return JobQuadraticSystem(**parameters)
