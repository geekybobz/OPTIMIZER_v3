"""Initial guess generators for OPTIMIZER v3.

Why this package exists
-----------------------
Optimization quality depends heavily on the starting controls.  This package provides
direct, smooth, reproducible initial guesses that can be generated from
``system.control_spec()`` without knowing the physics of the system.  The goal is to
make it easy to compare many starts: zero, constants, harmonic pulses, localized
pulses, random smooth pulses, Fourier pulses, and perturbations around known results.

How it fits the architecture
----------------------------
- each function returns the standard ``Controls`` container.
- functions accept either a system or a ``ControlSpec``.
- shared amplitude, channel, endpoint, envelope, and scale behavior lives in
  ``base.py``.
- the public facade exposes these as direct ``opt.*_guess(...)`` calls.

Reviewer invariants
-------------------
- guess generators do not call ``system.evaluate`` or ``system.gradient``.
- generated controls are finite and shape-valid.
- random generators are reproducible through explicit seeds.
"""

from optimizer.guesses.composite import mix_guess, perturb_guess, scale_guess
from optimizer.guesses.harmonic import (
    cosine_guess,
    fourier_guess,
    gaussian_guess,
    sinc_guess,
    sine_guess,
)
from optimizer.guesses.random import random_fourier_guess, random_guess, random_smooth_guess
from optimizer.guesses.simple import constant_guess, ramp_guess, zero_guess

__all__ = [
    "constant_guess",
    "cosine_guess",
    "fourier_guess",
    "gaussian_guess",
    "mix_guess",
    "perturb_guess",
    "ramp_guess",
    "random_fourier_guess",
    "random_guess",
    "random_smooth_guess",
    "scale_guess",
    "sinc_guess",
    "sine_guess",
    "zero_guess",
]
