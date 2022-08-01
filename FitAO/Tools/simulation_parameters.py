"""Define class for storing and validating user-provided simulation parameters.

This module defines the class `SimulationParameters`, which `FitAO.Tools.parameter_parser` uses to
process and validate user-provided simulation parameter files. The dictionaries read from these
.toml-files can be automatically converted into a `SimulationParameters`-instance, which can later
be converted back into a dictionary with `SimulationParameters.dict()`.

`SimulationParameters` is used to specify exactly which simulation parameters are supported by
this package and what their types should be. The class validates that the user-provided parameters
were given in the correct format, and will produce an error if e.g. any section or parameter name
in the file is not identified, cannot be cast from a string to the appropriate type, or does not
satisfy any additional constraints (e.g. most floats are required to be non-negative).

Note:
    If a parameter is not supported
    Internally, `SimulationParameters` and its attributes use the `pydantic`-package, which allows
    for very s
"""
from pydantic import (
    BaseModel,
    Extra,
    ValidationError,
    NonNegativeFloat,
    NonNegativeInt,
    confloat,
)
from typing import Optional, List, Union



class Parameters(BaseModel):
    """Base class for global configuration settings.

    This is the superclass which `SimulationParameters` and its attributes all inherit from.
    """
    class Config:
        # Disallow parameters which are not explicitly listed, as each parameter needs to be individually supported
        extra = Extra.forbid
        # Disallow changing attribute values after model creation, as this can inadvertently lead to
        allow_mutation = False


class Loop(Parameters):
    niter: NonNegativeInt
    ittime: NonNegativeFloat


class Geometry(Parameters):
    zenithangle: float
    pupdiam: NonNegativeFloat


class Telescope(Parameters):
    diam: NonNegativeFloat
    cobs: NonNegativeFloat


class Atmosphere(Parameters):
    r0: NonNegativeFloat
    nscreens: NonNegativeInt
    frac: List[NonNegativeFloat]
    alt: List[float]
    windspeed: List[float]
    winddir: List[float]
    L0: Union[NonNegativeFloat, List[NonNegativeFloat]]


class Target(Parameters):
    xpos: float
    ypos: float
    Lambda: NonNegativeFloat
    mag: NonNegativeFloat


class GuideStar(Parameters):
    pass


class WFS(Parameters):
    type: str
    nxsub: NonNegativeInt
    npix: NonNegativeInt
    pixsize: NonNegativeFloat
    fracsub: NonNegativeFloat
    xpos: float
    ypos: float
    Lambda: float  # TODO: Nonnegative?
    gsmag: NonNegativeFloat
    optthroughput: NonNegativeFloat
    zerop: NonNegativeFloat
    noise: NonNegativeFloat
    atmos_seen: bool


class DM(Parameters):
    type: str
    nact: NonNegativeInt
    alt: float
    thresh: NonNegativeFloat
    coupling: NonNegativeFloat
    unitpervolt: NonNegativeFloat
    push4imat: NonNegativeFloat


class Centroider(Parameters):
    nwfs: NonNegativeInt
    type: str


class Controller(Parameters):
    type: str
    nwfs: List[NonNegativeInt]
    ndm: List[NonNegativeInt]
    maxcond: NonNegativeFloat  # TODO: Not sure what this one is
    delay: NonNegativeInt
    gain: NonNegativeFloat
    modopti: bool

class Unsupported(Parameters, extra=Extra.allow):
    """Container for additional user parameters not officially supported.


    """
    pass


class SimulationParameters(Parameters):
    simul_name: str
    loop: Loop
    geom: Geometry
    tel: Telescope
    atmos: Atmosphere
    target: List[Target]
    gs: Optional[List[GuideStar]]
    wfs: List[WFS]
    dm: List[DM]
    centroider: List[Centroider]
    controller: List[Controller]
    # data: Unsupported

    def