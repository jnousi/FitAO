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
from __future__ import annotations
from pydantic import (
    BaseModel,
    Extra,
    NonNegativeFloat,
    NonNegativeInt,
    validator,
    root_validator,
)
from pydantic.typing import get_origin
from typing import Optional, List, Union
import logging
from inspect import isclass


logger = logging.getLogger(__name__)

def _alphanumeric_lowercase(field_name: str) -> str:
    alias = ''.join(filter(str.isalnum, field_name))
    return alias.lower()


class Parameters(BaseModel):
    """Base class for global configuration settings.

    This is the superclass which `SimulationParameters` and its attributes all inherit from.
    """
    @root_validator(pre=True, allow_reuse=True)
    def preprocess_field_names(cls, values):
        new_values = {_alphanumeric_lowercase(k): v for (k, v) in values.items()}
        return new_values

    class Config:
        # Disallow parameters which are not explicitly listed. If they are needed, they can
        # be passed to the Unsupported-class.
        extra = Extra.forbid

        # Disallow changing attribute values after model creation.
        allow_mutation = False

        # Enable using aliases for field names
        allow_population_by_field_name = True

        alias_generator = _alphanumeric_lowercase


class Loop(Parameters):
    n_iterations: NonNegativeInt
    step_time: NonNegativeFloat


class Geometry(Parameters):
    zenith_angle: Optional[float]
    pupil_diameter: Optional[NonNegativeFloat]


class Telescope(Parameters):
    diameter: NonNegativeFloat
    obstruction_ratio: NonNegativeFloat


class Atmosphere(Parameters):
    r0: NonNegativeFloat
    n_screens: NonNegativeInt
    fractional_r0: List[NonNegativeFloat]
    altitude: List[float]
    L0: Union[NonNegativeFloat, List[NonNegativeFloat]]
    wind_speed: Optional[List[float]]
    wind_direction: Optional[List[float]]


class Target(Parameters):
    x_position: float
    y_position: float
    magnitude: NonNegativeFloat
    wavelength: Optional[NonNegativeFloat]


class GuideStar(Parameters):
    pass


# TODO: Some of these params should probably be under GuideStar, and WFS may need a GuideStar field
class WavefrontSensor(Parameters):
    type: str
    x_subapertures: NonNegativeInt
    n_pixels: NonNegativeInt
    pixel_size: NonNegativeFloat
    x_position: float
    y_position: float
    fracsub: Optional[NonNegativeFloat]
    wavelength: Optional[float]  # TODO: Nonnegative?
    altitude: Optional[float]
    lltx: Optional[float]  # Not sure what these are
    llty: Optional[float]
    guidestar_magnitude: Optional[NonNegativeFloat]
    optical_throughput: Optional[NonNegativeFloat]
    zerop: Optional[NonNegativeFloat]
    noise: Optional[NonNegativeFloat]
    atmosphere_seen: Optional[bool]
    fssize: Optional[float]
    modulation_amplitude: Optional[float]  # pyr_amp
    pyr_pup_sep: Optional[float]
    pyr_npts: Optional[float]


class DeformableMirror(Parameters):
    type: str
    n_actuators: NonNegativeInt
    altitude: float
    threshold: Optional[NonNegativeFloat]
    coupling: Optional[NonNegativeFloat]
    unit_per_volt: Optional[NonNegativeFloat]
    calibration_constant: Optional[NonNegativeFloat]


class Centroider(Parameters):
    type: str
    n_wavefront_sensors: NonNegativeInt


class Controller(Parameters):
    type: str
    n_wavefront_sensors: List[NonNegativeInt]
    n_deformable_mirrors: List[NonNegativeInt]
    delay: NonNegativeInt
    maxcond: Optional[NonNegativeFloat]  # TODO: Not sure what this one is
    gain: Optional[NonNegativeFloat]
    modopti: Optional[bool]


class Unsupported(Parameters, extra=Extra.allow):
    """Container for extra parameters which aren't officially supported.


    """
    pass


class SimulationParameters(Parameters):
    simulation_name: Optional[str]
    loop: Optional[Loop]
    geometry: Optional[Geometry]
    telescope: Optional[Telescope]
    atmosphere: Optional[Atmosphere]
    target: Optional[List[Target]]
    guidestar: Optional[List[GuideStar]]
    wavefront_sensor: Optional[List[WavefrontSensor]]
    deformable_mirror: Optional[List[DeformableMirror]]
    centroider: Optional[List[Centroider]]
    controller: Optional[List[Controller]]
    extra_data: Optional[Unsupported]

    @validator('*', pre=True, allow_reuse=True)
    def _convert_to_list(cls, value, field):
        """Checks if fields expecting a list are given a list element instead.

        Fields which expect lists of `Parameters`-objects can be a bit unintuitive when
        it comes to the TOML-specification, since the parameter file sections need double
        brackets, e.g. [[target]]. If the user has entered it as [target] instead, this
        function will allow it to still be accepted by converting it to a list, while also
        warning the user that using single brackets may result in an error.
        """
        # If value is already a list, we don't need to do anything.
        if isinstance(value, list):
            return value

        # Skip fields which don't expect a list. field.outer_type_ gives the field
        # type definition (with Optional removed), and get_origin gives the outer-most
        # type in that type definition, which should be list
        if get_origin(field.outer_type_) is not list:
            return value

        # Skip fields which expect a list of something other than a subclass of Parameters
        if not isclass(field.type_) or not issubclass(field.type_, Parameters):
            return value

        logger.warning(
            f"Note: the correct syntax for defining {field.name} is [[{field.name}]]. "
            f"When you use [{field.name}], you will get an error if you try to define more "
            f"than one of them. If you only have one of these sections, everything should "
            f"work fine, but we recommend that you fix this nonetheless. If nothing else, "
            f"it'll at least get rid of this annoying warning!"
        )

        return [value]


# TODO: Maybe add some automation here, like automatically adding plurals
# Consider moving this into another file with a cleaner implementation

field_name_aliases = {
    'SimulationParameters': [
        ['simulation_name', 'simulation_name', 'sim_name', 'simul_name'],
        ['loop', 'iteration'],
        ['geometry', 'geom'],
        ['telescope', 'tel'],
        ['atmosphere', 'atm', 'atmos'],
        ['target', 'tar', 'tars', 'targets'],
        ['guidestar', 'gs', 'gss', 'guidestars'],
        ['wavefront_sensor', 'wfs', 'wfss', 'wavefront_sensors'],
        ['deformable_mirror', 'dm', 'dms', 'deformable_mirrors'],
        ['centroider', 'centroiders'],
        ['controller', 'controllers'],
        ['extra_data', 'extra', 'extras', 'unsupported'],
    ],

    'Loop': [
        ['n_iterations', 'n_iter', 'n_iters', 'n_iterations'],
        ['step_time', 'it_time', 'iteration_time'],
    ],

    'Geometry': [
        ['zenith_angle', 'zenith'],
        ['pupil_diameter', 'pup_diam', 'pupil_diam'],
    ],

    'Telescope': [
        ['diameter', 'diam'],
        ['obstruction_ratio', 'obs_ratio', 'cobs', ],
    ],

    'Atmosphere': [
        ['r0', 'fried'],
        ['n_screens', 'n_screen', 'n_layers', 'n_layer'],
        ['fractional_r0', 'frac', 'fractional'],
        ['altitude', 'alt', 'alts', 'altitudes', 'height', 'heights'],
        ['L0', 'outer_scale'],
        ['wind_speed', 'wind_speeds'],
        ['wind_direction', 'wind_dir', 'wind_dirs', 'wind_directions'],
    ],

    'Target': [
        ['x_position', 'x_pos', 'x_coord', 'x_coordinate', 'x_loc', 'x_location'],
        ['y_position', 'y_pos', 'y_coord', 'y_coordinate', 'y_loc', 'y_location'],
        ['magnitude', 'mag'],
        ['wavelength', 'lambda',],
    ],

    'GuideStar': [

    ],

    'WavefrontSensor': [
        ['type', ],
        ['x_subapertures', 'x_sub', 'x_subs', 'x_subaps', 'sub', 'subs', 'subaps', 'subapertures'],
        ['n_pixels', 'n_pix', 'n_pixel'],
        ['pixel_size', 'pix_size'],
        ['x_position', 'x_pos', 'x_coord', 'x_coordinate', 'x_loc', 'x_location'],
        ['y_position', 'y_pos', 'y_coord', 'y_coordinate', 'y_loc', 'y_location'],
        ['fracsub', ],
        ['wavelength', 'lambda'],
        ['alt', 'altitude', 'gs_alt', 'gs_altitude', 'guidestar_altitude'],
        ['lltx'],
        ['llty'],
        ['guidestar_magnitude', 'mag', 'magnitude', 'gs_mag'],
        ['optical_throughput', 'opt_throughput'],
        ['zerop', ],
        ['noise', ],
        ['atmosphere_seen', 'atm_seen', 'atmos_seen'],
        ['fssize', ],
        ['pyr_amp', ],
        ['pyr_pup_sep', ],
        ['pyr_npts', ],
    ],

    'DeformableMirror': [
        ['type', ],
        ['n_actuators', 'n_act', 'n_acts', 'n_actuator', 'act', 'acts', 'actuator', 'actuators'],
        ['altitude', ],
        ['threshold', ],
        ['coupling', ],
        ['unit_per_volt', ],
        ['calibration_constant', 'push4imat', ],
    ],

    'Centroider': [
        ['n_wavefront_sensors', ],
        ['type', ],
    ],

    'Controller': [
        ['type', ],
        ['n_wavefront_sensors', ],
        ['n_deformable_mirrors', ],
        ['delay', ],
        ['maxcond', ],
        ['gain', ],
        ['modopti', ],
    ],

    'Unsupported': [

    ]
}
