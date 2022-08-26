from __future__ import annotations
import logging
try:
    import tomllib  # tomli is a built-in library in Python 3.11
except ModuleNotFoundError:
    import tomli as tomllib
from FitAO.Tools.simulation_parameters import SimulationParameters
import pydantic

logger = logging.getLogger(__name__)


def parse_parameter_file(file_path: str) -> SimulationParameters:
    """Read and validate a user-provided parameter file.

    This function reads the user-provided parameter file and validates the provided data to
    ensure that all required parameters and sections have been provided, as well as ensuring
    where possible that the provided values make sense. See ``SimulationParameters`` for more
    details on the validation process.

    Args:
        file_path: Path to the file which specifies simulation parameters.

    Returns:
        SimulationParameters: A container for all the simulation parameters which have been
        read, validated and logically organised. See ``SimulationParameters`` for more details.

    Raises:
        FitAOParameterFileReadError: If the provided file path cannot be read, for example if
            it doesn't exist.
        FitAOParameterFileParseError: If the given file is not a valid TOML-file; this is
            usually just caused by a typo, and while the error text itself may not be very
            helpful, it does include the line where the error occurred.
        FitAOParameterFileValidationError: If the file failed a validation check, i.e. the file
            doesn't fully match our parameter file specification for some reason. Examples
            include missing required parameters or sections, misspellings in parameter or
            section names, or trying to provide an incompatible value for a parameter, e.g. a
            negative number for a parameter that should be non-negative. The error will
            include a list of all failed validation checks.

            These validations are done to avoid the more serious problem of passing invalid
            data to simulation environments, and follow the principle that if a program is
            going to fail, it should fail as early as possible and in an obvious manner.
    """
    logger.info(f'Opening parameter file at {file_path}.')
    try:
        with open(file_path, "rb") as file:
            data = tomllib.load(file)
    except OSError as error:
        logger.exception(f'Trying to open parameter file {file_path} produced an error.')
        raise FitAOParameterFileReadError(file_path) from error
    except tomllib.TOMLDecodeError as error:
        # TODO: TOML produces a very unhelpful error if the same section is in the file twice,
        # which is a fairly likely scenario for new users if they don't realise that you need
        # e.g. [[wavefrontsensor]] rather than [wavefrontsensor]
        logger.exception(f'Trying to parse parameter file {file_path} as a TOML-file produced an '
                         f'error.')
        raise FitAOParameterFileParseError(file_path) from error
    try:
        data = SimulationParameters.parse_obj(data)
    except pydantic.ValidationError as error:
        logger.exception(f'Trying to validate parameter file {file_path} produced an error.')
        raise FitAOParameterFileValidationError(error) from error

    logger.info(f'Parameter file successfully loaded and validated!')
    return data


class FitAOParameterParserError(Exception):
    """Base exception type for this module.

    All other exceptions in this module inherit from `FitAOParameterParserError`. This allows
    users to catch all expected errors from this module with a single except-clause without
    hiding possible unexpected errors with an overly broad except-clause.
    """
    pass


class FitAOParameterFileReadError(FitAOParameterParserError, OSError):
    def __init__(self, file_path: str = ""):
        self.file_path = file_path
        super().__init__(
            f'Could not read parameter file {self.file_path}.'
        )


class FitAOParameterFileParseError(FitAOParameterParserError):
    def __init__(self, file_path: str = ""):
        self.file_path = file_path
        super().__init__(
            f'Could not parse parameter file {self.file_path}, it must not have been '
            f'a valid TOML file.'
        )


class FitAOParameterFileValidationError(FitAOParameterParserError):
    def __init__(self, error: pydantic.ValidationError, file_path: str = ""):
        self.file_path = file_path
        self.error = error
        try:  # Try to convert pydantic's ValidationError into a more readable form
            n_errors, error_details = self._parse_pydantic_validation_errors()
            super().__init__(
                f'Validation failed for parameter file {self.file_path} due to {n_errors} '
                f'errors:\n{error_details}')
        except NotImplementedError:
            super().__init__(str(self.error))
        except Exception:  # If anything goes wrong, preserve the original error
            logger.exception("Rewriting pydantic's ValidationError produced an error.")
            super().__init__(str(self.error))

    def _parse_pydantic_validation_errors(self) -> (int, str):
        """Write pydantic errors into a user-friendly string.

        Notes:
            This function involves a fair bit of "black magic", since

        Args:
            error: Error produced by pydantic model creation.

        Returns:
            int: Number of errors.
            str: String of error descriptions, with each error on its own line.
        """
        raise NotImplementedError("Work in progress")
        # errors = self.error.errors()
        # raw_errors = self.error.raw_errors
        # n_errors = len(errors)
        #
        # error_details = []
        #
        # for err in errors:
        #     # Tuple where first entry is field name in SimulationParameters (if any), and
        #     # possible other entries are field names within the field's model
        #     location, *extra = err['loc']
        #     type_ = err['type']
        #     message = err['msg']
        #
        #     error_details.append(f'{location}')
        #
        # error_str = '\n'.join(error_details)
        #
        # return n_errors, error_str

#
# def _interpret_pydantic_type():
#
#
#     # Literal black magic, only way I found to check types
#
#     # if location has multiple entries, it must be a field inside another model
#     len(location) > 1
#
#     # location always has at least one entry, but it may be empty
#
#     # get field from field name
#     fields = SimulationParameters.__fields__
#
#     # check if field is a model, also works for list[model] etc.
#     issubclass(field.type_, pydantic.BaseModel)
#
#     # check if field is just a model or if it's wrapped inside e.g. an iterable
#     field.type_ is field.outer_type_
