"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import cirq_google.api.v2.program_pb2
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor = ...

class Result(google.protobuf.message.Message):
    """The overall results of running a Program."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    SWEEP_RESULTS_FIELD_NUMBER: builtins.int
    @property
    def sweep_results(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___SweepResult]:
        """The results for each ParameterSweep. These will be in the same order
        as the parameter_sweeps repeated field in the RunContext that generated
        these results.
        """
        pass
    def __init__(self,
        *,
        sweep_results : typing.Optional[typing.Iterable[global___SweepResult]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"sweep_results",b"sweep_results"]) -> None: ...
global___Result = Result

class SweepResult(google.protobuf.message.Message):
    """The measurement results for a particular ParameterSweep."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    REPETITIONS_FIELD_NUMBER: builtins.int
    PARAMETERIZED_RESULTS_FIELD_NUMBER: builtins.int
    repetitions: builtins.int = ...
    """The total number of repetitions that were performed for this sweep.
    This is reported in order to make it possible to decode the bytes
    in the measurement results.
    """

    @property
    def parameterized_results(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ParameterizedResult]:
        """The results along with the parameters that generated these results.
        These represent the expanded parameters defined int he ParameterSweep
        which this SweepResult corresponds to.
        """
        pass
    def __init__(self,
        *,
        repetitions : builtins.int = ...,
        parameterized_results : typing.Optional[typing.Iterable[global___ParameterizedResult]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"parameterized_results",b"parameterized_results",u"repetitions",b"repetitions"]) -> None: ...
global___SweepResult = SweepResult

class ParameterizedResult(google.protobuf.message.Message):
    """The parameters used to generate result along with the results for this
    set of parameters.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    PARAMS_FIELD_NUMBER: builtins.int
    MEASUREMENT_RESULTS_FIELD_NUMBER: builtins.int
    @property
    def params(self) -> global___ParameterDict:
        """The parameter dict that was used when generating these results."""
        pass
    @property
    def measurement_results(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___MeasurementResult]:
        """The results of the measurement. There is one of these results per
        measurement key in the program. Measurement keys in the program are
        unique.
        """
        pass
    def __init__(self,
        *,
        params : typing.Optional[global___ParameterDict] = ...,
        measurement_results : typing.Optional[typing.Iterable[global___MeasurementResult]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"params",b"params"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"measurement_results",b"measurement_results",u"params",b"params"]) -> None: ...
global___ParameterizedResult = ParameterizedResult

class MeasurementResult(google.protobuf.message.Message):
    """The results of a measurement for a specific measurement key."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    KEY_FIELD_NUMBER: builtins.int
    INSTANCES_FIELD_NUMBER: builtins.int
    QUBIT_MEASUREMENT_RESULTS_FIELD_NUMBER: builtins.int
    key: typing.Text = ...
    """The measurement key for the measurement."""

    instances: builtins.int = ...
    """Number of instances of this key in each circuit repetition."""

    @property
    def qubit_measurement_results(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___QubitMeasurementResult]:
        """For each qubit that is measured, these are the measurement results."""
        pass
    def __init__(self,
        *,
        key : typing.Text = ...,
        instances : builtins.int = ...,
        qubit_measurement_results : typing.Optional[typing.Iterable[global___QubitMeasurementResult]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"instances",b"instances",u"key",b"key",u"qubit_measurement_results",b"qubit_measurement_results"]) -> None: ...
global___MeasurementResult = MeasurementResult

class QubitMeasurementResult(google.protobuf.message.Message):
    """The result of a measurement on a single qubit."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    QUBIT_FIELD_NUMBER: builtins.int
    RESULTS_FIELD_NUMBER: builtins.int
    @property
    def qubit(self) -> cirq_google.api.v2.program_pb2.Qubit:
        """Which qubit was measured."""
        pass
    results: builtins.bytes = ...
    """These are the results of a measurement on a qubit. The number of bits
    measured is equal to repetitions * instances, where repetitions is defined
    in the SweepResult message, and instances is defined in the MeasurementResult
    message.

    The bytes in this field are constructed as follows:

    1. The results of the measurements produce a list of bits ordered by
       the round of repetition and instance within a round.

    2. This list is broken up into blocks of 8 bits, with the final block
       potentially not being a full 8 bits.

    3. Each of the blocks of 8 bits is encoded into a byte, using little
       endian notation. That is, the least significant bit of the byte is
       the first bit of the bit string, the second-least significant
       bit of the byte is the second bit of the bit string, etc. For the final
       byte which may not have 8 bits, little endian encoding is still used,
       but not all 8 bits may be used. This list of bytes then constitutes
       the bytes in this field.
    """

    def __init__(self,
        *,
        qubit : typing.Optional[cirq_google.api.v2.program_pb2.Qubit] = ...,
        results : builtins.bytes = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"qubit",b"qubit"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"qubit",b"qubit",u"results",b"results"]) -> None: ...
global___QubitMeasurementResult = QubitMeasurementResult

class ParameterDict(google.protobuf.message.Message):
    """A point sampled during a parameter sweep."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    class AssignmentsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text = ...
        value: builtins.float = ...
        def __init__(self,
            *,
            key : typing.Text = ...,
            value : builtins.float = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal[u"key",b"key",u"value",b"value"]) -> None: ...

    ASSIGNMENTS_FIELD_NUMBER: builtins.int
    @property
    def assignments(self) -> google.protobuf.internal.containers.ScalarMap[typing.Text, builtins.float]:
        """Maps parameter names to values."""
        pass
    def __init__(self,
        *,
        assignments : typing.Optional[typing.Mapping[typing.Text, builtins.float]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"assignments",b"assignments"]) -> None: ...
global___ParameterDict = ParameterDict
