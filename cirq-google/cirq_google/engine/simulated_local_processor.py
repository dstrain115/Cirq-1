# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime

from typing import Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING, Union

import cirq

from cirq_google.api import v2
from cirq_google.engine import calibration
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.local_simulation_type import LocalSimulationType
from cirq_google.engine.simulated_local_job import SimulatedLocalJob
from cirq_google.engine.simulated_local_program import SimulatedLocalProgram

import cirq_google.engine.validating_sampler as validating_sampler

if TYPE_CHECKING:
    from cirq_google.serialization.serializer import Serializer

VALID_LANGUAGES = [
    'type.googleapis.com/cirq.google.api.v2.Program',
    'type.googleapis.com/cirq.google.api.v2.FocusedCalibration',
    'type.googleapis.com/cirq.google.api.v2.BatchProgram',
]


class SimulatedLocalProcessor(AbstractLocalProcessor):
    """A processor backed by a sampler and device.

    Intended for local simulation testing, this processor will
    create a VAlidationSampler that will validate requests based on
    the provided device and an additional Callable (that can verify
    serialization constraints, for instance).  Jobs will then be
    executed using the provided sampler.

    This class also supports a list of calibration metrics that are
    stored in-memory to replicate Quantum Engine calibration metrics.

    This class can be used as a local emulator for the Quantum Engine
    API or for testing or mocking.

    Attributes:
        sammpler: A `cirq.Sampler` that can execute the quantum jobs.
        device: An optional device, for validation of qubit connectivity.
        validator: A Callable that can validate additional characteristics
            beyond the device, such as serialization, repetition limits, etc.
        calibrations: A dictionary of calibration metrics keyed by epoch seconds
            that can be returned by the processor.
    """

    def __init__(
        self,
        *args,
        sampler: cirq.Sampler = cirq.Simulator(),
        device: cirq.Device = cirq.UNCONSTRAINED_DEVICE,
        validator: validating_sampler.VALIDATOR_TYPE = None,
        simulation_type: LocalSimulationType = LocalSimulationType.SYNCHRONOUS,
        calibrations: Optional[Dict[int, calibration.Calibration]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calibrations = calibrations or {}
        self._device = device
        self._simulation_type = simulation_type
        self._validator = validator
        self._sampler = validating_sampler.ValidatingSampler(
            device=self._device, validator=self._validator, sampler=sampler
        )
        self._programs: Dict[str, AbstractLocalProgram] = {}

    def remove_program(self, program_id: str):
        """Remove reference to a child program."""
        if program_id in self._programs:
            del self._programs[program_id]

    def get_calibration(self, calibration_timestamp_seconds: int) -> calibration.Calibration:
        return self._calibrations[calibration_timestamp_seconds]

    def get_latest_calibration(self, timestamp: int) -> Optional[calibration.Calibration]:
        latest = None
        current_calibration = None
        for calibration_seconds in self._calibrations:
            if calibration_seconds <= timestamp and (
                latest is None or latest < calibration_seconds
            ):
                latest = calibration_seconds
                current_calibration = self._calibrations[latest]
        return current_calibration

    def get_current_calibration(self) -> Optional[calibration.Calibration]:
        return self.get_latest_calibration(int(datetime.datetime.now().timestamp()))

    def get_device(self, gate_sets: Optional[Iterable['Serializer']] = None) -> cirq.Device:
        """Returns a `Device` created from the processor's device specification.

        This method queries the processor to retrieve the device specification,
        which is then use to create a `SerializableDevice` that will validate
        that operations are supported and use the correct qubits.
        """
        return self._device

    def get_device_specification(self) -> Optional[v2.device_pb2.DeviceSpecification]:
        raise NotImplemented  # coverage: ignore

    def health(self):
        return 'OK'

    def list_calibrations(
        self,
        earliest_timestamp_seconds: Optional[int] = None,
        latest_timestamp_seconds: Optional[int] = None,
    ) -> List[calibration.Calibration]:
        calibration_list: List[calibration.Calibration] = []
        for calibration_seconds in self._calibrations:
            if (
                earliest_timestamp_seconds is not None
                and earliest_timestamp_seconds > calibration_seconds
            ):
                continue
            if (
                latest_timestamp_seconds is not None
                and latest_timestamp_seconds < calibration_seconds
            ):
                continue
            calibration_list.append(self._calibrations[calibration_seconds])
        return calibration_list

    def get_sampler(self, gate_set: Optional['Serializer'] = None) -> cirq.Sampler:
        return self._sampler

    def supported_languages(self) -> List[str]:
        return VALID_LANGUAGES

    def list_programs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
    ) -> List[AbstractLocalProgram]:
        programs: List[AbstractLocalProgram] = []
        for program in self._programs.values():
            if created_before is not None and created_before < program.create_time():
                continue
            if created_after is not None and created_after > program.create_time():
                continue
            if has_labels is not None:
                labels = program.labels()
                if any(key not in labels or labels[key] != has_labels[key] for key in has_labels):
                    continue
            programs.append(program)
        return programs

    def get_program(self, program_id: str) -> AbstractProgram:
        """Returns an AbstractProgram for an existing Quantum Engine program.

        Args:
            program_id: Unique ID of the program within the parent project.

        Returns:
            An AbstractProgram for the program.

        Raises:
            KeyError: if program is not found
        """
        return self._programs[program_id]

    def run_batch(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        params_list: List[cirq.Sweepable] = None,
        repetitions: int = 1,
        gate_set: Optional['Serializer'] = None,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
    ) -> SimulatedLocalJob:
        if program_id is None:
            program_id = self._create_id(id_type='program')
        if job_id is None:
            job_id = self._create_id(id_type='job')
        self._programs[program_id] = SimulatedLocalProgram(
            program_id=program_id,
            simulation_type=self._simulation_type,
            circuits=programs,
            engine=self.engine(),
            processor=self,
        )
        job = SimulatedLocalJob(
            job_id=job_id,
            processor_id=self.processor_id,
            parent_program=self._programs[program_id],
            repetitions=repetitions,
            sweeps=params_list,
            sampler=self._sampler,
            simulation_type=self._simulation_type,
        )
        self._programs[program_id].add_job(job_id, job)
        return job

    def run(
        self,
        program: cirq.Circuit,
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        param_resolver: cirq.ParamResolver = cirq.ParamResolver({}),
        repetitions: int = 1,
        gate_set: Optional['Serializer'] = None,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
    ) -> cirq.Result:
        """Runs the supplied Circuit on this processor.

        Args:
            program: The Circuit to execute. If a circuit is
                provided, a moment by moment schedule will be used.
            program_id: A user-provided identifier for the program. This must
                be unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-################YYMMDD' will be generated, where # is
                alphanumeric and YYMMDD is the current year, month, and day.
            job_id: Job identifier to use. If this is not provided, a random id
                of the format 'job-################YYMMDD' will be generated,
                where # is alphanumeric and YYMMDD is the current year, month,
                and day.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of labels to set on the job.
        Returns:
            A single Result for this run.
        """
        return self.run_sweep(
            program=program,
            program_id=program_id,
            job_id=job_id,
            params=[param_resolver],
            repetitions=repetitions,
            gate_set=gate_set,
            program_description=program_description,
            program_labels=program_labels,
            job_description=job_description,
            job_labels=job_labels,
        ).results()[0]

    def run_sweep(
        self,
        program: cirq.Circuit,
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        params: cirq.Sweepable = None,
        repetitions: int = 1,
        gate_set: Optional['Serializer'] = None,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
    ) -> SimulatedLocalJob:
        if program_id is None:
            program_id = self._create_id(id_type='program')
        if job_id is None:
            job_id = self._create_id(id_type='job')
        self._programs[program_id] = SimulatedLocalProgram(
            program_id=program_id,
            simulation_type=self._simulation_type,
            circuits=[program],
            processor=self,
            engine=self.engine(),
        )
        job = SimulatedLocalJob(
            job_id=job_id,
            processor_id=self.processor_id,
            parent_program=self._programs[program_id],
            repetitions=repetitions,
            sweeps=[params],
            sampler=self._sampler,
            simulation_type=self._simulation_type,
        )
        self._programs[program_id].add_job(job_id, job)
        return job

    def run_calibration(self, *args, **kwargs):
        raise NotImplemented  # coverage: ignore
