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

import cirq_google as cg
from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook


def test_get_device_sampler():
    result = get_qcs_objects_for_notebook(virtual=True)
    assert isinstance(result.device, cg.GridDevice)
    assert not result.signed_in
    assert isinstance(result.sampler, cg.ProcessorSampler)
    assert result.is_simulator
    assert result.processor_id == 'rainbow'
    assert result.project_id == 'fake_project'

    result = get_qcs_objects_for_notebook(processor_id='weber', virtual=True)
    assert isinstance(result.device, cg.GridDevice)
    assert not result.signed_in
    assert isinstance(result.sampler, cg.ProcessorSampler)
    assert result.is_simulator
    assert result.processor_id == 'weber'
    assert result.project_id == 'fake_project'

    # Note: if running locally with application default credentials,
    # you actually will be signed_in, so it may not be a simulator
    result = get_qcs_objects_for_notebook()
    assert isinstance(result.device, cg.GridDevice)
