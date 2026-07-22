# Copyright 2026 The Cirq Developers
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

import numpy as np
import pytest
import sympy

import cirq


def test_set_variable_init_and_properties():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    op = cirq.SetVariable(a, b + 1)
    assert op.target == a
    assert op.expression == b + 1
    assert op.qubits == ()
    assert op.with_qubits() == op

    with pytest.raises(ValueError, match="does not support qubits"):
        op.with_qubits(cirq.LineQubit(0))

    with pytest.raises(TypeError, match="target must be a sympy.Symbol"):
        cirq.SetVariable("not_a_symbol", 1)


def test_set_variable_repr_and_str():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    op = cirq.SetVariable(a, b + 1)
    assert str(op) == "SetVariable(a, b + 1)"
    assert repr(op) == "cirq.SetVariable(sympy.Symbol('a'), sympy.Symbol('b') + 1)"


def test_set_variable_simulation_constant():
    q = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(cirq.SetVariable(a, 1.0), cirq.Rx(rads=a).on(q))
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    expected_state = cirq.Circuit(cirq.Rx(rads=1.0).on(q)).final_state_vector()
    np.testing.assert_allclose(result.final_state_vector, expected_state, atol=1e-6)


def test_set_variable_simulation_expression():
    q = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    circuit = cirq.Circuit(cirq.SetVariable(b, a * 2), cirq.Rx(rads=b).on(q))
    sim = cirq.Simulator()
    result = sim.simulate(circuit, param_resolver={'a': 0.5})
    expected_state = cirq.Circuit(cirq.Rx(rads=1.0).on(q)).final_state_vector()
    np.testing.assert_allclose(result.final_state_vector, expected_state, atol=1e-6)


def test_set_variable_simulation_measurement_dependency():
    q0, q1 = cirq.LineQubit.range(2)
    a = sympy.Symbol('a')

    # We must explicitly separate X, measure, SetVariable, and Rx into separate moments
    # to guarantee correct execution order (since SetVariable has no qubits and would
    # otherwise be aligned alongside measurement).
    circuit = cirq.Circuit(
        cirq.Moment(cirq.X(q0)),
        cirq.Moment(cirq.measure(q0, key='m')),
        cirq.Moment(cirq.SetVariable(a, sympy.Symbol('m') * np.pi)),
        cirq.Moment(cirq.Rx(rads=a).on(q1)),
    )
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    # q0 is 1, so m=1, a=pi, Rx(pi) on q1 rotates |0> to -i|1>.
    # So q1 final state is |1> (up to phase).
    expected_state = cirq.Circuit(cirq.X(q0), cirq.Rx(rads=np.pi).on(q1)).final_state_vector()
    np.testing.assert_allclose(np.abs(result.final_state_vector), np.abs(expected_state), atol=1e-6)


def test_set_variable_json():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    op = cirq.SetVariable(a, b + 1)
    json_text = cirq.to_json(op)
    restored_op = cirq.read_json(json_text=json_text)
    assert restored_op == op
