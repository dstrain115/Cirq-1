# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cirq/google/api/v2/calibration.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from cirq.google.api.v2 import metrics_pb2 as cirq_dot_google_dot_api_dot_v2_dot_metrics__pb2
from cirq.google.api.v2 import program_pb2 as cirq_dot_google_dot_api_dot_v2_dot_program__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='cirq/google/api/v2/calibration.proto',
  package='cirq.google.api.v2',
  syntax='proto3',
  serialized_options=_b('\n\035com.google.cirq.google.api.v2B\027FocusedCalibrationProtoP\001'),
  serialized_pb=_b('\n$cirq/google/api/v2/calibration.proto\x12\x12\x63irq.google.api.v2\x1a cirq/google/api/v2/metrics.proto\x1a cirq/google/api/v2/program.proto\"J\n\x12\x46ocusedCalibration\x12\x34\n\x06layers\x18\x01 \x03(\x0b\x32$.cirq.google.api.v2.CalibrationLayer\"X\n\x10\x43\x61librationLayer\x12\x18\n\x10\x63\x61libration_type\x18\x01 \x01(\t\x12*\n\x05layer\x18\x02 \x01(\x0b\x32\x1b.cirq.google.api.v2.Program\"W\n\x18\x46ocusedCalibrationResult\x12;\n\x07results\x18\x01 \x03(\x0b\x32*.cirq.google.api.v2.CalibrationLayerResult\"\xc4\x01\n\x16\x43\x61librationLayerResult\x12\x36\n\x04\x63ode\x18\x01 \x01(\x0e\x32(.cirq.google.api.v2.CalibrationLayerCode\x12\x15\n\rerror_message\x18\x02 \x01(\t\x12\r\n\x05token\x18\x03 \x01(\t\x12\x34\n\x07metrics\x18\x04 \x01(\x0b\x32#.cirq.google.api.v2.MetricsSnapshot\x12\x16\n\x0evalid_until_ms\x18\x05 \x01(\x04*\xa7\x01\n\x14\x43\x61librationLayerCode\x12\"\n\x1e\x43\x41LIBRATION_RESULT_UNSPECIFIED\x10\x00\x12\x0b\n\x07SUCCESS\x10\x01\x12\x0f\n\x0b\x45RROR_OTHER\x10\x02\x12\x1c\n\x18\x45RROR_INVALID_PARAMETERS\x10\x03\x12\x11\n\rERROR_TIMEOUT\x10\x04\x12\x1c\n\x18\x45RROR_CALIBRATION_FAILED\x10\x05\x42:\n\x1d\x63om.google.cirq.google.api.v2B\x17\x46ocusedCalibrationProtoP\x01\x62\x06proto3')
  ,
  dependencies=[cirq_dot_google_dot_api_dot_v2_dot_metrics__pb2.DESCRIPTOR,cirq_dot_google_dot_api_dot_v2_dot_program__pb2.DESCRIPTOR,])

_CALIBRATIONLAYERCODE = _descriptor.EnumDescriptor(
  name='CalibrationLayerCode',
  full_name='cirq.google.api.v2.CalibrationLayerCode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CALIBRATION_RESULT_UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUCCESS', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR_OTHER', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR_INVALID_PARAMETERS', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR_TIMEOUT', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR_CALIBRATION_FAILED', index=5, number=5,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=583,
  serialized_end=750,
)
_sym_db.RegisterEnumDescriptor(_CALIBRATIONLAYERCODE)

CalibrationLayerCode = enum_type_wrapper.EnumTypeWrapper(_CALIBRATIONLAYERCODE)
CALIBRATION_RESULT_UNSPECIFIED = 0
SUCCESS = 1
ERROR_OTHER = 2
ERROR_INVALID_PARAMETERS = 3
ERROR_TIMEOUT = 4
ERROR_CALIBRATION_FAILED = 5



_FOCUSEDCALIBRATION = _descriptor.Descriptor(
  name='FocusedCalibration',
  full_name='cirq.google.api.v2.FocusedCalibration',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='layers', full_name='cirq.google.api.v2.FocusedCalibration.layers', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=128,
  serialized_end=202,
)


_CALIBRATIONLAYER = _descriptor.Descriptor(
  name='CalibrationLayer',
  full_name='cirq.google.api.v2.CalibrationLayer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='calibration_type', full_name='cirq.google.api.v2.CalibrationLayer.calibration_type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='layer', full_name='cirq.google.api.v2.CalibrationLayer.layer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=204,
  serialized_end=292,
)


_FOCUSEDCALIBRATIONRESULT = _descriptor.Descriptor(
  name='FocusedCalibrationResult',
  full_name='cirq.google.api.v2.FocusedCalibrationResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='results', full_name='cirq.google.api.v2.FocusedCalibrationResult.results', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=294,
  serialized_end=381,
)


_CALIBRATIONLAYERRESULT = _descriptor.Descriptor(
  name='CalibrationLayerResult',
  full_name='cirq.google.api.v2.CalibrationLayerResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='cirq.google.api.v2.CalibrationLayerResult.code', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='error_message', full_name='cirq.google.api.v2.CalibrationLayerResult.error_message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='token', full_name='cirq.google.api.v2.CalibrationLayerResult.token', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='metrics', full_name='cirq.google.api.v2.CalibrationLayerResult.metrics', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='valid_until_ms', full_name='cirq.google.api.v2.CalibrationLayerResult.valid_until_ms', index=4,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=384,
  serialized_end=580,
)

_FOCUSEDCALIBRATION.fields_by_name['layers'].message_type = _CALIBRATIONLAYER
_CALIBRATIONLAYER.fields_by_name['layer'].message_type = cirq_dot_google_dot_api_dot_v2_dot_program__pb2._PROGRAM
_FOCUSEDCALIBRATIONRESULT.fields_by_name['results'].message_type = _CALIBRATIONLAYERRESULT
_CALIBRATIONLAYERRESULT.fields_by_name['code'].enum_type = _CALIBRATIONLAYERCODE
_CALIBRATIONLAYERRESULT.fields_by_name['metrics'].message_type = cirq_dot_google_dot_api_dot_v2_dot_metrics__pb2._METRICSSNAPSHOT
DESCRIPTOR.message_types_by_name['FocusedCalibration'] = _FOCUSEDCALIBRATION
DESCRIPTOR.message_types_by_name['CalibrationLayer'] = _CALIBRATIONLAYER
DESCRIPTOR.message_types_by_name['FocusedCalibrationResult'] = _FOCUSEDCALIBRATIONRESULT
DESCRIPTOR.message_types_by_name['CalibrationLayerResult'] = _CALIBRATIONLAYERRESULT
DESCRIPTOR.enum_types_by_name['CalibrationLayerCode'] = _CALIBRATIONLAYERCODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FocusedCalibration = _reflection.GeneratedProtocolMessageType('FocusedCalibration', (_message.Message,), {
  'DESCRIPTOR' : _FOCUSEDCALIBRATION,
  '__module__' : 'cirq.google.api.v2.calibration_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v2.FocusedCalibration)
  })
_sym_db.RegisterMessage(FocusedCalibration)

CalibrationLayer = _reflection.GeneratedProtocolMessageType('CalibrationLayer', (_message.Message,), {
  'DESCRIPTOR' : _CALIBRATIONLAYER,
  '__module__' : 'cirq.google.api.v2.calibration_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v2.CalibrationLayer)
  })
_sym_db.RegisterMessage(CalibrationLayer)

FocusedCalibrationResult = _reflection.GeneratedProtocolMessageType('FocusedCalibrationResult', (_message.Message,), {
  'DESCRIPTOR' : _FOCUSEDCALIBRATIONRESULT,
  '__module__' : 'cirq.google.api.v2.calibration_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v2.FocusedCalibrationResult)
  })
_sym_db.RegisterMessage(FocusedCalibrationResult)

CalibrationLayerResult = _reflection.GeneratedProtocolMessageType('CalibrationLayerResult', (_message.Message,), {
  'DESCRIPTOR' : _CALIBRATIONLAYERRESULT,
  '__module__' : 'cirq.google.api.v2.calibration_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v2.CalibrationLayerResult)
  })
_sym_db.RegisterMessage(CalibrationLayerResult)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
