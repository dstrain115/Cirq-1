# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cirq/google/api/v1/params.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='cirq/google/api/v1/params.proto',
  package='cirq.google.api.v1',
  syntax='proto3',
  serialized_options=b'\n\035com.google.cirq.google.api.v1B\013ParamsProtoP\001',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1f\x63irq/google/api/v1/params.proto\x12\x12\x63irq.google.api.v1\"V\n\x0eParameterSweep\x12\x13\n\x0brepetitions\x18\x01 \x01(\x05\x12/\n\x05sweep\x18\x02 \x01(\x0b\x32 .cirq.google.api.v1.ProductSweep\"=\n\x0cProductSweep\x12-\n\x07\x66\x61\x63tors\x18\x01 \x03(\x0b\x32\x1c.cirq.google.api.v1.ZipSweep\";\n\x08ZipSweep\x12/\n\x06sweeps\x18\x01 \x03(\x0b\x32\x1f.cirq.google.api.v1.SingleSweep\"\x8d\x01\n\x0bSingleSweep\x12\x15\n\rparameter_key\x18\x01 \x01(\t\x12,\n\x06points\x18\x02 \x01(\x0b\x32\x1a.cirq.google.api.v1.PointsH\x00\x12\x30\n\x08linspace\x18\x03 \x01(\x0b\x32\x1c.cirq.google.api.v1.LinspaceH\x00\x42\x07\n\x05sweep\"\x18\n\x06Points\x12\x0e\n\x06points\x18\x01 \x03(\x02\"G\n\x08Linspace\x12\x13\n\x0b\x66irst_point\x18\x01 \x01(\x02\x12\x12\n\nlast_point\x18\x02 \x01(\x02\x12\x12\n\nnum_points\x18\x03 \x01(\x03\"\x8c\x01\n\rParameterDict\x12G\n\x0b\x61ssignments\x18\x01 \x03(\x0b\x32\x32.cirq.google.api.v1.ParameterDict.AssignmentsEntry\x1a\x32\n\x10\x41ssignmentsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\x42.\n\x1d\x63om.google.cirq.google.api.v1B\x0bParamsProtoP\x01\x62\x06proto3'
)




_PARAMETERSWEEP = _descriptor.Descriptor(
  name='ParameterSweep',
  full_name='cirq.google.api.v1.ParameterSweep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='repetitions', full_name='cirq.google.api.v1.ParameterSweep.repetitions', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sweep', full_name='cirq.google.api.v1.ParameterSweep.sweep', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=55,
  serialized_end=141,
)


_PRODUCTSWEEP = _descriptor.Descriptor(
  name='ProductSweep',
  full_name='cirq.google.api.v1.ProductSweep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='factors', full_name='cirq.google.api.v1.ProductSweep.factors', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=143,
  serialized_end=204,
)


_ZIPSWEEP = _descriptor.Descriptor(
  name='ZipSweep',
  full_name='cirq.google.api.v1.ZipSweep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='sweeps', full_name='cirq.google.api.v1.ZipSweep.sweeps', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=206,
  serialized_end=265,
)


_SINGLESWEEP = _descriptor.Descriptor(
  name='SingleSweep',
  full_name='cirq.google.api.v1.SingleSweep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='parameter_key', full_name='cirq.google.api.v1.SingleSweep.parameter_key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='points', full_name='cirq.google.api.v1.SingleSweep.points', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='linspace', full_name='cirq.google.api.v1.SingleSweep.linspace', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
    _descriptor.OneofDescriptor(
      name='sweep', full_name='cirq.google.api.v1.SingleSweep.sweep',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=268,
  serialized_end=409,
)


_POINTS = _descriptor.Descriptor(
  name='Points',
  full_name='cirq.google.api.v1.Points',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='points', full_name='cirq.google.api.v1.Points.points', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=411,
  serialized_end=435,
)


_LINSPACE = _descriptor.Descriptor(
  name='Linspace',
  full_name='cirq.google.api.v1.Linspace',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='first_point', full_name='cirq.google.api.v1.Linspace.first_point', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='last_point', full_name='cirq.google.api.v1.Linspace.last_point', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num_points', full_name='cirq.google.api.v1.Linspace.num_points', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=437,
  serialized_end=508,
)


_PARAMETERDICT_ASSIGNMENTSENTRY = _descriptor.Descriptor(
  name='AssignmentsEntry',
  full_name='cirq.google.api.v1.ParameterDict.AssignmentsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='cirq.google.api.v1.ParameterDict.AssignmentsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='cirq.google.api.v1.ParameterDict.AssignmentsEntry.value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=601,
  serialized_end=651,
)

_PARAMETERDICT = _descriptor.Descriptor(
  name='ParameterDict',
  full_name='cirq.google.api.v1.ParameterDict',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='assignments', full_name='cirq.google.api.v1.ParameterDict.assignments', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_PARAMETERDICT_ASSIGNMENTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=511,
  serialized_end=651,
)

_PARAMETERSWEEP.fields_by_name['sweep'].message_type = _PRODUCTSWEEP
_PRODUCTSWEEP.fields_by_name['factors'].message_type = _ZIPSWEEP
_ZIPSWEEP.fields_by_name['sweeps'].message_type = _SINGLESWEEP
_SINGLESWEEP.fields_by_name['points'].message_type = _POINTS
_SINGLESWEEP.fields_by_name['linspace'].message_type = _LINSPACE
_SINGLESWEEP.oneofs_by_name['sweep'].fields.append(
  _SINGLESWEEP.fields_by_name['points'])
_SINGLESWEEP.fields_by_name['points'].containing_oneof = _SINGLESWEEP.oneofs_by_name['sweep']
_SINGLESWEEP.oneofs_by_name['sweep'].fields.append(
  _SINGLESWEEP.fields_by_name['linspace'])
_SINGLESWEEP.fields_by_name['linspace'].containing_oneof = _SINGLESWEEP.oneofs_by_name['sweep']
_PARAMETERDICT_ASSIGNMENTSENTRY.containing_type = _PARAMETERDICT
_PARAMETERDICT.fields_by_name['assignments'].message_type = _PARAMETERDICT_ASSIGNMENTSENTRY
DESCRIPTOR.message_types_by_name['ParameterSweep'] = _PARAMETERSWEEP
DESCRIPTOR.message_types_by_name['ProductSweep'] = _PRODUCTSWEEP
DESCRIPTOR.message_types_by_name['ZipSweep'] = _ZIPSWEEP
DESCRIPTOR.message_types_by_name['SingleSweep'] = _SINGLESWEEP
DESCRIPTOR.message_types_by_name['Points'] = _POINTS
DESCRIPTOR.message_types_by_name['Linspace'] = _LINSPACE
DESCRIPTOR.message_types_by_name['ParameterDict'] = _PARAMETERDICT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ParameterSweep = _reflection.GeneratedProtocolMessageType('ParameterSweep', (_message.Message,), {
  'DESCRIPTOR' : _PARAMETERSWEEP,
  '__module__' : 'cirq.google.api.v1.params_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v1.ParameterSweep)
  })
_sym_db.RegisterMessage(ParameterSweep)

ProductSweep = _reflection.GeneratedProtocolMessageType('ProductSweep', (_message.Message,), {
  'DESCRIPTOR' : _PRODUCTSWEEP,
  '__module__' : 'cirq.google.api.v1.params_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v1.ProductSweep)
  })
_sym_db.RegisterMessage(ProductSweep)

ZipSweep = _reflection.GeneratedProtocolMessageType('ZipSweep', (_message.Message,), {
  'DESCRIPTOR' : _ZIPSWEEP,
  '__module__' : 'cirq.google.api.v1.params_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v1.ZipSweep)
  })
_sym_db.RegisterMessage(ZipSweep)

SingleSweep = _reflection.GeneratedProtocolMessageType('SingleSweep', (_message.Message,), {
  'DESCRIPTOR' : _SINGLESWEEP,
  '__module__' : 'cirq.google.api.v1.params_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v1.SingleSweep)
  })
_sym_db.RegisterMessage(SingleSweep)

Points = _reflection.GeneratedProtocolMessageType('Points', (_message.Message,), {
  'DESCRIPTOR' : _POINTS,
  '__module__' : 'cirq.google.api.v1.params_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v1.Points)
  })
_sym_db.RegisterMessage(Points)

Linspace = _reflection.GeneratedProtocolMessageType('Linspace', (_message.Message,), {
  'DESCRIPTOR' : _LINSPACE,
  '__module__' : 'cirq.google.api.v1.params_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v1.Linspace)
  })
_sym_db.RegisterMessage(Linspace)

ParameterDict = _reflection.GeneratedProtocolMessageType('ParameterDict', (_message.Message,), {

  'AssignmentsEntry' : _reflection.GeneratedProtocolMessageType('AssignmentsEntry', (_message.Message,), {
    'DESCRIPTOR' : _PARAMETERDICT_ASSIGNMENTSENTRY,
    '__module__' : 'cirq.google.api.v1.params_pb2'
    # @@protoc_insertion_point(class_scope:cirq.google.api.v1.ParameterDict.AssignmentsEntry)
    })
  ,
  'DESCRIPTOR' : _PARAMETERDICT,
  '__module__' : 'cirq.google.api.v1.params_pb2'
  # @@protoc_insertion_point(class_scope:cirq.google.api.v1.ParameterDict)
  })
_sym_db.RegisterMessage(ParameterDict)
_sym_db.RegisterMessage(ParameterDict.AssignmentsEntry)


DESCRIPTOR._options = None
_PARAMETERDICT_ASSIGNMENTSENTRY._options = None
# @@protoc_insertion_point(module_scope)
