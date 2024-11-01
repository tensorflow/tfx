# Copyright 2023 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests MakeProtoPlaceholder end to end, including resolution."""

import base64
import functools
import importlib
import os
import pytest
from typing import Any, Optional, TypeVar, Union

import tensorflow as tf
from tfx.dsl.compiler import placeholder_utils
from tfx.dsl.placeholder import placeholder as ph
from tfx.dsl.placeholder import proto_placeholder
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import execution_invocation_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import proto_utils

from google.protobuf import descriptor_pb2
from google.protobuf import empty_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2



@pytest.fixture(autouse=True,scope="module")
def cleanup():
    yield
    importlib.reload(pipeline_pb2)

_ExecutionInvocation = functools.partial(
    ph.make_proto, execution_invocation_pb2.ExecutionInvocation()
)
_MetadataStoreValue = functools.partial(
    ph.make_proto, metadata_store_pb2.Value()
)
_UpdateOptions = functools.partial(
    ph.make_proto, pipeline_pb2.UpdateOptions()
)

_P = TypeVar('_P', bound=message.Message)


def resolve(
    p: ph.Placeholder, exec_properties: Optional[dict[str, Any]] = None
) -> Any:
  """Resolves the given placeholder."""
  if isinstance(p, proto_placeholder.MakeProtoPlaceholder):
    p = p.serialize(ph.ProtoSerializationFormat.TEXT_FORMAT)
  return placeholder_utils.resolve_placeholder_expression(
      p.encode(),
      placeholder_utils.ResolutionContext(
          exec_info=data_types.ExecutionInfo(
              pipeline_run_id='test-run-id',
              exec_properties=exec_properties,
              pipeline_info=pipeline_pb2.PipelineInfo(id='test-pipeline-id'),
          ),
      ),
  )


def validate_and_get_descriptors(
    p: ph.Placeholder,
) -> descriptor_pb2.FileDescriptorSet:
  assert isinstance(p, proto_placeholder.MakeProtoPlaceholder)
  op = p.encode().operator.make_proto_op
  assert op.HasField('file_descriptors')

  # Make sure the generated descriptors can be loaded into a fresh pool.
  try:
    proto_utils.get_pool_with_descriptors(
        op.file_descriptors, descriptor_pool.DescriptorPool()
    )
  except Exception as e:
    raise ValueError(f'Got invalid descriptors: {op.file_descriptors}') from e

  return op.file_descriptors


def parse_text_proto(
    textproto: str,
    proto_class: type[_P] = execution_invocation_pb2.ExecutionInvocation,
) -> _P:
  """Parses and returns a textproto."""
  return text_format.Parse(textproto, proto_class())


# This test covers the DSL side (proto_placeholder.py) that runs at pipeline
# construction time _plus_ the resolution side (placeholder_utils.py) that runs
# at pipeline runtime. There are additional DSL-only test cases in
# ./placeholder_test.py and additional resolution-only test cases in
# dsl/compiler/placeholder_utils_test.py
class MakeProtoPlaceholderTest(tf.test.TestCase):

  def test_Empty(self):
    self.assertEqual(
        '',
        resolve(
            ph.make_proto(execution_invocation_pb2.ExecutionInvocation())
        ),
    )

  def test_BaseOnly(self):
    actual = resolve(
        ph.make_proto(
            execution_invocation_pb2.ExecutionInvocation(tmp_dir='/foo')
        )
    )
    self.assertProtoEquals(
        """
        tmp_dir: "/foo"
        """,
        parse_text_proto(actual),
    )

  def test_FieldOnly(self):
    actual = resolve(_ExecutionInvocation(tmp_dir='/foo'))
    self.assertProtoEquals(
        """
        tmp_dir: "/foo"
        """,
        parse_text_proto(actual),
    )

  def test_ScalarFieldTypes(self):
    def _resolve_and_parse(p: ph.Placeholder) -> metadata_store_pb2.Value:
      return parse_text_proto(resolve(p), metadata_store_pb2.Value)

    # Can't do them all at once due to oneof.
    self.assertProtoEquals(
        'int_value: 42',
        _resolve_and_parse(_MetadataStoreValue(int_value=42)),
    )
    self.assertProtoEquals(
        'double_value: 42.42',
        _resolve_and_parse(_MetadataStoreValue(double_value=42.42)),
    )
    self.assertProtoEquals(
        'string_value: "foo42"',
        _resolve_and_parse(_MetadataStoreValue(string_value='foo42')),
    )
    self.assertProtoEquals(
        'bool_value: true',
        _resolve_and_parse(_MetadataStoreValue(bool_value=True)),
    )

  def test_EnumField(self):
    actual = resolve(
        _UpdateOptions(reload_policy=pipeline_pb2.UpdateOptions.PARTIAL)
    )
    self.assertProtoEquals(
        """
        reload_policy: PARTIAL
        """,
        parse_text_proto(actual, pipeline_pb2.UpdateOptions),
    )

  def test_FieldPlaceholder(self):
    actual = resolve(
        _ExecutionInvocation(tmp_dir=ph.execution_invocation().pipeline_run_id)
    )
    self.assertProtoEquals(
        """
        tmp_dir: "test-run-id"
        """,
        parse_text_proto(actual),
    )

  def test_EnumStringPlaceholder(self):
    actual = resolve(
        _UpdateOptions(reload_policy=ph.exec_property('reload_policy')),
        exec_properties={'reload_policy': 'ALL'},
    )
    self.assertProtoEquals(
        """
        reload_policy: ALL
        """,
        parse_text_proto(actual, pipeline_pb2.UpdateOptions),
    )

  def test_EnumIntPlaceholder(self):
    actual = resolve(
        _UpdateOptions(reload_policy=ph.exec_property('reload_policy')),
        exec_properties={'reload_policy': 1},
    )
    self.assertProtoEquals(
        """
        reload_policy: PARTIAL
        """,
        parse_text_proto(actual, pipeline_pb2.UpdateOptions),
    )

  def test_EmptyFieldPlaceholder(self):
    actual = resolve(
        _ExecutionInvocation(tmp_dir=ph.execution_invocation().frontend_url)
    )
    self.assertProtoEquals(
        """
        tmp_dir: ""
        """,
        parse_text_proto(actual),
    )

  def test_NoneIntoOptionalField(self):
    actual = resolve(_ExecutionInvocation(tmp_dir=None))
    self.assertProtoEquals('', parse_text_proto(actual))

  def test_NonePlaceholderIntoOptionalField(self):
    actual = resolve(
        _ExecutionInvocation(tmp_dir=ph.execution_invocation().frontend_url)
    )
    self.assertProtoEquals('', parse_text_proto(actual))

  def test_NoneExecPropIntoOptionalField(self):
    # When an exec prop has type Union[T, None] and the user passes None, it is
    # actually completely absent from the exec_properties dict in
    # ExecutionInvocation. See also b/172001324 and the corresponding todo in
    # placeholder_utils.py.
    actual = resolve(
        _UpdateOptions(reload_policy=ph.exec_property('reload_policy')),
        exec_properties={},  # Intentionally empty.
    )
    self.assertProtoEquals(
        '',
        parse_text_proto(actual, pipeline_pb2.UpdateOptions),
    )

  def test_BareSubmessage(self):
    actual = resolve(
        _ExecutionInvocation(
            pipeline_info=pipeline_pb2.PipelineInfo(id='foo-id')
        )
    )
    self.assertProtoEquals(
        """
        pipeline_info {
          id: "foo-id"
        }
        """,
        parse_text_proto(actual),
    )

  def test_SubmessageDict(self):
    actual = resolve(_ExecutionInvocation(pipeline_info=dict(id='foo-id')))
    self.assertProtoEquals(
        """
        pipeline_info {
          id: "foo-id"
        }
        """,
        parse_text_proto(actual),
    )

  def test_SubmessageMakeProtoPlaceholder(self):
    actual = resolve(
        _ExecutionInvocation(
            pipeline_info=ph.make_proto(
                pipeline_pb2.PipelineInfo(),
                id=ph.execution_invocation().pipeline_run_id,
            )
        )
    )
    self.assertProtoEquals(
        """
        pipeline_info {
          id: "test-run-id"
        }
        """,
        parse_text_proto(actual),
    )

  def test_SubmessageProtoGetterPlaceholder(self):
    with self.assertRaises(ValueError):
      resolve(
          _ExecutionInvocation(
              # Assigning an entire sub-proto (PipelineInfo in this case) from a
              # non-make_proto placeholder is currently not supported. Though
              # it could be, see b/327639307#comment26.
              pipeline_info=ph.execution_invocation().pipeline_info
          )
      )

  def test_SubmessageOverwrite(self):
    actual = resolve(
        ph.make_proto(
            execution_invocation_pb2.ExecutionInvocation(
                pipeline_info=pipeline_pb2.PipelineInfo(
                    id='this will be overwritten'
                )
            ),
            pipeline_info=ph.make_proto(
                pipeline_pb2.PipelineInfo(),
                id=ph.execution_invocation().pipeline_run_id,
            ),
        )
    )
    self.assertProtoEquals(
        """
        pipeline_info {
          id: "test-run-id"
        }
        """,
        parse_text_proto(actual),
    )

  def test_NoneIntoSubmessage(self):
    actual = resolve(_ExecutionInvocation(pipeline_info=None))
    self.assertProtoEquals('', parse_text_proto(actual))

  def test_RepeatedField(self):
    actual = resolve(
        ph.make_proto(
            execution_invocation_pb2.ExecutionInvocation(
                pipeline_node=pipeline_pb2.PipelineNode(
                    upstream_nodes=['a', 'b']
                )
            ),
            pipeline_node=ph.make_proto(
                pipeline_pb2.PipelineNode(),
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                    ph.execution_invocation().pipeline_run_id + '-bar',
                ],
            ),
        )
    )
    self.assertProtoEquals(
        """
        pipeline_node {
          upstream_nodes: "a"
          upstream_nodes: "b"
          upstream_nodes: "test-run-id-foo"
          upstream_nodes: "test-run-id-bar"
        }
        """,
        parse_text_proto(actual),
    )

  def test_RepeatedFieldSingleItem(self):
    actual = resolve(
        _ExecutionInvocation(
            pipeline_node=ph.make_proto(
                pipeline_pb2.PipelineNode(),
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                ],
            ),
        )
    )
    self.assertProtoEquals(
        """
        pipeline_node {
          upstream_nodes: "test-run-id-foo"
        }
        """,
        parse_text_proto(actual),
    )

  def test_RepeatedFieldFalsyItem(self):
    actual = resolve(
        ph.make_proto(
            execution_invocation_pb2.ExecutionInvocation(
                pipeline_node=pipeline_pb2.PipelineNode(upstream_nodes=[''])
            ),
            pipeline_node=ph.make_proto(
                pipeline_pb2.PipelineNode(),
                upstream_nodes=[
                    ph.execution_invocation().frontend_url,
                ],
            ),
        )
    )
    self.assertProtoEquals(
        """
        pipeline_node {
          upstream_nodes: ""
          upstream_nodes: ""
        }
        """,
        parse_text_proto(actual),
    )

  def test_RepeatedFieldNoneItem(self):
    actual = resolve(
        ph.make_proto(
            execution_invocation_pb2.ExecutionInvocation(
                pipeline_node=pipeline_pb2.PipelineNode()
            ),
            pipeline_node=ph.make_proto(
                pipeline_pb2.PipelineNode(),
                upstream_nodes=[
                    'foo',
                    ph.exec_property('reload_policy'),  # Will be None.
                    'bar',
                ],
            ),
        ),
        exec_properties={},  # Intentionally empty.
    )
    self.assertProtoEquals(
        """
        pipeline_node {
          upstream_nodes: "foo"
          upstream_nodes: "bar"
        }
        """,
        parse_text_proto(actual),
    )

  def test_NoneIntoRepeatedField(self):
    actual = resolve(
        ph.make_proto(pipeline_pb2.PipelineNode(), upstream_nodes=None)
    )
    self.assertProtoEquals('', parse_text_proto(actual))

  def test_EmptyPlaceholderListIntoRepeatedField(self):
    actual = resolve(
        ph.make_proto(
            pipeline_pb2.PipelineNode(),
            upstream_nodes=ph.execution_invocation().pipeline_node.upstream_nodes,
        )
    )
    self.assertProtoEquals('', parse_text_proto(actual))

  def test_EmptyListPlaceholderIntoRepeatedField(self):
    actual = resolve(
        ph.make_proto(
            pipeline_pb2.PipelineNode(), upstream_nodes=ph.make_list([])
        )
    )
    self.assertProtoEquals('', parse_text_proto(actual))

  def test_RepeatedSubmessage(self):
    actual = resolve(
        ph.make_proto(
            pipeline_pb2.StructuralRuntimeParameter(),
            parts=[
                pipeline_pb2.StructuralRuntimeParameter.StringOrRuntimeParameter(
                    constant_value='foo'
                ),
                ph.make_proto(
                    pipeline_pb2.StructuralRuntimeParameter.StringOrRuntimeParameter(),
                    constant_value=ph.execution_invocation().pipeline_run_id,
                ),
            ],
        )
    )
    self.assertProtoEquals(
        """
        parts {
          constant_value: "foo"
        }
        parts {
          constant_value: "test-run-id"
        }
        """,
        parse_text_proto(actual, pipeline_pb2.StructuralRuntimeParameter),
    )

  def test_AnySubmessageBareMessage(self):
    actual = resolve(
        _MetadataStoreValue(
            proto_value=pipeline_pb2.PipelineNode(
                upstream_nodes=['foo', 'bar'],
            )
        )
    )
    self.assertProtoEquals(
        """
        proto_value {
          [type.googleapis.com/tfx.orchestration.PipelineNode] {
            upstream_nodes: "foo"
            upstream_nodes: "bar"
          }
        }
        """,
        parse_text_proto(actual, metadata_store_pb2.Value),
    )

  def test_AnySubmessagePlaceholder(self):
    actual = resolve(
        _MetadataStoreValue(
            # We can directly assign a message of any type and it will pack it.
            proto_value=ph.make_proto(
                pipeline_pb2.PipelineNode(),
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                ],
            )
        )
    )
    self.assertProtoEquals(
        """
        proto_value {
          [type.googleapis.com/tfx.orchestration.PipelineNode] {
            upstream_nodes: "test-run-id-foo"
          }
        }
        """,
        parse_text_proto(actual, metadata_store_pb2.Value),
    )

  def test_MapFieldScalarValue(self):
    actual = resolve(
        _ExecutionInvocation(
            extra_flags={
                'fookey': 'foovalue',
                'barkey': 'barvalue',
            }
        )
    )
    self.assertProtoEquals(
        """
        extra_flags {
          key: "fookey"
          value: "foovalue"
        }
        extra_flags {
          key: "barkey"
          value: "barvalue"
        }
        """,
        parse_text_proto(actual),
    )

  def test_MapFieldScalarPlaceholderValue(self):
    actual = resolve(
        _ExecutionInvocation(
            extra_flags={
                'fookey': ph.execution_invocation().pipeline_run_id,
                'barkey': 'bar-' + ph.execution_invocation().pipeline_run_id,
            }
        )
    )
    self.assertProtoEquals(
        """
        extra_flags {
          key: "fookey"
          value: "test-run-id"
        }
        extra_flags {
          key: "barkey"
          value: "bar-test-run-id"
        }
        """,
        parse_text_proto(actual),
    )

  def test_MapFieldScalarNoneValue(self):
    actual = resolve(
        _ExecutionInvocation(
            extra_flags={
                'fookey': ph.exec_property('reload_policy'),  # Will be None.
                'barkey': None,
                'notnone': 'this is not none',
            }
        ),
        exec_properties={},  # Intentionally empty.
    )
    self.assertProtoEquals(
        """
        extra_flags {
          key: "notnone"
          value: "this is not none"
        }
        """,
        parse_text_proto(actual),
    )

  def test_MapFieldSubmessageValue(self):
    actual = resolve(
        _ExecutionInvocation(
            execution_properties={
                'fookey': _MetadataStoreValue(
                    string_value=ph.execution_invocation().pipeline_run_id
                ),
                'barkey': metadata_store_pb2.Value(int_value=42),
            }
        )
    )
    self.assertProtoEquals(
        """
        execution_properties {
          key: "fookey"
          value {
            string_value: "test-run-id"
          }
        }
        execution_properties {
          key: "barkey"
          value {
            int_value: 42
          }
        }
        """,
        parse_text_proto(actual),
    )

  def test_MapFieldPlaceholderKey(self):
    actual = resolve(
        _ExecutionInvocation(
            extra_flags=[
                (ph.execution_invocation().pipeline_run_id, 'foovalue'),
            ]
        )
    )
    self.assertProtoEquals(
        """
        extra_flags {
          key: "test-run-id"
          value: "foovalue"
        }
        """,
        parse_text_proto(actual),
    )

  def test_RejectsMapFieldScalarNoneKey(self):
    with self.assertRaises(ValueError):
      resolve(
          _ExecutionInvocation(
              extra_flags=[(
                  ph.exec_property('reload_policy'),  # Will be None.
                  'foo',
              )]
          ),
          exec_properties={},  # Intentionally empty.
      )
    with self.assertRaises(ValueError):
      resolve(_ExecutionInvocation(extra_flags={None: 'foo'}))

  def test_MapFieldScalarValueEmpty(self):
    actual = resolve(_ExecutionInvocation(extra_flags={}))
    self.assertProtoEquals('', parse_text_proto(actual))
    actual = resolve(_ExecutionInvocation(extra_flags=[]))
    self.assertProtoEquals('', parse_text_proto(actual))

  def test_PlusItemGetter(self):
    actual = resolve(
        _ExecutionInvocation(
            pipeline_node=ph.make_proto(
                pipeline_pb2.PipelineNode(),
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                ],
            ),
            # It's a little silly, but it's something that should work:
            # Constructing an entire proto through placeholders, only to
            # read out a single field from it:
        ).pipeline_node.upstream_nodes[0]
    )
    self.assertProtoEquals('test-run-id-foo', actual)

  def test_BinarySerializationBase64(self):
    actual = resolve(
        ph.make_proto(
            execution_invocation_pb2.ExecutionInvocation(
                tmp_dir='/foo',
                pipeline_node=pipeline_pb2.PipelineNode(
                    upstream_nodes=['a', 'b']
                ),
            ),
            pipeline_node=ph.make_proto(
                pipeline_pb2.PipelineNode(),
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                    ph.execution_invocation().pipeline_run_id + '-bar',
                ],
            ),
        )
        .serialize(ph.ProtoSerializationFormat.BINARY)
        .b64encode()
    )

    expected = execution_invocation_pb2.ExecutionInvocation(
        tmp_dir='/foo',
        pipeline_node=pipeline_pb2.PipelineNode(
            upstream_nodes=['a', 'b', 'test-run-id-foo', 'test-run-id-bar']
        ),
    ).SerializeToString()
    expected = base64.urlsafe_b64encode(expected).decode('ascii')

    self.assertEqual(expected, actual)

  def _normalize_descriptors(
      self, descriptor_set: descriptor_pb2.FileDescriptorSet
  ):
    """Evens out some differences between test environments."""
    for file in descriptor_set.file:
      # Depending on the environment where the test is run, the proto files may
      # be stored in different places. So we just strip away the entire
      # directory to make them compare successfully.
      file.name = os.path.basename(file.name)
      file.dependency[:] = [os.path.basename(dep) for dep in file.dependency]

      # The options may differ between environments and we don't need to assert
      # them.
      file.ClearField('options')
      for message_type in file.message_type:
        message_type.ClearField('options')
        for field in message_type.field:
          field.ClearField('options')

  def assertDescriptorsEqual(
      self,
      expected: Union[descriptor_pb2.FileDescriptorSet, str],
      actual: descriptor_pb2.FileDescriptorSet,
  ):
    """Compares descriptors with some tolerance for filenames and options."""
    if isinstance(expected, str):
      expected = text_format.Parse(expected, descriptor_pb2.FileDescriptorSet())
    self._normalize_descriptors(expected)
    self._normalize_descriptors(actual)
    self.assertProtoEquals(expected, actual)

  def test_ShrinksDescriptors_SimpleBaseMessage(self):
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/py/tfx/proto/orchestration/execution_invocation.proto"
          package: "tfx.orchestration"
          message_type {
            name: "ExecutionInvocation"
            field {
              name: "tmp_dir"
              number: 10
              label: LABEL_OPTIONAL
              type: TYPE_STRING
            }
            reserved_range {
              start: 1
              end: 2
            }
            reserved_range {
              start: 2
              end: 3
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(
            ph.make_proto(
                execution_invocation_pb2.ExecutionInvocation(tmp_dir='/foo')
            )
        ),
    )

  def test_ShrinksDescriptors_NestedBaseMessage(self):
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/py/tfx/proto/orchestration/pipeline.proto"
          package: "tfx.orchestration"
          message_type {
            name: "PipelineNode"
            field {
              name: "upstream_nodes"
              number: 7
              label: LABEL_REPEATED
              type: TYPE_STRING
            }
          }
          syntax: "proto3"
        }
        file {
          name: "third_party/py/tfx/proto/orchestration/execution_invocation.proto"
          package: "tfx.orchestration"
          dependency: "third_party/py/tfx/proto/orchestration/pipeline.proto"
          message_type {
            name: "ExecutionInvocation"
            field {
              name: "pipeline_node"
              number: 9
              label: LABEL_OPTIONAL
              type: TYPE_MESSAGE
              type_name: ".tfx.orchestration.PipelineNode"
            }
            reserved_range {
              start: 1
              end: 2
            }
            reserved_range {
              start: 2
              end: 3
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(
            ph.make_proto(
                execution_invocation_pb2.ExecutionInvocation(
                    pipeline_node=pipeline_pb2.PipelineNode(
                        upstream_nodes=['a', 'b'],
                    )
                )
            )
        ),
    )

  def test_ShrinksDescriptors_RepeatedFieldInBaseMessage(self):
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/py/tfx/proto/orchestration/pipeline.proto"
          package: "tfx.orchestration"
          message_type {
            name: "StructuralRuntimeParameter"
            field {
              name: "parts"
              number: 1
              label: LABEL_REPEATED
              type: TYPE_MESSAGE
              type_name: ".tfx.orchestration.StructuralRuntimeParameter.StringOrRuntimeParameter"
            }
            nested_type {
              name: "StringOrRuntimeParameter"
              field {
                name: "constant_value"
                number: 1
                label: LABEL_OPTIONAL
                type: TYPE_STRING
                oneof_index: 0
              }
              oneof_decl {
                name: "value"
              }
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(
            ph.make_proto(
                pipeline_pb2.StructuralRuntimeParameter(
                    parts=[
                        pipeline_pb2.StructuralRuntimeParameter.StringOrRuntimeParameter(
                            constant_value='foo',
                        )
                    ]
                )
            )
        ),
    )

  def test_ShrinksDescriptors_MapFieldInBaseMessage(self):
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/ml_metadata/proto/metadata_store.proto"
          package: "ml_metadata"
          message_type {
            name: "Value"
            field {
              name: "string_value"
              number: 3
              label: LABEL_OPTIONAL
              type: TYPE_STRING
              oneof_index: 0
            }
            oneof_decl {
              name: "value"
            }
          }
        }
        file {
          name: "third_party/py/tfx/proto/orchestration/execution_invocation.proto"
          package: "tfx.orchestration"
          dependency: "third_party/ml_metadata/proto/metadata_store.proto"
          message_type {
            name: "ExecutionInvocation"
            field {
              name: "execution_properties"
              number: 3
              label: LABEL_REPEATED
              type: TYPE_MESSAGE
              type_name: ".tfx.orchestration.ExecutionInvocation.ExecutionPropertiesEntry"
            }
            nested_type {
              name: "ExecutionPropertiesEntry"
              field {
                name: "key"
                number: 1
                label: LABEL_OPTIONAL
                type: TYPE_STRING
              }
              field {
                name: "value"
                number: 2
                label: LABEL_OPTIONAL
                type: TYPE_MESSAGE
                type_name: ".ml_metadata.Value"
              }
              options {
                map_entry: true
              }
            }
            reserved_range {
              start: 1
              end: 2
            }
            reserved_range {
              start: 2
              end: 3
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(
            ph.make_proto(
                execution_invocation_pb2.ExecutionInvocation(
                    execution_properties={
                        'foo': metadata_store_pb2.Value(string_value='bar'),
                    }
                )
            )
        ),
    )

  def test_ShrinksDescriptors_AnyFieldUnderBaseMessage(self):
    pb = metadata_store_pb2.Value()
    pb.proto_value.Pack(pipeline_pb2.PipelineNode(upstream_nodes=['a', 'b']))
    self.assertDescriptorsEqual(
        """
        file {
          name: "google/protobuf/any.proto"
          package: "google.protobuf"
          message_type {
            name: "Any"
            field {
              name: "type_url"
              number: 1
              label: LABEL_OPTIONAL
              type: TYPE_STRING
            }
            field {
              name: "value"
              number: 2
              label: LABEL_OPTIONAL
              type: TYPE_BYTES
            }
          }
          syntax: "proto3"
        }
        file {
          name: "third_party/ml_metadata/proto/metadata_store.proto"
          package: "ml_metadata"
          dependency: "google/protobuf/any.proto"
          message_type {
            name: "Value"
            field {
              name: "proto_value"
              number: 5
              label: LABEL_OPTIONAL
              type: TYPE_MESSAGE
              type_name: ".google.protobuf.Any"
              oneof_index: 0
            }
            oneof_decl {
              name: "value"
            }
          }
        }
        """,
        validate_and_get_descriptors(ph.make_proto(pb)),
    )

  def test_ShrinksDescriptors_SimplePlaceholder(self):
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/py/tfx/proto/orchestration/execution_invocation.proto"
          package: "tfx.orchestration"
          message_type {
            name: "ExecutionInvocation"
            field {
              name: "tmp_dir"
              number: 10
              label: LABEL_OPTIONAL
              type: TYPE_STRING
            }
            reserved_range {
              start: 1
              end: 2
            }
            reserved_range {
              start: 2
              end: 3
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(_ExecutionInvocation(tmp_dir='/foo')),
    )

  def test_ShrinksDescriptors_EnumField(self):
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/py/tfx/proto/orchestration/pipeline.proto"
          package: "tfx.orchestration"
          message_type {
            name: "UpdateOptions"
            field {
              name: "reload_policy"
              number: 1
              label: LABEL_OPTIONAL
              type: TYPE_ENUM
              type_name: ".tfx.orchestration.UpdateOptions.ReloadPolicy"
            }
            enum_type {
              name: "ReloadPolicy"
              value {
                name: "ALL"
                number: 0
              }
              value {
                name: "PARTIAL"
                number: 1
              }
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(
            _UpdateOptions(reload_policy=pipeline_pb2.UpdateOptions.PARTIAL)
        ),
    )

  def assertDescriptorContents(
      self,
      fds: descriptor_pb2.FileDescriptorSet,
      expected_types: set[str],
      expected_fields: set[str],
  ) -> None:
    # Instead of asserting the entire descriptor proto, which would be quite
    # verbose, we only check that the right messages and fields were included.
    included_types: set[str] = set()
    included_fields: set[str] = set()

    def _collect_messages(
        name_prefix: str, message_descriptor: descriptor_pb2.DescriptorProto
    ) -> None:
      msg_name = f'{name_prefix}.{message_descriptor.name}'
      included_types.add(msg_name)
      for nested_type in message_descriptor.nested_type:
        _collect_messages(msg_name, nested_type)
      included_types.update(
          {f'{msg_name}.{e.name}' for e in message_descriptor.enum_type}
      )
      for field in message_descriptor.field:
        included_fields.add(f'{msg_name}.{field.name}')

    for fd in fds.file:
      for message_type in fd.message_type:
        _collect_messages(fd.package, message_type)
      included_types.update({f'{fd.package}.{e.name}' for e in fd.enum_type})

    self.assertSameElements(expected_types, included_types)
    self.assertSameElements(expected_fields, included_fields)

  def test_ShrinksDescriptors_ComplexPlaceholder(self):
    fds = validate_and_get_descriptors(
        ph.make_proto(
            execution_invocation_pb2.ExecutionInvocation(
                pipeline_info=pipeline_pb2.PipelineInfo(
                    id='this will be overwritten'
                )
            ),
            pipeline_info=ph.make_proto(
                pipeline_pb2.PipelineInfo(),
                id=ph.execution_invocation().pipeline_run_id,
            ),
            pipeline_node=ph.make_proto(
                pipeline_pb2.PipelineNode(),
                upstream_nodes=[
                    ph.execution_invocation().frontend_url,
                ],
            ),
            execution_properties={
                'fookey': _MetadataStoreValue(
                    proto_value=_UpdateOptions(
                        reload_policy=pipeline_pb2.UpdateOptions.PARTIAL
                    ),
                ),
                'barkey': metadata_store_pb2.Value(int_value=42),
            },
        )
    )

    self.assertDescriptorContents(
        fds,
        {
            # For the Value.proto_value field, which is of type Any:
            'google.protobuf.Any',
            'ml_metadata.Value',
            'tfx.orchestration.ExecutionInvocation',
            # For the ExecutionInvocation.execution_properties map<> field:
            'tfx.orchestration.ExecutionInvocation.ExecutionPropertiesEntry',
            'tfx.orchestration.PipelineInfo',
            'tfx.orchestration.PipelineNode',
            'tfx.orchestration.UpdateOptions',
            'tfx.orchestration.UpdateOptions.ReloadPolicy',
        },
        {
            'google.protobuf.Any.type_url',
            'google.protobuf.Any.value',
            'ml_metadata.Value.int_value',
            'ml_metadata.Value.proto_value',
            'tfx.orchestration.ExecutionInvocation.ExecutionPropertiesEntry.key',
            'tfx.orchestration.ExecutionInvocation.ExecutionPropertiesEntry.value',
            'tfx.orchestration.ExecutionInvocation.execution_properties',
            'tfx.orchestration.ExecutionInvocation.pipeline_info',
            'tfx.orchestration.ExecutionInvocation.pipeline_node',
            'tfx.orchestration.PipelineInfo.id',
            'tfx.orchestration.PipelineNode.upstream_nodes',
            'tfx.orchestration.UpdateOptions.reload_policy',
        },
    )

  def test_ShrinksDescriptors_ListPlaceholderIntoRepeatedField(self):
    fds = validate_and_get_descriptors(
        ph.make_proto(
            pipeline_pb2.StructuralRuntimeParameter(),
            parts=ph.make_list([
                ph.make_proto(
                    pipeline_pb2.StructuralRuntimeParameter.StringOrRuntimeParameter(),
                    constant_value=ph.execution_invocation().pipeline_run_id,
                ),
            ]),
        )
    )

    self.assertDescriptorContents(
        fds,
        {
            'tfx.orchestration.StructuralRuntimeParameter',
            'tfx.orchestration.StructuralRuntimeParameter.StringOrRuntimeParameter',
        },
        {
            'tfx.orchestration.StructuralRuntimeParameter.parts',
            'tfx.orchestration.StructuralRuntimeParameter.StringOrRuntimeParameter.constant_value',
        },
    )

  def test_ShrinksDescriptors_EmptySubmessage(self):
    # It's important that the PipelineNode message is present, even with no
    # fields inside.
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/py/tfx/proto/orchestration/pipeline.proto"
          package: "tfx.orchestration"
          message_type {
            name: "PipelineNode"
          }
          syntax: "proto3"
        }
        file {
          name: "third_party/py/tfx/proto/orchestration/execution_invocation.proto"
          package: "tfx.orchestration"
          dependency: "third_party/py/tfx/proto/orchestration/pipeline.proto"
          message_type {
            name: "ExecutionInvocation"
            field {
              name: "pipeline_node"
              number: 9
              label: LABEL_OPTIONAL
              type: TYPE_MESSAGE
              type_name: ".tfx.orchestration.PipelineNode"
            }
            reserved_range {
              start: 1
              end: 2
            }
            reserved_range {
              start: 2
              end: 3
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(
            _ExecutionInvocation(
                pipeline_node=ph.make_proto(pipeline_pb2.PipelineNode())
            )
        ),
    )

  def test_ShrinksDescriptors_EmptyAnyMessage(self):
    actual = validate_and_get_descriptors(
        _MetadataStoreValue(proto_value=empty_pb2.Empty())
    )

    # For the empty.proto descriptor, we clear the package and proto syntax
    # version, because it's different in different environments and we don't
    # want to assert it below.
    self.assertNotEmpty(actual.file)
    self.assertEndsWith(actual.file[0].name, 'empty.proto')
    actual.file[0].ClearField('package')
    actual.file[0].ClearField('syntax')

    self.assertDescriptorsEqual(
        """
        file {
          name: "google/protobuf/empty.proto"
          message_type {
            name: "Empty"
          }
        }
        file {
          name: "google/protobuf/any.proto"
          package: "google.protobuf"
          message_type {
            name: "Any"
            field {
              name: "type_url"
              number: 1
              label: LABEL_OPTIONAL
              type: TYPE_STRING
            }
            field {
              name: "value"
              number: 2
              label: LABEL_OPTIONAL
              type: TYPE_BYTES
            }
          }
          syntax: "proto3"
        }
        file {
          name: "third_party/ml_metadata/proto/metadata_store.proto"
          package: "ml_metadata"
          dependency: "google/protobuf/any.proto"
          message_type {
            name: "Value"
            field {
              name: "proto_value"
              number: 5
              label: LABEL_OPTIONAL
              type: TYPE_MESSAGE
              type_name: ".google.protobuf.Any"
              oneof_index: 0
            }
            oneof_decl {
              name: "value"
            }
          }
        }
        """,
        actual,
    )

  def test_ShrinksDescriptors_NestedMessage(self):
    # The declaration of PipelineOrNode is nested inside the Pipeline proto.
    # In that case, we must not drop the outer Pipeline proto, as that would
    # also drop the nested proto.
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/py/tfx/proto/orchestration/pipeline.proto"
          package: "tfx.orchestration"
          message_type {
            name: "PipelineNode"
          }
          message_type {
            name: "Pipeline"
            nested_type {
              name: "PipelineOrNode"
              field {
                name: "pipeline_node"
                number: 1
                label: LABEL_OPTIONAL
                type: TYPE_MESSAGE
                type_name: ".tfx.orchestration.PipelineNode"
                oneof_index: 0
              }
              oneof_decl {
                name: "node"
              }
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(
            ph.make_proto(
                pipeline_pb2.Pipeline.PipelineOrNode(),
                pipeline_node=ph.make_proto(pipeline_pb2.PipelineNode()),
            )
        ),
    )

  def test_ShrinksDescriptors_SameFileTwice(self):
    # This contains two separate MakeProtoOperators for UpdateOptions, with a
    # different field. The resulting descriptor should contain both fields.
    # Crucially, there is no file-level dependency from the top-level
    # metadata_store.proto to the inner pipeline.proto, which declares the
    # UpdateOptions. So the _only_ place where the metadata_store.proto and thus
    # UpdateOptions descriptors are coming from are the inner MakeProtoOperator.
    fds = validate_and_get_descriptors(
        ph.make_proto(
            metadata_store_pb2.Artifact(),
            properties={
                'fookey': _MetadataStoreValue(
                    proto_value=_UpdateOptions(
                        reload_policy=pipeline_pb2.UpdateOptions.PARTIAL
                    ),
                ),
                'barkey': _MetadataStoreValue(
                    proto_value=_UpdateOptions(
                        reload_nodes=['a', 'b'],
                    ),
                ),
            },
        )
    )

    self.assertDescriptorContents(
        fds,
        {
            # For the Value.proto_value field, which is of type Any:
            'google.protobuf.Any',
            'ml_metadata.Artifact',
            # For the Artifact.properties map<> field:
            'ml_metadata.Artifact.PropertiesEntry',
            'ml_metadata.Value',
            'tfx.orchestration.UpdateOptions',
            'tfx.orchestration.UpdateOptions.ReloadPolicy',
        },
        {
            'google.protobuf.Any.type_url',
            'google.protobuf.Any.value',
            'ml_metadata.Artifact.properties',
            'ml_metadata.Artifact.PropertiesEntry.key',
            'ml_metadata.Artifact.PropertiesEntry.value',
            'ml_metadata.Value.proto_value',
            'tfx.orchestration.UpdateOptions.reload_policy',
            'tfx.orchestration.UpdateOptions.reload_nodes',
        },
    )

  def test_ShrinksDescriptors_Proto3OptionalFieldPopulated(self):
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/py/tfx/proto/orchestration/pipeline.proto"
          package: "tfx.orchestration"
          message_type {
            name: "NodeExecutionOptions"
            field {
              name: "max_execution_retries"
              number: 6
              label: LABEL_OPTIONAL
              type: TYPE_UINT32
              oneof_index: 0
              proto3_optional: true
            }
            oneof_decl {
              name: "_max_execution_retries"
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(
            ph.make_proto(
                pipeline_pb2.NodeExecutionOptions(),
                max_execution_retries=42,
            )
        ),
    )

  def test_ShrinksDescriptors_Proto3OptionalFieldUnpopulated(self):
    self.assertDescriptorsEqual(
        """
        file {
          name: "third_party/py/tfx/proto/orchestration/pipeline.proto"
          package: "tfx.orchestration"
          message_type {
            name: "NodeExecutionOptions"
            field {
              name: "node_success_optional"
              number: 5
              label: LABEL_OPTIONAL
              type: TYPE_BOOL
            }
          }
          syntax: "proto3"
        }
        """,
        validate_and_get_descriptors(
            ph.make_proto(
                pipeline_pb2.NodeExecutionOptions(node_success_optional=True),
            )
        ),
    )
