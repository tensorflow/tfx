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
"""Tests CreateProtoPlaceholder end to end, including resolution."""

import base64
from typing import Any, Dict, Optional, Type, TypeVar

import tensorflow as tf
from tfx.dsl.compiler import placeholder_utils
from tfx.dsl.placeholder import placeholder as ph
from tfx.dsl.placeholder import proto_placeholder
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import execution_invocation_pb2
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import message
from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2

_ExecutionInvocation = ph.create_proto(
    execution_invocation_pb2.ExecutionInvocation
)
_PipelineInfo = ph.create_proto(pipeline_pb2.PipelineInfo)
_PipelineNode = ph.create_proto(pipeline_pb2.PipelineNode)
_MetadataStoreValue = ph.create_proto(metadata_store_pb2.Value)
_UpdateOptions = ph.create_proto(pipeline_pb2.UpdateOptions)
_StructuralRuntimeParameter = ph.create_proto(
    pipeline_pb2.StructuralRuntimeParameter
)
_StringOrRuntimeParameter = ph.create_proto(
    pipeline_pb2.StructuralRuntimeParameter.StringOrRuntimeParameter
)

_P = TypeVar('_P', bound=message.Message)


def resolve(
    p: ph.Placeholder, exec_properties: Optional[Dict[str, Any]] = None
) -> Any:
  """Resolves the given placeholder."""
  if isinstance(p, proto_placeholder.CreateProtoPlaceholder):
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


def parse_text_proto(
    textproto: str,
    proto_class: Type[_P] = execution_invocation_pb2.ExecutionInvocation,
) -> _P:
  """Parses and returns a textproto."""
  return text_format.Parse(textproto, proto_class())


# This test covers the DSL side (proto_placeholder.py) that runs at pipeline
# construction time _plus_ the resolution side (placeholder_utils.py) that runs
# at pipeline runtime. There are additional DSL-only test cases in
# ./placeholder_test.py and additional resolution-only test cases in
# dsl/compiler/placeholder_utils_test.py
class ProtoPlaceholderTest(tf.test.TestCase):

  def testCreateProtoPlaceholder_Empty(self):
    self.assertEqual('', resolve(_ExecutionInvocation()))

  def testCreateProtoPlaceholder_BaseOnly(self):
    actual = resolve(
        _ExecutionInvocation(
            execution_invocation_pb2.ExecutionInvocation(tmp_dir='/foo')
        )
    )
    self.assertProtoEquals(
        """
        tmp_dir: "/foo"
        """,
        parse_text_proto(actual),
    )

  def testCreateProtoPlaceholder_FieldOnly(self):
    actual = resolve(_ExecutionInvocation(tmp_dir='/foo'))
    self.assertProtoEquals(
        """
        tmp_dir: "/foo"
        """,
        parse_text_proto(actual),
    )

  def testCreateProtoPlaceholder_ScalarFieldTypes(self):
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

  def testCreateProtoPlaceholder_EnumField(self):
    actual = resolve(
        _UpdateOptions(reload_policy=pipeline_pb2.UpdateOptions.PARTIAL)
    )
    self.assertProtoEquals(
        """
        reload_policy: PARTIAL
        """,
        parse_text_proto(actual, pipeline_pb2.UpdateOptions),
    )

  def testCreateProtoPlaceholder_FieldPlaceholder(self):
    actual = resolve(
        _ExecutionInvocation(tmp_dir=ph.execution_invocation().pipeline_run_id)
    )
    self.assertProtoEquals(
        """
        tmp_dir: "test-run-id"
        """,
        parse_text_proto(actual),
    )

  def testCreateProtoPlaceholder_EnumStringPlaceholder(self):
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

  def testCreateProtoPlaceholder_EnumIntPlaceholder(self):
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

  def testCreateProtoPlaceholder_EmptyFieldPlaceholder(self):
    actual = resolve(
        _ExecutionInvocation(tmp_dir=ph.execution_invocation().frontend_url)
    )
    self.assertProtoEquals(
        """
        tmp_dir: ""
        """,
        parse_text_proto(actual),
    )

  def testCreateProtoPlaceholder_NoneIntoOptionalField(self):
    actual = resolve(_ExecutionInvocation(tmp_dir=None))
    self.assertProtoEquals('', parse_text_proto(actual))

  def testCreateProtoPlaceholder_NonePlaceholderIntoOptionalField(self):
    actual = resolve(
        _ExecutionInvocation(tmp_dir=ph.execution_invocation().frontend_url)
    )
    self.assertProtoEquals('', parse_text_proto(actual))

  def testCreateProtoPlaceholder_NoneExecPropIntoOptionalField(self):
    # When an exec prop has type Union[T, None] and the user passes None, it is
    # actually completely absent from the exec_properties dict in
    # ExecutionInvocation.
    actual = resolve(
        _UpdateOptions(reload_policy=ph.exec_property('reload_policy')),
        exec_properties={},  # Intentionally empty.
    )
    self.assertProtoEquals(
        '',
        parse_text_proto(actual, pipeline_pb2.UpdateOptions),
    )

  def testCreateProtoPlaceholder_BareSubmessage(self):
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

  def testCreateProtoPlaceholder_SubmessageCreateProtoPlaceholder(self):
    actual = resolve(
        _ExecutionInvocation(
            pipeline_info=_PipelineInfo(
                id=ph.execution_invocation().pipeline_run_id
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

  def testCreateProtoPlaceholder_SubmessageProtoGetterPlaceholder(self):
    actual = resolve(
        _ExecutionInvocation(
            pipeline_info=ph.execution_invocation().pipeline_info
        )
    )
    self.assertProtoEquals(
        """
        pipeline_info {
          id: "test-pipeline-id"
        }
        """,
        parse_text_proto(actual),
    )

  def testCreateProtoPlaceholder_SubmessageOverwrite(self):
    actual = resolve(
        _ExecutionInvocation(
            execution_invocation_pb2.ExecutionInvocation(
                pipeline_info=pipeline_pb2.PipelineInfo(
                    id='this will be overwritten'
                )
            ),
            pipeline_info=_PipelineInfo(
                id=ph.execution_invocation().pipeline_run_id
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

  def testCreateProtoPlaceholder_NoneIntoSubmessage(self):
    actual = resolve(_ExecutionInvocation(pipeline_info=None))
    self.assertProtoEquals('', parse_text_proto(actual))

  def testCreateProtoPlaceholder_EmptyPlaceholderIntoSubmessage(self):
    actual = resolve(
        _ExecutionInvocation(
            pipeline_node=ph.execution_invocation().pipeline_node
        )
    )
    self.assertProtoEquals(
        """
        pipeline_node {}
        """,
        parse_text_proto(actual),
    )

  def testCreateProtoPlaceholder_RepeatedField(self):
    actual = resolve(
        _ExecutionInvocation(
            execution_invocation_pb2.ExecutionInvocation(
                pipeline_node=pipeline_pb2.PipelineNode(
                    upstream_nodes=['a', 'b']
                )
            ),
            pipeline_node=_PipelineNode(
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                    ph.execution_invocation().pipeline_run_id + '-bar',
                ]
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

  def testCreateProtoPlaceholder_RepeatedFieldSingleItem(self):
    actual = resolve(
        _ExecutionInvocation(
            pipeline_node=_PipelineNode(
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                ]
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

  def testCreateProtoPlaceholder_RepeatedFieldFalsyItem(self):
    actual = resolve(
        _ExecutionInvocation(
            execution_invocation_pb2.ExecutionInvocation(
                pipeline_node=pipeline_pb2.PipelineNode(upstream_nodes=[''])
            ),
            pipeline_node=_PipelineNode(
                upstream_nodes=[
                    ph.execution_invocation().frontend_url,
                ]
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

  def testCreateProtoPlaceholder_NoneIntoRepeatedField(self):
    actual = resolve(_PipelineNode(upstream_nodes=None))
    self.assertProtoEquals('', parse_text_proto(actual))

  def testCreateProtoPlaceholder_EmptyPlaceholderListIntoRepeatedField(self):
    actual = resolve(
        _PipelineNode(
            upstream_nodes=ph.execution_invocation().pipeline_node.upstream_nodes
        )
    )
    self.assertProtoEquals('', parse_text_proto(actual))

  def testCreateProtoPlaceholder_EmptyListPlaceholderIntoRepeatedField(self):
    actual = resolve(_PipelineNode(upstream_nodes=ph.to_list([])))
    self.assertProtoEquals('', parse_text_proto(actual))

  def testCreateProtoPlaceholder_RepeatedSubmessage(self):
    actual = resolve(
        _StructuralRuntimeParameter(
            parts=[
                pipeline_pb2.StructuralRuntimeParameter.StringOrRuntimeParameter(
                    constant_value='foo'
                ),
                _StringOrRuntimeParameter(
                    constant_value=ph.execution_invocation().pipeline_run_id
                ),
            ]
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

  def testCreateProtoPlaceholder_AnySubmessageBareMessage(self):
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

  def testCreateProtoPlaceholder_AnySubmessagePlaceholder(self):
    actual = resolve(
        _MetadataStoreValue(
            # We can directly assign a message of any type and it will pack it.
            proto_value=_PipelineNode(
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                ]
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

  def testCreateProtoPlaceholder_NonePlaceholderIntoAnySubmessage(self):
    actual = resolve(
        _MetadataStoreValue(proto_value=ph.execution_invocation().pipeline_node)
    )
    self.assertProtoEquals(
        """
        proto_value {
          [type.googleapis.com/tfx.orchestration.PipelineNode] {}
        }
        """,
        parse_text_proto(actual, metadata_store_pb2.Value),
    )

  def testCreateProtoPlaceholder_MapFieldScalarValue(self):
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

  def testCreateProtoPlaceholder_MapFieldScalarPlaceholderValue(self):
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

  def testCreateProtoPlaceholder_MapFieldScalarNoneValue(self):
    actual = resolve(
        _ExecutionInvocation(
            extra_flags={
                'fookey': ph.exec_property('reload_policy'),
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

  def testCreateProtoPlaceholder_MapFieldSubmessageValue(self):
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

  def testCreateProtoPlaceholder_MapFieldPlaceholderKey(self):
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

  def testCreateProtoPlaceholder_RejectsMapFieldScalarNoneKey(self):
    with self.assertRaises(ValueError):
      resolve(
          _ExecutionInvocation(
              extra_flags=[(ph.exec_property('reload_policy'), 'foo')]
          ),
          exec_properties={},  # Intentionally empty.
      )
    with self.assertRaises(ValueError):
      resolve(_ExecutionInvocation(extra_flags={None: 'foo'}))

  def testCreateProtoPlaceholder_MapFieldScalarValueEmpty(self):
    actual = resolve(_ExecutionInvocation(extra_flags={}))
    self.assertProtoEquals('', parse_text_proto(actual))
    actual = resolve(_ExecutionInvocation(extra_flags=[]))
    self.assertProtoEquals('', parse_text_proto(actual))

  def testCreateProtoPlaceholder_PlusItemGetter(self):
    actual = resolve(
        _ExecutionInvocation(
            pipeline_node=_PipelineNode(
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                ]
            ),
            # It's a little silly, but it's something that should work:
            # Constructing an entire proto through placeholders, only to
            # read out a single field from it:
        ).pipeline_node.upstream_nodes[0]
    )
    self.assertProtoEquals('test-run-id-foo', actual)

  def test_CreateProtoPlaceholder_BinarySerializationBase64(self):
    actual = resolve(
        _ExecutionInvocation(
            execution_invocation_pb2.ExecutionInvocation(
                tmp_dir='/foo',
                pipeline_node=pipeline_pb2.PipelineNode(
                    upstream_nodes=['a', 'b']
                ),
            ),
            pipeline_node=_PipelineNode(
                upstream_nodes=[
                    ph.execution_invocation().pipeline_run_id + '-foo',
                    ph.execution_invocation().pipeline_run_id + '-bar',
                ]
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


if __name__ == '__main__':
  tf.test.main()
