# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.compiler.placeholder_utils."""

import base64
import re

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.compiler import placeholder_utils
from tfx.orchestration.portable import data_types
from tfx.proto import infra_validator_pb2
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import execution_invocation_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import proto_utils

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import json_format
from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2

# Concatenate the URI of `examples` input artifact's `train` split with /1
_CONCAT_SPLIT_URI_EXPRESSION = """
operator {
  concat_op {
    expressions {
      operator {
        artifact_uri_op {
          expression {
            operator {
              index_op{
                expression {
                  placeholder {
                    type: INPUT_ARTIFACT
                    key: "examples"
                  }
                }
                index: 0
              }
            }
          }
          split: "train"
        }
      }
    }
    expressions{
      value {
        string_value: "/"
      }
    }
    expressions{
      value {
        int_value: 1
      }
    }
  }
}
"""

_WANT_EXEC_INVOCATION = """
execution_properties {
  key: "proto_property"
  value {
    string_value: "{\\n  \\"tensorflow_serving\\": {\\n    \\"tags\\": [\\n      \\"latest\\",\\n      \\"1.15.0-gpu\\"\\n    ]\\n  }\\n}"
  }
}
execution_properties {
  key: "list_proto_property"
  value {
    string_value: "[\\"{\\\\n  \\\\\\"tensorflow_serving\\\\\\": {\\\\n    \\\\\\"tags\\\\\\": [\\\\n      \\\\\\"latest\\\\\\",\\\\n      \\\\\\"1.15.0-gpu\\\\\\"\\\\n    ]\\\\n  }\\\\n}\\"]"
  }
}
execution_properties_with_schema {
  key: "proto_property"
  value {
    field_value {
      string_value: "{\\n  \\"tensorflow_serving\\": {\\n    \\"tags\\": [\\n      \\"latest\\",\\n      \\"1.15.0-gpu\\"\\n    ]\\n  }\\n}"
    }
  }
}
execution_properties_with_schema {
  key: "list_proto_property"
  value {
    field_value {
      string_value: "[\\"{\\\\n  \\\\\\"tensorflow_serving\\\\\\": {\\\\n    \\\\\\"tags\\\\\\": [\\\\n      \\\\\\"latest\\\\\\",\\\\n      \\\\\\"1.15.0-gpu\\\\\\"\\\\n    ]\\\\n  }\\\\n}\\"]"
    }
    schema {
      value_type {
        list_type {
          proto_type {
            message_type: "tfx.components.infra_validator.ServingSpec"
          }
        }
      }
    }
  }
}
output_metadata_uri: "test_executor_output_uri"
input_dict {
  key: "examples"
  value {
    elements {
      artifact {
        artifact {
          uri: "/tmp"
          properties {
            key: "split_names"
            value {
              string_value: "[\\"train\\", \\"eval\\"]"
            }
          }
          properties {
            key: "version"
            value {
              int_value: 42
            }
          }
          custom_properties {
            key: "custom_key"
            value {
              string_value: "custom_value"
            }
          }
        }
        type {
          name: "Examples"
          properties {
            key: "span"
            value: INT
          }
          properties {
            key: "split_names"
            value: STRING
          }
          properties {
            key: "version"
            value: INT
          }
          base_type: DATASET
        }
      }
    }
  }
}
input_dict {
  key: "model"
  value {
    elements {
      artifact {
        artifact {
        }
        type {
          name: "Model"
          base_type: MODEL
        }
      }
    }
  }
}
output_dict {
  key: "blessing"
  value {
    elements {
      artifact {
        artifact {
        }
        type {
          name: "ModelBlessing"
        }
      }
    }
  }
}
stateful_working_dir: "test_stateful_working_dir"
pipeline_info {
   id: "test_pipeline_id"
}
pipeline_node {
  node_info {
    type {
      name: "infra_validator"
    }
  }
}
"""


class PlaceholderUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    examples = [standard_artifacts.Examples()]
    examples[0].uri = "/tmp"
    examples[0].split_names = artifact_utils.encode_split_names(
        ["train", "eval"])
    examples[0].version = 42
    examples[0].set_string_custom_property("custom_key", "custom_value")
    self._serving_spec = infra_validator_pb2.ServingSpec()
    self._serving_spec.tensorflow_serving.tags.extend(["latest", "1.15.0-gpu"])
    self._resolution_context = placeholder_utils.ResolutionContext(
        exec_info=data_types.ExecutionInfo(
            input_dict={
                "model": [standard_artifacts.Model()],
                "examples": examples,
            },
            output_dict={"blessing": [standard_artifacts.ModelBlessing()]},
            exec_properties={
                "proto_property": proto_utils.proto_to_json(self._serving_spec),
                "list_proto_property": [self._serving_spec],
            },
            execution_output_uri="test_executor_output_uri",
            stateful_working_dir="test_stateful_working_dir",
            pipeline_node=pipeline_pb2.PipelineNode(
                node_info=pipeline_pb2.NodeInfo(
                    type=metadata_store_pb2.ExecutionType(
                        name="infra_validator"))),
            pipeline_info=pipeline_pb2.PipelineInfo(id="test_pipeline_id")),
        executor_spec=executable_spec_pb2.PythonClassExecutableSpec(
            class_path="test_class_path"),
    )
    # Resolution context to simulate missing optional values.
    self._none_resolution_context = placeholder_utils.ResolutionContext(
        exec_info=data_types.ExecutionInfo(
            input_dict={},
            output_dict={},
            exec_properties={},
            pipeline_node=pipeline_pb2.PipelineNode(
                node_info=pipeline_pb2.NodeInfo(
                    type=metadata_store_pb2.ExecutionType(
                        name="infra_validator"))),
            pipeline_info=pipeline_pb2.PipelineInfo(id="test_pipeline_id")),
        executor_spec=None,
        platform_config=None)

  def testConcatArtifactUri(self):
    pb = text_format.Parse(_CONCAT_SPLIT_URI_EXPRESSION,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), "/tmp/Split-train/1")

  def testArtifactProperty(self):
    placeholder_expression = """
      operator {
        artifact_property_op {
          expression {
            operator {
              index_op{
                expression {
                  placeholder {
                    type: INPUT_ARTIFACT
                    key: "examples"
                  }
                }
                index: 0
              }
            }
          }
          key: "version"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), 42)

    self.assertEqual(
        placeholder_utils.debug_str(pb),
        "input(\"examples\")[0].property(\"version\")")

  def testArtifactCustomProperty(self):
    placeholder_expression = """
      operator {
        artifact_property_op {
          expression {
            operator {
              index_op{
                expression {
                  placeholder {
                    type: INPUT_ARTIFACT
                    key: "examples"
                  }
                }
                index: 0
              }
            }
          }
          key: "custom_key"
          is_custom_property: True
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), "custom_value")

    self.assertEqual(
        placeholder_utils.debug_str(pb),
        "input(\"examples\")[0].custom_property(\"custom_key\")")

  def testArtifactUriNoneAccess(self):
    # Access a missing optional channel.
    placeholder_expression = """
      operator {
        artifact_uri_op {
          expression {
            operator {
              index_op{
                expression {
                  placeholder {
                    type: INPUT_ARTIFACT
                    key: "examples"
                  }
                }
                index: 0
              }
            }
          }
          split: "train"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    self.assertIsNone(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._none_resolution_context))

  def testArtifactValueOperator(self):
    test_artifact = standard_artifacts.Integer()
    test_artifact.uri = self.create_tempfile().full_path
    test_artifact.value = 42
    self._resolution_context = placeholder_utils.ResolutionContext(
        exec_info=data_types.ExecutionInfo(
            input_dict={
                "channel_1": [test_artifact],
            },
            pipeline_node=pipeline_pb2.PipelineNode(
                node_info=pipeline_pb2.NodeInfo()),
            pipeline_info=pipeline_pb2.PipelineInfo(id="test_pipeline_id")))
    pb = text_format.Parse(
        """
        operator {
          artifact_value_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: INPUT_ARTIFACT
                      key: "channel_1"
                    }
                  }
                  index: 0
                }
              }
            }
          }
        }
    """, placeholder_pb2.PlaceholderExpression())
    resolved_value = placeholder_utils.resolve_placeholder_expression(
        pb, self._resolution_context)
    self.assertEqual(resolved_value, 42)

  def testJsonValueArtifactWithIndexOperator(self):
    test_artifact = standard_artifacts.JsonValue()
    test_artifact.uri = self.create_tempfile().full_path
    test_artifact.value = {"test_key": [42, 42.0]}
    self._resolution_context = placeholder_utils.ResolutionContext(
        exec_info=data_types.ExecutionInfo(
            input_dict={
                "channel_1": [test_artifact],
            },
            pipeline_node=pipeline_pb2.PipelineNode(
                node_info=pipeline_pb2.NodeInfo()),
            pipeline_info=pipeline_pb2.PipelineInfo(id="test_pipeline_id")))
    pb = text_format.Parse(
        """
        operator {
          index_op {
            expression {
              operator {
                index_op {
                  expression {
                    operator {
                      artifact_value_op {
                        expression {
                          operator {
                            index_op {
                              expression {
                                placeholder {
                                  type: INPUT_ARTIFACT
                                  key: "channel_1"
                                }
                              }
                              index: 0
                            }
                          }
                        }
                      }
                    }
                  }
                  key: "test_key"
                }
              }
            }
            index: 1
          }
        }
    """, placeholder_pb2.PlaceholderExpression())
    resolved_value = placeholder_utils.resolve_placeholder_expression(
        pb, self._resolution_context)
    self.assertEqual(resolved_value, 42.0)

  def testProtoExecPropertyPrimitiveField(self):
    # Access a non-message type proto field
    placeholder_expression = """
      operator {
        index_op {
          expression {
            operator {
              proto_op {
                expression {
                  placeholder {
                    type: EXEC_PROPERTY
                    key: "proto_property"
                  }
                }
                proto_schema {
                  message_type: "tfx.components.infra_validator.ServingSpec"
                }
                proto_field_path: ".tensorflow_serving"
                proto_field_path: ".tags"
              }
            }
          }
          index: 1
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.index_op.expression.operator.proto_op.proto_schema.file_descriptors.file.append(
        fd)

    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), "1.15.0-gpu")

  def testListProtoExecPropertyIndex(self):
    placeholder_expression = """
      operator {
          proto_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: EXEC_PROPERTY
                      key: "list_proto_property"
                    }
                  }
                  index: 0
                }
              }
            }
            serialization_format: JSON
          }
        }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    expected_json_serialization = """\
{
  "tensorflow_serving": {
    "tags": [
      "latest",
      "1.15.0-gpu"
    ]
  }
}"""

    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), expected_json_serialization)

  def testListExecPropertySerializationJson(self):
    placeholder_expression = """
      operator {
        list_serialization_op {
          expression {
            operator {
              proto_op {
                expression {
                  placeholder {
                    type: EXEC_PROPERTY
                    key: "proto_property"
                  }
                }
                proto_schema {
                  message_type: "tfx.components.infra_validator.ServingSpec"
                }
                proto_field_path: ".tensorflow_serving"
                proto_field_path: ".tags"
              }
            }
          }
          serialization_format: JSON
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    expected_json_serialization = '["latest", "1.15.0-gpu"]'
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), expected_json_serialization)

  def testListExecPropertySerializationCommaSeparatedString(self):
    placeholder_expression = """
      operator {
        list_serialization_op {
          expression {
            operator {
              proto_op {
                expression {
                  placeholder {
                    type: EXEC_PROPERTY
                    key: "proto_property"
                  }
                }
                proto_schema {
                  message_type: "tfx.components.infra_validator.ServingSpec"
                }
                proto_field_path: ".tensorflow_serving"
                proto_field_path: ".tags"
              }
            }
          }
          serialization_format: COMMA_SEPARATED_STR
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    expected_serialization = '"latest","1.15.0-gpu"'
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), expected_serialization)

  def testListConcat(self):
    placeholder_expression = """
      operator {
        list_concat_op {
          expressions {
            operator {
              artifact_property_op {
                expression {
                  operator {
                    index_op{
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "examples"
                        }
                      }
                      index: 0
                    }
                  }
                }
                key: "version"
              }
            }
          }
          expressions {
            value {
              string_value: "random_str"
            }
          }
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    expected_result = [42, "random_str"]
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), expected_result)

  def testListConcatAndSerialize(self):
    placeholder_expression = """
      operator {
        list_serialization_op {
          expression {
            operator {
              list_concat_op {
                expressions {
                  operator {
                    artifact_property_op {
                      expression {
                        operator {
                          index_op{
                            expression {
                              placeholder {
                                type: INPUT_ARTIFACT
                                key: "examples"
                              }
                            }
                            index: 0
                          }
                        }
                      }
                      key: "version"
                    }
                  }
                }
                expressions {
                  value {
                    string_value: "random_str"
                  }
                }
              }
            }
          }
          serialization_format: JSON
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    expected_result = '[42, "random_str"]'
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), expected_result)

  def testProtoExecPropertyMessageFieldTextFormat(self):
    # Access a message type proto field
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: EXEC_PROPERTY
              key: "proto_property"
            }
          }
          proto_schema {
            message_type: "tfx.components.infra_validator.ServingSpec"
          }
          proto_field_path: ".tensorflow_serving"
          serialization_format: TEXT_FORMAT
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    # If proto_field_path points to a message type field, the message will
    # be rendered using text_format.
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context),
        "tags: \"latest\"\ntags: \"1.15.0-gpu\"\n")

  def testProtoExecPropertyRepeatedField(self):
    # Access a repeated field.
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: EXEC_PROPERTY
              key: "proto_property"
            }
          }
          proto_schema {
            message_type: "tfx.components.infra_validator.ServingSpec"
          }
          proto_field_path: ".tensorflow_serving"
          proto_field_path: ".tags"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), ["latest", "1.15.0-gpu"])

  def testProtoExecPropertyInvalidField(self):
    # Access a repeated field.
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: EXEC_PROPERTY
              key: "proto_property"
            }
          }
          proto_schema {
            message_type: "tfx.components.infra_validator.ServingSpec"
          }
          proto_field_path: ".some_invalid_field"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    with self.assertRaises(ValueError):
      placeholder_utils.resolve_placeholder_expression(pb,
                                                       self._resolution_context)

  def testProtoExecPropertyNoneAccess(self):
    # Access a missing optional exec property.
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: EXEC_PROPERTY
              key: "proto_property"
            }
          }
          proto_schema {
            message_type: "tfx.components.infra_validator.ServingSpec"
          }
          proto_field_path: ".tensorflow_serving"
          proto_field_path: ".tags"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    self.assertIsNone(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._none_resolution_context))

  def testSerializeDoubleValue(self):
    # Read a primitive value
    placeholder_expression = """
      value {
        double_value: 1.000000009
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), 1.000000009)

  def testProtoRuntimeInfoPlaceholderMessageField(self):
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: RUNTIME_INFO
              key: "executor_spec"
            }
          }
          proto_field_path: ".class_path"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), "test_class_path")

  def testProtoRuntimeInfoNoneAccess(self):
    # Access a missing platform config.
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: RUNTIME_INFO
              key: "platform_config"
            }
          }
          proto_schema {
            message_type: "tfx.components.infra_validator.ServingSpec"
          }
          proto_field_path: ".tensorflow_serving"
          proto_field_path: ".tags"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    self.assertIsNone(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._none_resolution_context))

  def testProtoSerializationJSON(self):
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: EXEC_PROPERTY
              key: "proto_property"
            }
          }
          proto_schema {
            message_type: "tfx.components.infra_validator.ServingSpec"
          }
          serialization_format: JSON
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    expected_json_serialization = """\
{
  "tensorflow_serving": {
    "tags": [
      "latest",
      "1.15.0-gpu"
    ]
  }
}"""

    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context), expected_json_serialization)

  def testProtoWithoutSerializationFormat(self):
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: EXEC_PROPERTY
              key: "proto_property"
            }
          }
          proto_schema {
            message_type: "tfx.components.infra_validator.ServingSpec"
          }
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())

    # Prepare FileDescriptorSet
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    pb.operator.proto_op.proto_schema.file_descriptors.file.append(fd)

    with self.assertRaises(ValueError):
      placeholder_utils.resolve_placeholder_expression(pb,
                                                       self._resolution_context)

  def testExecutionInvocationPlaceholderSimple(self):
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: EXEC_INVOCATION
            }
          }
          serialization_format: JSON
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    resolved = placeholder_utils.resolve_placeholder_expression(
        pb, self._resolution_context)
    got_exec_invocation = json_format.Parse(
        resolved, execution_invocation_pb2.ExecutionInvocation())
    want_exec_invocation = text_format.Parse(
        _WANT_EXEC_INVOCATION, execution_invocation_pb2.ExecutionInvocation())
    fd = descriptor_pb2.FileDescriptorProto()
    infra_validator_pb2.ServingSpec().DESCRIPTOR.file.CopyToProto(fd)
    want_exec_invocation.execution_properties_with_schema[
        "list_proto_property"].schema.value_type.list_type.proto_type.file_descriptors.file.append(
            fd)
    self.assertProtoEquals(want_exec_invocation, got_exec_invocation)

  def testExecutionInvocationPlaceholderAccessProtoField(self):
    placeholder_expression = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: EXEC_INVOCATION
            }
          }
          proto_field_path: ".stateful_working_dir"
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    resolved = placeholder_utils.resolve_placeholder_expression(
        pb, self._resolution_context)
    self.assertEqual(resolved, "test_stateful_working_dir")

  def testExecutionInvocationDescriptor(self):
    # Test if ExecutionInvocation proto is in the default descriptor pool
    pool = descriptor_pool.Default()
    message_descriptor = pool.FindMessageTypeByName(
        "tfx.orchestration.ExecutionInvocation")
    self.assertEqual("tfx.orchestration.ExecutionInvocation",
                     message_descriptor.full_name)

  def testBase64EncodeOperator(self):
    placeholder_expression = """
      operator {
        base64_encode_op {
          expression {
            operator {
              index_op {
                expression {
                  operator {
                    proto_op {
                      expression {
                        placeholder {
                          type: EXEC_PROPERTY
                          key: "proto_property"
                        }
                      }
                      proto_schema {
                        message_type: "tfx.components.infra_validator.ServingSpec"
                      }
                      proto_field_path: ".tensorflow_serving"
                      proto_field_path: ".tags"
                    }
                  }
                }
                index: 0
              }
            }
          }
        }
      }
    """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, self._resolution_context),
        base64.urlsafe_b64encode(b"latest").decode("ASCII"))

  def _assert_serialized_proto_b64encode_eq(self, serialize_format, expected):
    placeholder_expression = """
        operator {
          base64_encode_op {
            expression {
              operator {
                proto_op {
                  expression {
                    placeholder {
                      type: EXEC_PROPERTY
                      key: "proto_property"
                    }
                  }
                  proto_schema {
                    message_type: "tfx.components.infra_validator.ServingSpec"
                  }
                  serialization_format: """ + serialize_format + """
                }
              }
            }
          }
        }
      """
    pb = text_format.Parse(placeholder_expression,
                           placeholder_pb2.PlaceholderExpression())
    resolved_base64_str = placeholder_utils.resolve_placeholder_expression(
        pb, self._resolution_context)
    decoded = base64.urlsafe_b64decode(resolved_base64_str).decode()
    self.assertEqual(decoded, expected)

  def testJsonSerializedProtoBase64Encode(self):
    expected_json_str = json_format.MessageToJson(
        message=self._serving_spec,
        sort_keys=True,
        preserving_proto_field_name=True)
    self._assert_serialized_proto_b64encode_eq("JSON", expected_json_str)

  def testTextFormatSerializedProtoBase64Encode(self):
    expected_text_format_str = text_format.MessageToString(self._serving_spec)
    self._assert_serialized_proto_b64encode_eq("TEXT_FORMAT",
                                               expected_text_format_str)

  def testBinarySerializedProtoBase64Encode(self):
    expected_binary_str = self._serving_spec.SerializeToString().decode()
    self._assert_serialized_proto_b64encode_eq("BINARY", expected_binary_str)

  def testDebugPlaceholder(self):
    pb = text_format.Parse(_CONCAT_SPLIT_URI_EXPRESSION,
                           placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.debug_str(pb),
        "(input(\"examples\")[0].split_uri(\"train\") + \"/\" + \"1\")")

    another_pb_str = """
      operator {
        proto_op {
          expression {
            placeholder {
              type: EXEC_PROPERTY
              key: "serving_spec"
            }
          }
          proto_schema {
            message_type: "tfx.components.infra_validator.ServingSpec"
          }
          proto_field_path: ".tensorflow_serving"
          serialization_format: TEXT_FORMAT
        }
      }
    """
    another_pb = text_format.Parse(another_pb_str,
                                   placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.debug_str(another_pb),
        "exec_property(\"serving_spec\").tensorflow_serving.serialize(TEXT_FORMAT)"
    )


class PredicateResolutionTest(parameterized.TestCase, tf.test.TestCase):

  def _createResolutionContext(self, input_values_dict):
    input_dict = {}
    for channel_name, values in input_values_dict.items():
      input_dict[channel_name] = []
      for value in values:
        artifact = standard_artifacts.Integer()
        artifact.uri = self.create_tempfile().full_path
        artifact.value = value
        input_dict[channel_name].append(artifact)

    return placeholder_utils.ResolutionContext(
        exec_info=data_types.ExecutionInfo(
            input_dict=input_dict,
            pipeline_node=pipeline_pb2.PipelineNode(
                node_info=pipeline_pb2.NodeInfo()),
            pipeline_info=pipeline_pb2.PipelineInfo(id="test_pipeline_id")))

  @parameterized.named_parameters(
      {
          "testcase_name": "1==1",
          "input_values_dict": {
              "channel_1": [1],
              "channel_2": [1],
          },
          "comparison_op": placeholder_pb2.ComparisonOperator.Operation.EQUAL,
          "expected_result": True,
      },
      {
          "testcase_name": "1==2",
          "input_values_dict": {
              "channel_1": [1],
              "channel_2": [2],
          },
          "comparison_op": placeholder_pb2.ComparisonOperator.Operation.EQUAL,
          "expected_result": False,
      },
      {
          "testcase_name":
              "1<2",
          "input_values_dict": {
              "channel_1": [1],
              "channel_2": [2],
          },
          "comparison_op":
              placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          "expected_result":
              True,
      },
      {
          "testcase_name":
              "1<1",
          "input_values_dict": {
              "channel_1": [1],
              "channel_2": [1],
          },
          "comparison_op":
              placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          "expected_result":
              False,
      },
      {
          "testcase_name":
              "2<1",
          "input_values_dict": {
              "channel_1": [2],
              "channel_2": [1],
          },
          "comparison_op":
              placeholder_pb2.ComparisonOperator.Operation.LESS_THAN,
          "expected_result":
              False,
      },
      {
          "testcase_name":
              "2>1",
          "input_values_dict": {
              "channel_1": [2],
              "channel_2": [1],
          },
          "comparison_op":
              placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN,
          "expected_result":
              True,
      },
      {
          "testcase_name":
              "1>1",
          "input_values_dict": {
              "channel_1": [1],
              "channel_2": [1],
          },
          "comparison_op":
              placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN,
          "expected_result":
              False,
      },
      {
          "testcase_name":
              "1>2",
          "input_values_dict": {
              "channel_1": [1],
              "channel_2": [2],
          },
          "comparison_op":
              placeholder_pb2.ComparisonOperator.Operation.GREATER_THAN,
          "expected_result":
              False,
      },
  )
  def testComparisonOperator(self, input_values_dict, comparison_op,
                             expected_result):
    resolution_context = self._createResolutionContext(input_values_dict)
    # Similar to:
    #   some_channel.future()[0].value <?> other_channel.future()[0].value
    pb = text_format.Parse(
        """
      operator {
        compare_op {
          lhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "channel_1"
                        }
                      }
                      index: 0
                    }
                  }
                }
              }
            }
          }
          rhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "channel_2"
                        }
                      }
                      index: 0
                    }
                  }
                }
              }
            }
          }
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    pb.operator.compare_op.op = comparison_op
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, resolution_context), expected_result)

  def _createTrueFalsePredsAndResolutionContext(self):
    """Outputs predicate expressions that evaluate to some constant boolean.

    To test the evaluation of AND, OR, NOT expressions, we want to assert
    that the evaluation code has the same truth table as the corresponding
    operators they are implementing.

    This helper method outputs one predicate expression that always evaluates to
    `True` (true_pb), and one predicate expression that always evaluates to
    `False` (false_pb), as well as the resolution context that produces those
    results.

    true_pb is effectively `1 == 1`.
    false_pb is effectively `1 < 1`.

    These expressions are meant to be used as test inputs for logical
    expressions.

    For example, to assert that `not(True) == False`, construct a placeholder
    expression that represents the NOT operator, copy true_pb into the
    NOT operator's sub expression field, then resolve this placeholder
    expression using the code to be tested, and assert that the resolved value
    is equal to `False`.

    Returns:
      A tuple with three items:
      - A Placeholder expression that always evaluates to True using the given
        ResolutionContext,
      - A Placeholder expression that always evaluates to False using the given
        ResolutionContext, and
      - The ResolutionContext for evaluating the expression.
    """

    resolution_context = self._createResolutionContext({"channel_1": [1]})
    # Evaluating true_pb using the above resolution context is equivalent to
    # evaluating `1 == 1`.
    # Always evaluates to True.
    true_pb = text_format.Parse(
        """
      operator {
        compare_op {
          lhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "channel_1"
                        }
                      }
                      index: 0
                    }
                  }
                }
              }
            }
          }
          rhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "channel_1"
                        }
                      }
                      index: 0
                    }
                  }
                }
              }
            }
          }
          op: EQUAL
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    # This assertion is just to re-assure the reader of this test code that
    # true_pb does evaluate to True, as promised.
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            true_pb, resolution_context), True)

    # Evaluating false_pb using the above resolution context is equivalent to
    # evaluating `1 < 1`.
    # Always evaluates to False.
    false_pb = text_format.Parse(
        """
      operator {
        compare_op {
          lhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "channel_1"
                        }
                      }
                      index: 0
                    }
                  }
                }
              }
            }
          }
          rhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "channel_1"
                        }
                      }
                      index: 0
                    }
                  }
                }
              }
            }
          }
          op: LESS_THAN
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    # This assertion is just to re-assure the reader of this test code that
    # false_pb does evaluate to False, as promised.
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            false_pb, resolution_context), False)
    return true_pb, false_pb, resolution_context

  def testNotOperator(self):
    true_pb, false_pb, resolution_context = (
        self._createTrueFalsePredsAndResolutionContext())

    # assert not(True) == False
    not_true_pb = placeholder_pb2.PlaceholderExpression()
    not_true_pb.operator.unary_logical_op.op = (
        placeholder_pb2.UnaryLogicalOperator.Operation.NOT)
    not_true_pb.operator.unary_logical_op.expression.CopyFrom(true_pb)
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            not_true_pb, resolution_context), False)

    # assert not(False) == True
    not_false_pb = placeholder_pb2.PlaceholderExpression()
    not_false_pb.operator.unary_logical_op.op = (
        placeholder_pb2.UnaryLogicalOperator.Operation.NOT)
    not_false_pb.operator.unary_logical_op.expression.CopyFrom(false_pb)
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            not_false_pb, resolution_context), True)

  @parameterized.named_parameters(
      {
          "testcase_name": "true_and_true",
          "lhs_evaluates_to_true": True,
          "rhs_evaluates_to_true": True,
          "op": placeholder_pb2.BinaryLogicalOperator.Operation.AND,
          "expected_result": True,
      },
      {
          "testcase_name": "true_and_false",
          "lhs_evaluates_to_true": True,
          "rhs_evaluates_to_true": False,
          "op": placeholder_pb2.BinaryLogicalOperator.Operation.AND,
          "expected_result": False,
      },
      {
          "testcase_name": "false_and_true",
          "lhs_evaluates_to_true": False,
          "rhs_evaluates_to_true": True,
          "op": placeholder_pb2.BinaryLogicalOperator.Operation.AND,
          "expected_result": False,
      },
      {
          "testcase_name": "false_and_false",
          "lhs_evaluates_to_true": False,
          "rhs_evaluates_to_true": False,
          "op": placeholder_pb2.BinaryLogicalOperator.Operation.AND,
          "expected_result": False,
      },
      {
          "testcase_name": "true_or_true",
          "lhs_evaluates_to_true": True,
          "rhs_evaluates_to_true": True,
          "op": placeholder_pb2.BinaryLogicalOperator.Operation.OR,
          "expected_result": True,
      },
      {
          "testcase_name": "true_or_false",
          "lhs_evaluates_to_true": True,
          "rhs_evaluates_to_true": False,
          "op": placeholder_pb2.BinaryLogicalOperator.Operation.OR,
          "expected_result": True,
      },
      {
          "testcase_name": "false_or_true",
          "lhs_evaluates_to_true": False,
          "rhs_evaluates_to_true": True,
          "op": placeholder_pb2.BinaryLogicalOperator.Operation.OR,
          "expected_result": True,
      },
      {
          "testcase_name": "false_or_false",
          "lhs_evaluates_to_true": False,
          "rhs_evaluates_to_true": False,
          "op": placeholder_pb2.BinaryLogicalOperator.Operation.OR,
          "expected_result": False,
      },
  )
  def testBinaryLogicalOperator(self, lhs_evaluates_to_true,
                                rhs_evaluates_to_true, op, expected_result):
    true_pb, false_pb, resolution_context = (
        self._createTrueFalsePredsAndResolutionContext())

    pb = placeholder_pb2.PlaceholderExpression()
    pb.operator.binary_logical_op.op = op
    pb.operator.binary_logical_op.lhs.CopyFrom(
        true_pb if lhs_evaluates_to_true else false_pb)
    pb.operator.binary_logical_op.rhs.CopyFrom(
        true_pb if rhs_evaluates_to_true else false_pb)
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            pb, resolution_context), expected_result)

  def testNestedExpression(self):
    true_pb, false_pb, resolution_context = (
        self._createTrueFalsePredsAndResolutionContext())

    true_and_false_pb = placeholder_pb2.PlaceholderExpression()
    true_and_false_pb.operator.binary_logical_op.op = (
        placeholder_pb2.BinaryLogicalOperator.Operation.AND)
    true_and_false_pb.operator.binary_logical_op.lhs.CopyFrom(true_pb)
    true_and_false_pb.operator.binary_logical_op.rhs.CopyFrom(false_pb)

    not_false_pb = placeholder_pb2.PlaceholderExpression()
    not_false_pb.operator.unary_logical_op.op = (
        placeholder_pb2.UnaryLogicalOperator.Operation.NOT)
    not_false_pb.operator.unary_logical_op.expression.CopyFrom(false_pb)

    # assert (True and False) and not(False) == False
    nested_pb_1 = placeholder_pb2.PlaceholderExpression()
    nested_pb_1.operator.binary_logical_op.op = (
        placeholder_pb2.BinaryLogicalOperator.Operation.AND)
    nested_pb_1.operator.binary_logical_op.lhs.CopyFrom(true_and_false_pb)
    nested_pb_1.operator.binary_logical_op.rhs.CopyFrom(not_false_pb)
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            nested_pb_1, resolution_context), False)

    # assert (True and False) or not(False) == True
    nested_pb_2 = placeholder_pb2.PlaceholderExpression()
    nested_pb_2.operator.binary_logical_op.op = (
        placeholder_pb2.BinaryLogicalOperator.Operation.OR)
    nested_pb_2.operator.binary_logical_op.lhs.CopyFrom(true_and_false_pb)
    nested_pb_2.operator.binary_logical_op.rhs.CopyFrom(not_false_pb)
    self.assertEqual(
        placeholder_utils.resolve_placeholder_expression(
            nested_pb_2, resolution_context), True)

  def testDebugPlaceholder(self):
    pb = text_format.Parse(
        """
      operator {
        compare_op {
          lhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "channel_1"
                        }
                      }
                      index: 0
                    }
                  }
                }
              }
            }
          }
          rhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "channel_2"
                        }
                      }
                      index: 0
                    }
                  }
                }
              }
            }
          }
          op: EQUAL
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    self.assertEqual(
        placeholder_utils.debug_str(pb),
        "(input(\"channel_1\")[0].value == input(\"channel_2\")[0].value)")

    another_pb = text_format.Parse(
        """
      operator {
        binary_logical_op {
          lhs {
            operator {
              binary_logical_op {
                lhs {
                  operator {
                    unary_logical_op {
                      expression {
                        operator {
                          compare_op {
                            lhs {
                              operator {
                                artifact_value_op {
                                  expression {
                                    operator {
                                      index_op {
                                        expression {
                                          placeholder {
                                            key: "channel_11_key"
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                            rhs {
                              operator {
                                artifact_value_op {
                                  expression {
                                    operator {
                                      index_op {
                                        expression {
                                          placeholder {
                                            key: "channel_12_key"
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                            op: LESS_THAN
                          }
                        }
                      }
                      op: NOT
                    }
                  }
                }
                rhs {
                  operator {
                    compare_op {
                      lhs {
                        operator {
                          artifact_value_op {
                            expression {
                              operator {
                                index_op {
                                  expression {
                                    placeholder {
                                      key: "channel_21_key"
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      rhs {
                        operator {
                          artifact_value_op {
                            expression {
                              operator {
                                index_op {
                                  expression {
                                    placeholder {
                                      key: "channel_22_key"
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      op: GREATER_THAN
                    }
                  }
                }
                op: AND
              }
            }
          }
          rhs {
            operator {
              unary_logical_op {
                expression {
                  operator {
                    compare_op {
                      lhs {
                        operator {
                          artifact_value_op {
                            expression {
                              operator {
                                index_op {
                                  expression {
                                    placeholder {
                                      key: "channel_3_key"
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      rhs {
                        value {
                          string_value: "foo"
                        }
                      }
                      op: EQUAL
                    }
                  }
                }
                op: NOT
              }
            }
          }
          op: OR
        }
      }
    """, placeholder_pb2.PlaceholderExpression())
    actual_debug_str = placeholder_utils.debug_str(another_pb)
    expected_debug_str_pretty = """
        (
          (
            not(
              (
                input("channel_11_key")[0].value
                <
                input("channel_12_key")[0].value
              )
            ) and
            (
              input("channel_21_key")[0].value
              >
              input("channel_22_key")[0].value
            )
          )
          or
          not(
            (
              input("channel_3_key")[0].value == "foo"
            )
          )
        )
        """
    self.assertEqual(
        re.sub(r"\s+", "", actual_debug_str),
        re.sub(r"\s+", "", expected_debug_str_pretty))


if __name__ == "__main__":
  tf.test.main()
