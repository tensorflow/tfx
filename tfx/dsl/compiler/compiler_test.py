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
"""Tests for tfx.dsl.compiler.compiler."""
import os
import threading
import unittest

from absl import flags
from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.compiler import compiler
from tfx.dsl.compiler.testdata import additional_properties_test_pipeline_async
from tfx.dsl.compiler.testdata import channel_union_pipeline
from tfx.dsl.compiler.testdata import composable_pipeline
from tfx.dsl.compiler.testdata import conditional_pipeline
from tfx.dsl.compiler.testdata import consumer_pipeline
from tfx.dsl.compiler.testdata import consumer_pipeline_different_project
from tfx.dsl.compiler.testdata import dynamic_exec_properties_pipeline
from tfx.dsl.compiler.testdata import foreach_pipeline
from tfx.dsl.compiler.testdata import iris_pipeline_async
from tfx.dsl.compiler.testdata import iris_pipeline_sync
from tfx.dsl.compiler.testdata import non_existent_component_pipeline
from tfx.dsl.compiler.testdata import optional_and_allow_empty_pipeline
from tfx.dsl.compiler.testdata import pipeline_root_placeholder
from tfx.dsl.compiler.testdata import pipeline_with_annotations
from tfx.dsl.compiler.testdata import resolver_function_pipeline
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact
from tfx.types import channel
from tfx.utils import import_utils

from google.protobuf import text_format

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    "persist_test_protos", False, "Use for regenerating test data. With "
    "test_strategy=local, proto pbtxt files are persisted to "
    "/tmp/<test_name>.pbtxt")


def _maybe_persist_pipeline_proto(pipeline_proto: pipeline_pb2.Pipeline,
                                  to_path: str) -> None:
  if FLAGS.persist_test_protos:
    with open(to_path, mode="w+") as f:
      f.write(text_format.MessageToString(pipeline_proto))


class _MyType(artifact.Artifact):
  TYPE_NAME = "MyTypeName"
  PROPERTIES = {
      "string_value": artifact.Property(artifact.PropertyType.STRING),
  }


class CompilerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # pylint: disable=g-bad-name
    self.maxDiff = 80 * 1000  # Let's hear what assertEqual has to say.

  def _get_test_pipeline_definition(self, module) -> pipeline.Pipeline:
    """Gets the pipeline definition from module."""
    return import_utils.import_func_from_module(module.__name__,
                                                "create_test_pipeline")()

  def _get_test_pipeline_pb(self, file_name: str) -> pipeline_pb2.Pipeline:
    """Reads expected pipeline pb from a text proto file."""
    test_pb_filepath = os.path.join(
        os.path.dirname(__file__), "testdata", file_name)
    with open(test_pb_filepath) as text_pb_file:
      return text_format.ParseLines(text_pb_file, pipeline_pb2.Pipeline())

  @unittest.skipIf(tf.__version__ < "2",
                   "Large proto comparison has a bug not fixed with TF < 2.")
  @parameterized.named_parameters(
      ("_additional_properties_test_pipeline_async",
       additional_properties_test_pipeline_async,
       "additional_properties_test_pipeline_async_ir.pbtxt"),
      ("_sync_pipeline", iris_pipeline_sync, "iris_pipeline_sync_ir.pbtxt"),
      ("_async_pipeline", iris_pipeline_async, "iris_pipeline_async_ir.pbtxt"),
      ("_conditional_pipeline", conditional_pipeline,
       "conditional_pipeline_ir.pbtxt"),
      ("_foreach", foreach_pipeline, "foreach_pipeline_ir.pbtxt"),
      ("_channel_union_pipeline", channel_union_pipeline,
       "channel_union_pipeline_ir.pbtxt"),
      ("_pipeline_root_placeholder", pipeline_root_placeholder,
       "pipeline_root_placeholder_ir.pbtxt"),
      ("_dynamic_exec_properties_pipeline", dynamic_exec_properties_pipeline,
       "dynamic_exec_properties_pipeline_ir.pbtxt"),
      ("_pipeline_with_annotations", pipeline_with_annotations,
       "pipeline_with_annotations_ir.pbtxt"),
      ("_composable_pipeline", composable_pipeline,
       "composable_pipeline_ir.pbtxt"),
      ("_resolver_function_pipeline", resolver_function_pipeline,
       "resolver_function_pipeline_ir.pbtxt"),
      ("_consumber_pipeline", consumer_pipeline, "consumer_pipeline_ir.pbtxt"),
      ("_consumer_pipeline_different_project",
       consumer_pipeline_different_project,
       "consumer_pipeline_different_project_ir.pbtxt"),
      ("_optional_and_allow_empty_pipeline", optional_and_allow_empty_pipeline,
       "optional_and_allow_empty_pipeline_ir.pbtxt"),
  )
  def testCompile(self, pipeline_module, expected_result_path):
    """Tests compiling the whole pipeline."""
    dsl_compiler = compiler.Compiler()
    compiled_pb = dsl_compiler.compile(
        self._get_test_pipeline_definition(pipeline_module))
    expected_pb = self._get_test_pipeline_pb(expected_result_path)
    _maybe_persist_pipeline_proto(compiled_pb, f"/tmp/{expected_result_path}")
    self.assertProtoEquals(expected_pb, compiled_pb)

  def testCompileAdditionalPropertyTypeError(self):
    dsl_compiler = compiler.Compiler()
    test_pipeline = self._get_test_pipeline_definition(
        additional_properties_test_pipeline_async)
    custom_producer = next(
        c for c in test_pipeline.components if isinstance(
            c, additional_properties_test_pipeline_async.CustomProducer))
    custom_producer.outputs["stats"].additional_properties[
        "span"] = "wrong_type"
    with self.assertRaisesRegex(TypeError, "Expected INT but given STRING"):
      dsl_compiler.compile(test_pipeline)

  def testCompileDynamicExecPropTypeError(self):
    dsl_compiler = compiler.Compiler()
    test_pipeline = self._get_test_pipeline_definition(
        dynamic_exec_properties_pipeline)
    downstream_component = next(
        c for c in test_pipeline.components
        if isinstance(c, dynamic_exec_properties_pipeline.DownstreamComponent))
    instance_a = _MyType()
    instance_b = _MyType()
    test_wrong_type_channel = channel.Channel(_MyType).set_artifacts(
        [instance_a, instance_b]).future()
    downstream_component.exec_properties["input_num"] = test_wrong_type_channel
    with self.assertRaisesRegex(
        ValueError,
        "Dynamic execution property only supports ValueArtifact typed channel."
    ):
      dsl_compiler.compile(test_pipeline)

  def testCompileNoneExistentNodeError(self):
    dsl_compiler = compiler.Compiler()
    test_pipeline = self._get_test_pipeline_definition(
        non_existent_component_pipeline)
    with self.assertRaisesRegex(
        ValueError,
        "Node .* references downstream node .* which is not present in the "
        "pipeline"):
      dsl_compiler.compile(test_pipeline)

  def test_DefineAtSub_CompileAtMain(self):
    result_holder = []

    def define_pipeline():
      result_holder.append(conditional_pipeline.create_test_pipeline())

    t = threading.Thread(target=define_pipeline)
    t.start()
    t.join()
    self.assertLen(result_holder, 1)
    p = result_holder[0]
    compiled_pb = compiler.Compiler().compile(p)
    expected_pb = self._get_test_pipeline_pb("conditional_pipeline_ir.pbtxt")
    self.assertProtoEquals(expected_pb, compiled_pb)

  def test_DefineAtSub_CompileAtSub(self):
    result_holder = []

    def define_and_compile_pipeline():
      p = conditional_pipeline.create_test_pipeline()
      result_holder.append(compiler.Compiler().compile(p))

    t = threading.Thread(target=define_and_compile_pipeline)
    t.start()
    t.join()
    self.assertLen(result_holder, 1)
    expected_pb = self._get_test_pipeline_pb("conditional_pipeline_ir.pbtxt")
    self.assertProtoEquals(expected_pb, result_holder[0])


if __name__ == "__main__":
  tf.test.main()
