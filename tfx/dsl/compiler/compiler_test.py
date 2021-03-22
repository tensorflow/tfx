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

from absl.testing import parameterized

import tensorflow as tf
from tfx.dsl.compiler import compiler
from tfx.dsl.compiler.testdata import additional_properties_test_pipeline_async
from tfx.dsl.compiler.testdata import iris_pipeline_async
from tfx.dsl.compiler.testdata import iris_pipeline_sync
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import import_utils

from google.protobuf import text_format


class CompilerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(CompilerTest, self).setUp()
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

  @parameterized.named_parameters(
      ("additional_properties_test_pipeline_async",
       additional_properties_test_pipeline_async,
       "additional_properties_test_pipeline_async_ir.pbtxt"),
      ("sync_pipeline", iris_pipeline_sync, "iris_pipeline_sync_ir.pbtxt"),
      ("async_pipeline", iris_pipeline_async, "iris_pipeline_async_ir.pbtxt"))
  def testCompile(self, pipeline_module, expected_result_path):
    """Tests compiling the whole pipeline."""
    dsl_compiler = compiler.Compiler()
    compiled_pb = dsl_compiler.compile(
        self._get_test_pipeline_definition(pipeline_module))
    expected_pb = self._get_test_pipeline_pb(expected_result_path)
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


if __name__ == "__main__":
  tf.test.main()
