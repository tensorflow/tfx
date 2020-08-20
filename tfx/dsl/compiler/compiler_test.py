# Lint as: python2, python3
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

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow_model_analysis as tfma

from google.protobuf import text_format
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.compiler import compiler
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input


class CompilerTest(tf.test.TestCase):

  def setUp(self):
    super(CompilerTest, self).setUp()
    self._set_up_test_pipeline()
    self._set_up_test_pipeline_pb()
    # pylint: disable=g-bad-name
    self.maxDiff = 80 * 1000  # Let's hear what assertEqual has to say.

  def _set_up_test_pipeline(self):
    """Builds an Iris example pipeline with slight changes."""
    pipeline_name = "iris"
    iris_root = "iris_root"
    serving_model_dir = os.path.join(iris_root, "serving_model", pipeline_name)
    tfx_root = "tfx_root"
    data_path = os.path.join(tfx_root, "data_path")
    pipeline_root = os.path.join(tfx_root, "pipelines", pipeline_name)
    self.test_pipeline_info = data_types.PipelineInfo(pipeline_name, iris_root)

    example_gen = CsvExampleGen(input=external_input(data_path))

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True)

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"])

    trainer = Trainer(
        # Use RuntimeParameter as module_file to test out RuntimeParameter in
        # compiler.
        module_file=data_types.RuntimeParameter(
            name="module_file",
            default=os.path.join(iris_root, "iris_utils.py"),
            ptype=str),
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=2000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5))

    model_resolver = ResolverNode(
        instance_name="latest_blessed_model_resolver",
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing))

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(signature_name="eval")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                thresholds={
                    "sparse_categorical_accuracy":
                        tfma.config.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.6}),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10}))
                })
        ])
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config)

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    self._pipeline = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            trainer,
            model_resolver,
            evaluator,
            pusher,
        ],
        enable_cache=True,
        beam_pipeline_args=[])

  def _set_up_test_pipeline_pb(self):
    """Read expected pipeline pb from a text proto file."""
    test_pb_filepath = os.path.join(
        os.path.dirname(__file__), "testdata", "iris_pipeline_ir.pbtxt")
    with open(test_pb_filepath) as text_pb_file:
      self._pipeline_pb = text_format.ParseLines(text_pb_file,
                                                 pipeline_pb2.Pipeline())

  def testCompile(self):
    """Test compiling the whole pipeline."""
    c = compiler.Compiler()
    compiled_pb = c.compile(self._pipeline)
    self.assertProtoEquals(self._pipeline_pb, compiled_pb)


if __name__ == "__main__":
  tf.test.main()
