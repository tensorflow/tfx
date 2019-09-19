# Copyright 2019 Google LLC. All Rights Reserved.
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
"""CIFAR-10 example using TFX DSL on Kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow.runner import KubeflowRunner
from tfx.orchestration.kubeflow.kubeflow_dag_runner import KubeflowDagRunnerConfig
from tfx.proto import evaluator_pb2
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input
from kfp import onprem


_pipeline_name = 'cifar10'
_tfx_root = '/mnt/tfx-pv'
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)
_module_file = os.path.join(_pipeline_root, 'cifar10_utils.py')
_serving_model_dir = os.path.join(_pipeline_root, 'serving_model', _pipeline_name)


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, serving_model_dir: Text) -> pipeline.Pipeline:
    examples = external_input(data_root)
    input_split = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='train.tfrecord'),
        example_gen_pb2.Input.Split(name='eval', pattern='test.tfrecord')
    ])
    example_gen = ImportExampleGen(input_base=examples, input_config=input_split)
    statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)
    infer_schema = SchemaGen(stats=statistics_gen.outputs.output, infer_feature_shape=True)
    validate_stats = ExampleValidator(
        stats=statistics_gen.outputs.output,
        schema=infer_schema.outputs.output)
    transform = Transform(
        input_data=example_gen.outputs.examples,
        schema=infer_schema.outputs.output,
        module_file=module_file)
    trainer = Trainer(
        module_file=module_file,
        examples=transform.outputs.transformed_examples,
        schema=infer_schema.outputs.output,
        transform_output=transform.outputs.transform_output,
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=500))
    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model_exports=trainer.outputs.output,
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
            evaluator_pb2.SingleSlicingSpec()
        ]))
    model_validator = ModelValidator(
        examples=example_gen.outputs.examples, model=trainer.outputs.output)
    pusher = Pusher(
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, infer_schema, validate_stats, transform,
            trainer, model_analyzer, model_validator, pusher
        ],
        log_root='/var/tmp/tfx/logs',
    )


if __name__ == '__main__':
    mount_volume_op = onprem.mount_pvc(
        "tfx-pvc",
        "tfx-pv",
        _tfx_root)
    config = KubeflowDagRunnerConfig(
        pipeline_operator_funcs=[mount_volume_op],
        tfx_image='tensorflow/tfx:0.14.0'
    )
    _pipeline = _create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=os.path.join(_pipeline_root, 'data'),
        module_file=_module_file,
        serving_model_dir=_serving_model_dir,
        )
    KubeflowRunner(config=config).run(_pipeline)
