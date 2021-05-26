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
"""Define KubeflowV2DagRunner to run the pipeline."""

import os
from absl import logging

from tfx.experimental.templates.taxi.pipeline import configs
from tfx.experimental.templates.taxi.pipeline import pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.proto import trainer_pb2

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
# NOTE: It is recommended to have a separated OUTPUT_DIR which is *outside* of
#       the source code structure. Please change OUTPUT_DIR to other location
#       where we can store outputs of the pipeline.
_OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
# - Metadata will be written to metadata service backend.
_PIPELINE_ROOT = os.path.join(_OUTPUT_DIR, 'tfx_pipeline_output',
                              configs.PIPELINE_NAME)

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
_SERVING_MODEL_DIR = os.path.join(_PIPELINE_ROOT, 'serving_model')

# Specifies data file directory. DATA_PATH should be a directory containing CSV
# files for CsvExampleGen in this example. By default, data files are in the
# GCS path: `gs://{GCS_BUCKET_NAME}/tfx-template/data/`. Using a GCS path is
# recommended for KFP.
#
# One can optionally choose to use a data source located inside of the container
# built by the template, by specifying
# _DATA_PATH = 'data'. Note that Dataflow does not support use container as a
# dependency currently, so this means CsvExampleGen cannot be used with Dataflow
# (step 8 in the template notebook).
_DATA_PATH = 'gs://{}/tfx-template/data/taxi/'.format(configs.GCS_BUCKET_NAME)


def run():
  """Define a pipeline to be executed using Kubeflow V2 runner."""

  runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
      default_image=configs.PIPELINE_IMAGE)

  dsl_pipeline = pipeline.create_pipeline(
      pipeline_name=configs.PIPELINE_NAME,
      pipeline_root=_PIPELINE_ROOT,
      data_path=_DATA_PATH,
      # TODO(step 7): (Optional) Uncomment here to use BigQueryExampleGen.
      # query=configs.BIG_QUERY_QUERY,
      preprocessing_fn=configs.PREPROCESSING_FN,
      run_fn=configs.RUN_FN,
      train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
      eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
      eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
      serving_model_dir=_SERVING_MODEL_DIR,
      # TODO(step 7): (Optional) Uncomment here to use provide GCP related
      #               config for BigQuery with Beam DirectRunner.
      # beam_pipeline_args=configs.
      # BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
      # TODO(step 8): (Optional) Uncomment below to use Dataflow.
      # beam_pipeline_args=configs.DATAFLOW_BEAM_PIPELINE_ARGS,
      # TODO(step 9): (Optional) Uncomment below to use Cloud AI Platform.
      # ai_platform_training_args=configs.GCP_AI_PLATFORM_TRAINING_ARGS,
      # TODO(step 9): (Optional) Uncomment below to use Cloud AI Platform.
      # ai_platform_serving_args=configs.GCP_AI_PLATFORM_SERVING_ARGS,
  )

  runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(config=runner_config)

  runner.run(pipeline=dsl_pipeline)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
