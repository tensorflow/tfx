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
"""Penguin example using TFX."""

import os
from typing import Dict, List, Text

from absl import app
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

_pipeline_name = 'penguin_kubeflow_gcp'

# Directory and data locations (uses Google Cloud Storage).
_input_bucket = 'gs://my-bucket'
_output_bucket = 'gs://my-bucket'
_data_root = os.path.join(_input_bucket, 'penguin', 'data')
# Directory and data locations.  This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(_output_bucket, 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

# Google Cloud Platform project id to use when deploying this pipeline.
# This project configuration is for running Dataflow, CAIP Training,
# CAIP Vizier (CloudTuner), and CAIP Prediction services.
_project_id = 'my-gcp-project'

# Python module file to inject customized logic into the TFX components. The
# Transform, Trainer and Tuner all require user-defined functions to run
# successfully. Copy this from the current directory to a GCS bucket and update
# the location below.
_module_file = os.path.join(_input_bucket, 'penguin',
                            'penguin_utils_cloud_tuner.py')

# Region to use for Dataflow jobs and AI Platform jobs.
#   Dataflow: https://cloud.google.com/dataflow/docs/concepts/regional-endpoints
#   AI Platform: https://cloud.google.com/ml-engine/docs/tensorflow/regions
_gcp_region = 'us-central1'

# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
_ai_platform_training_args = {
    'project': _project_id,
    'region': _gcp_region,
    # Starting from TFX 0.14, training on AI Platform uses custom containers:
    # https://cloud.google.com/ml-engine/docs/containers-overview
    # You can specify a custom container here. If not specified, TFX will use a
    # a public container image matching the installed version of TFX.
    # 'masterConfig': { 'imageUri': 'gcr.io/my-project/my-container' },
    # Note that if you do specify a custom container, ensure the entrypoint
    # calls into TFX's run_executor script (tfx/scripts/run_executor.py)
    # Both CloudTuner and the Google Cloud AI Platform extensions Tuner
    # component can be used together, in which case it allows distributed
    # parallel tuning backed by AI Platform Vizier's hyperparameter search
    # algorithm. However, in order to do so, the Cloud AI Platform Job must be
    # given access to the AI Platform Vizier service and Cloud Storage by
    # granting it the ml.admin and storage.objectAdmin roles.
    # https://cloud.google.com/ai-platform/training/docs/custom-service-account#custom
    # Then, you should specify the custom service account for the training job.
    'serviceAccount': '<SA_NAME>@my-gcp-project.iam.gserviceaccount.com',
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google
# Cloud AI Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
_ai_platform_serving_args = {
    'model_name': 'penguin',
    'project_id': _project_id,
    'regions': [_gcp_region]
}

# Beam args to run data processing on DataflowRunner.
#
# TODO(b/151114974): Remove `disk_size_gb` flag after default is increased.
# TODO(b/151116587): Remove `shuffle_mode` flag after default is changed.
# TODO(b/156874687): Remove `machine_type` after IP addresses are no longer a
#                    scaling bottleneck.
# TODO(b/171733562): Remove `use_runner_v2` once it is the default for Dataflow.
_beam_pipeline_args = [
    '--runner=DataflowRunner',
    '--project=' + _project_id,
    '--temp_location=' + os.path.join(_pipeline_root, 'tmp'),
    '--region=' + _gcp_region,

    # Temporary overrides of defaults.
    '--disk_size_gb=50',
    '--experiments=shuffle_mode=auto',
    '--machine_type=e2-standard-8',
    '--experiments=use_runner_v2',
]


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_root: Text,
    module_file: Text,
    ai_platform_training_args: Dict[Text, Text],
    ai_platform_serving_args: Dict[Text, Text],
    enable_tuning: bool,
    beam_pipeline_args: List[Text],
) -> tfx.dsl.Pipeline:
  """Implements the penguin pipeline with TFX and Kubeflow Pipeline.

  Args:
    pipeline_name: name of the TFX pipeline being created.
    pipeline_root: root directory of the pipeline. Should be a valid GCS path.
    data_root: uri of the penguin data.
    module_file: uri of the module files used in Trainer and Transform
      components.
    ai_platform_training_args: Args of CAIP training job. Please refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
      for detailed description.
    ai_platform_serving_args: Args of CAIP model deployment. Please refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
      for detailed description.
    enable_tuning: If True, the hyperparameter tuning through CloudTuner is
      enabled.
    beam_pipeline_args: List of beam pipeline options. Please refer to
      https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options.

  Returns:
    A TFX pipeline object.
  """
  # Number of epochs in training.
  train_steps = tfx.dsl.RuntimeParameter(
      name='train_steps',
      default=100,
      ptype=int,
  )

  # Number of epochs in evaluation.
  eval_steps = tfx.dsl.RuntimeParameter(
      name='eval_steps',
      default=50,
      ptype=int,
  )

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = tfx.components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = tfx.components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Performs transformations and feature engineering in training and serving.
  transform = tfx.components.Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  # Tunes the hyperparameters for model training based on user-provided Python
  # function. Note that once the hyperparameters are tuned, you can drop the
  # Tuner component from pipeline and feed Trainer with tuned hyperparameters.
  if enable_tuning:
    # The Tuner component launches 1 AIP Training job for flock management of
    # parallel tuning. For example, 2 workers (defined by num_parallel_trials)
    # in the flock management AIP Training job, each runs a search loop for
    # trials as shown below.
    #   Tuner component -> CAIP job X -> CloudTunerA -> tuning trials
    #                                 -> CloudTunerB -> tuning trials
    #
    # Distributed training for each trial depends on the Tuner
    # (kerastuner.BaseTuner) setup in tuner_fn. Currently CloudTuner is single
    # worker training per trial. DistributingCloudTuner (a subclass of
    # CloudTuner) launches remote distributed training job per trial.
    #
    # E.g., single worker training per trial
    #   ... -> CloudTunerA -> single worker training
    #       -> CloudTunerB -> single worker training
    # vs distributed training per trial
    #   ... -> DistributingCloudTunerA -> CAIP job Y -> master,worker1,2,3
    #       -> DistributingCloudTunerB -> CAIP job Z -> master,worker1,2,3
    tuner = tfx.components.Tuner(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args={'num_steps': train_steps},
        eval_args={'num_steps': eval_steps},
        tune_args=tfx.proto.TuneArgs(
            # num_parallel_trials=3 means that 3 search loops are
            # running in parallel.
            num_parallel_trials=3),
        custom_config={
            # Note that this TUNING_ARGS_KEY will be used to start the CAIP job
            # for parallel tuning (CAIP job X above).
            #
            # num_parallel_trials will be used to fill/overwrite the
            # workerCount specified by TUNING_ARGS_KEY:
            #   num_parallel_trials = workerCount + 1 (for master)
            tfx.extensions.google_cloud_ai_platform.experimental
            .TUNING_ARGS_KEY:
                ai_platform_training_args,
            # This working directory has to be a valid GCS path and will be used
            # to launch remote training job per trial.
            tfx.extensions.google_cloud_ai_platform.experimental
            .REMOTE_TRIALS_WORKING_DIR_KEY:
                os.path.join(_pipeline_root, 'trials'),
        })

  # Uses user-provided Python function that trains a model.
  trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      # If Tuner is in the pipeline, Trainer can take Tuner's output
      # best_hyperparameters artifact as input and utilize it in the user module
      # code.
      #
      # If there isn't Tuner in the pipeline, either use ImporterNode to import
      # a previous Tuner's output to feed to Trainer, or directly use the tuned
      # hyperparameters in user module code and set hyperparameters to None
      # here.
      #
      # Example of ImporterNode,
      #   hparams_importer = ImporterNode(
      #     source_uri='path/to/best_hyperparameters.txt',
      #     artifact_type=HyperParameters).with_id('import_hparams')
      #   ...
      #   hyperparameters = hparams_importer.outputs['result'],
      hyperparameters=(tuner.outputs['best_hyperparameters']
                       if enable_tuning else None),
      train_args={'num_steps': train_steps},
      eval_args={'num_steps': eval_steps},
      custom_config={
          tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
              ai_platform_training_args
      })

  # Get the latest blessed model for model validation.
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

  # Uses TFMA to compute evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='species')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.6}),
                      # Change threshold will be ignored if there is no
                      # baseline model resolved from MLMD (first run).
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
      ])

  evaluator = tfx.components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      custom_config={
          tfx.extensions.google_cloud_ai_platform.experimental
          .PUSHER_SERVING_ARGS_KEY:
              ai_platform_serving_args
      })

  components = [
      example_gen,
      statistics_gen,
      schema_gen,
      example_validator,
      transform,
      trainer,
      model_resolver,
      evaluator,
      pusher,
  ]
  if enable_tuning:
    components.append(tuner)

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      enable_cache=True,
      beam_pipeline_args=beam_pipeline_args)


def main(unused_argv):
  # Metadata config. The defaults works work with the installation of
  # KF Pipelines using Kubeflow. If installing KF Pipelines using the
  # lightweight deployment option, you may need to override the defaults.
  metadata_config = (tfx.orchestration.experimental
                     .get_default_kubeflow_metadata_config())

  # This pipeline automatically injects the Kubeflow TFX image if the
  # environment variable 'KUBEFLOW_TFX_IMAGE' is defined. The tfx
  # cli tool exports the environment variable to pass to the pipelines.
  tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)

  runner_config = tfx.orchestration.experimental.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config,
      # Specify custom docker image to use.
      tfx_image=tfx_image)

  tfx.orchestration.experimental.KubeflowDagRunner(config=runner_config).run(
      create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          enable_tuning=True,
          ai_platform_training_args=_ai_platform_training_args,
          ai_platform_serving_args=_ai_platform_serving_args,
          beam_pipeline_args=_beam_pipeline_args,
      ))


# $ tfx pipeline create \
# --pipeline-path=penguin_pipeline_kubeflow_gcp.py
# --endpoint='my-gcp-endpoint.pipelines.googleusercontent.com'
# See TFX CLI guide for creating TFX pipelines:
# https://github.com/tensorflow/tfx/blob/master/docs/guide/cli.md#create
# For endpoint, see guide on connecting to hosted AI Platform Pipelines:
# https://cloud.google.com/ai-platform/pipelines/docs/connecting-with-sdk
if __name__ == '__main__':
  app.run(main)
