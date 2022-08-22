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

import datetime
import multiprocessing
import os
import socket
import sys
from typing import List, Optional
import absl
from absl import flags

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.dsl.experimental.conditionals import conditional
from tfx.utils import proto_utils


flags.DEFINE_enum(
    'runner', 'DirectRunner', ['DirectRunner', 'FlinkRunner', 'SparkRunner'],
    'The Beam runner to execute Beam-powered components. '
    'For FlinkRunner or SparkRunner, first run setup/setup_beam_on_flink.sh '
    'or setup/setup_beam_on_spark.sh, respectively.')

flags.DEFINE_enum('model_framework', 'keras',
                  ['keras', 'flax_experimental', 'tfdf_experimental'],
                  'The modeling framework.')

# This example assumes that penguin data is stored in ~/penguin/data and the
# utility function is in ~/penguin. Feel free to customize as needed.
_penguin_root = os.path.join(os.environ['HOME'], 'penguin')
_data_root = os.path.join(_penguin_root, 'data')

# Directory and data locations.  This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')

# Pipeline arguments for Beam powered Components.
# LINT.IfChange
try:
  _parallelism = multiprocessing.cpu_count()
except NotImplementedError:
  _parallelism = 1
# LINT.ThenChange(setup/setup_beam_on_flink.sh)

# Common pipeline arguments used by both Flink and Spark runners.
_beam_portable_pipeline_args = [
    # The runner will instruct the original Python process to start Beam Python
    # workers.
    '--environment_type=LOOPBACK',
    # Start Beam Python workers as separate processes as opposed to threads.
    '--experiments=use_loopback_process_worker=True',
    '--sdk_worker_parallelism=%d' % _parallelism,

    # Setting environment_cache_millis to practically infinity enables
    # continual reuse of Beam SDK workers, improving performance.
    '--environment_cache_millis=1000000',

    # TODO(b/183057237): Obviate setting this.
    '--experiments=pre_optimize=all',
]

# Pipeline arguments for Beam powered Components.
# Arguments differ according to runner.
_beam_pipeline_args_by_runner = {
    'DirectRunner': [
        '--direct_running_mode=multi_processing',
        # 0 means auto-detect based on on the number of CPUs available
        # during execution time.
        '--direct_num_workers=0',
    ],
    'SparkRunner': [
        '--runner=SparkRunner',
        '--spark_submit_uber_jar',
        '--spark_rest_url=http://%s:6066' % socket.gethostname(),
    ] + _beam_portable_pipeline_args,
    'FlinkRunner': [
        '--runner=FlinkRunner',
        # LINT.IfChange
        '--flink_version=1.12',
        # LINT.ThenChange(setup/setup_beam_on_flink.sh)
        '--flink_submit_uber_jar',
        '--flink_master=http://localhost:8081',
        '--parallelism=%d' % _parallelism,
    ] + _beam_portable_pipeline_args
}

# Configs for ExampleGen and SpansResolver, e.g.,
#
# This will match the <input_base>/day3/* as ExampleGen's input and generate
# Examples artifact with Span equals to 3.
#   examplegen_input_config = tfx.proto.Input(splits=[
#       tfx.proto.Input.Split(name='input', pattern='day{SPAN}/*'),
#   ])
#   examplegen_range_config = tfx.proto.RangeConfig(
#       static_range=tfx.proto.StaticRange(
#           start_span_number=3, end_span_number=3))
#
# This will get the latest 2 Spans (Examples artifacts) from MLMD for training.
#   resolver_range_config = tfx.proto.RangeConfig(
#       rolling_range=tfx.proto.RollingRange(num_spans=2))
_examplegen_input_config = None
_examplegen_range_config_date = None
_resolver_range_config = None


@tfx.dsl.components.component
def RangeConfigGenerator(input_date: tfx.dsl.components.Parameter[str],
                         range_config: tfx.dsl.components.OutputArtifact[
                             tfx.types.standard_artifacts.String]):
  """Implements the custom component to convert date into span number.

  Args:
    input_date: input date to generate range_config.
    range_config: range_config to ExampleGen.
  """
  start_time = datetime.datetime(2022, 1,
                                 1)  # start time calculate span number from.
  datem = datetime.datetime.strptime(input_date, '%Y%m%d')
  span_number = (datetime.datetime(datem.year, datem.month, datem.day) -
                 start_time).days
  range_config_str = proto_utils.proto_to_json(
      tfx.proto.RangeConfig(
          static_range=tfx.proto.StaticRange(
              start_span_number=span_number, end_span_number=span_number)))
  range_config.value = range_config_str


def create_pipeline(  # pylint: disable=invalid-name
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    module_file: str,
    accuracy_threshold: float,
    serving_model_dir: str,
    metadata_path: str,
    user_provided_schema_path: Optional[str],
    enable_tuning: bool,
    enable_bulk_inferrer: bool,
    enable_example_diff: bool,
    examplegen_input_config: Optional[tfx.proto.Input],
    examplegen_range_config_date: Optional[str],
    resolver_range_config: Optional[tfx.proto.RangeConfig],
    beam_pipeline_args: List[str],
    # TODO(b/191634100): Always enable transform cache.
    enable_transform_input_cache: bool) -> tfx.dsl.Pipeline:
  """Implements the penguin pipeline with TFX.

  Args:
    pipeline_name: name of the TFX pipeline being created.
    pipeline_root: root directory of the pipeline.
    data_root: directory containing the penguin data.
    module_file: path to files used in Trainer and Transform components.
    accuracy_threshold: minimum accuracy to push the model.
    serving_model_dir: filepath to write pipeline SavedModel to.
    metadata_path: path to local pipeline ML Metadata store.
    user_provided_schema_path: path to user provided schema file.
    enable_tuning: If True, the hyperparameter tuning through KerasTuner is
      enabled.
    enable_bulk_inferrer: If True, the generated model will be used for a
      batch inference.
    enable_example_diff: If True, perform the feature skew detection.
    examplegen_input_config: ExampleGen's input_config.
    examplegen_range_config_date: date to generate the range_config to
      ExampleGen.
    resolver_range_config: SpansResolver's range_config. Specify this will
      enable SpansResolver to get a window of ExampleGen's output Spans for
      transform and training.
    beam_pipeline_args: list of beam pipeline options for LocalDAGRunner. Please
      refer to https://beam.apache.org/documentation/runners/direct/.
    enable_transform_input_cache: Indicates whether input cache should be used
      in Transform if available.

  Returns:
    A TFX pipeline object.
  """
  range_config = None
  if examplegen_range_config_date:
    input_config_generator = RangeConfigGenerator(  # pylint: disable=no-value-for-parameter
        input_date=examplegen_range_config_date)
    range_config = input_config_generator.outputs['range_config'].future(
    )[0].value

  example_gen = tfx.components.CsvExampleGen(
      input_base=os.path.join(data_root, 'labelled'),
      input_config=examplegen_input_config,
      range_config=range_config)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])

  if user_provided_schema_path:
    # Import user-provided schema.
    schema_gen = tfx.components.ImportSchemaGen(
        schema_file=user_provided_schema_path)
    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
  else:
    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)

  # Gets multiple Spans for transform and training.
  if resolver_range_config:
    examples_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.SpanRangeStrategy,
        config={
            'range_config': resolver_range_config
        },
        examples=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.Examples,
            producer_component_id=example_gen.id)).with_id('span_resolver')
    examples_resolver.add_upstream_node(example_gen)

  # Performs transformations and feature engineering in training and serving.
  if enable_transform_input_cache:
    transform_cache_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,
        cache=tfx.dsl.Channel(type=tfx.types.standard_artifacts.TransformCache)
    ).with_id('transform_cache_resolver')
    tft_resolved_cache = transform_cache_resolver.outputs['cache']
  else:
    tft_resolved_cache = None

  transform = tfx.components.Transform(
      examples=(examples_resolver.outputs['examples']
                if resolver_range_config else example_gen.outputs['examples']),
      schema=schema_gen.outputs['schema'],
      module_file=module_file,
      analyzer_cache=tft_resolved_cache)

  # Tunes the hyperparameters for model training based on user-provided Python
  # function. Note that once the hyperparameters are tuned, you can drop the
  # Tuner component from pipeline and feed Trainer with tuned hyperparameters.
  if enable_tuning:
    tuner = tfx.components.Tuner(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=tfx.proto.TrainArgs(num_steps=20),
        eval_args=tfx.proto.EvalArgs(num_steps=5))

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      # If Tuner is in the pipeline, Trainer can take Tuner's output
      # best_hyperparameters artifact as input and utilize it in the user module
      # code.
      #
      # If there isn't Tuner in the pipeline, either use Importer to import
      # a previous Tuner's output to feed to Trainer, or directly use the tuned
      # hyperparameters in user module code and set hyperparameters to None
      # here.
      #
      # Example of Importer,
      #   hparams_importer = Importer(
      #     source_uri='path/to/best_hyperparameters.txt',
      #     artifact_type=HyperParameters).with_id('import_hparams')
      #   ...
      #   hyperparameters = hparams_importer.outputs['result'],
      hyperparameters=(tuner.outputs['best_hyperparameters']
                       if enable_tuning else None),
      train_args=tfx.proto.TrainArgs(num_steps=100),
      eval_args=tfx.proto.EvalArgs(num_steps=5))

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
      model_specs=[
          tfma.ModelSpec(
              signature_name='serving_default',
              label_key='species_xf',
              preprocessing_function_names=['transform_features'])
      ],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': accuracy_threshold}),
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

  # Components declared within the conditional block will only be triggered
  # if the Predicate evaluates to True.
  #
  # In the example below,
  # evaluator.outputs['blessing'].future()[0].custom_property('blessed') == 1
  # is a Predicate, which will be evaluated during runtime.
  #
  # - evaluator.outputs['blessing'] is the output Channel 'blessing'.
  # - .future() turns the Channel into a Placeholder.
  # - [0] gets the first artifact from the 'blessing' Channel.
  # - .custom_property('blessed') gets a custom property called 'blessed' from
  #   that artifact.
  # - == 1 compares that property with 1. (An explicit comparison is needed.
  #   There's no automatic boolean conversion based on truthiness.)
  #
  # Note these operations are just placeholder, something like Mocks. They are
  # not evaluated until runtime. For more details, see tfx/dsl/placeholder/.
  with conditional.Cond(evaluator.outputs['blessing'].future()
                        [0].custom_property('blessed') == 1):
    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        # No need to pass model_blessing any more, since Pusher is already
        # guarded by a Conditional.
        # model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

  # Showcase for BulkInferrer component.
  if enable_bulk_inferrer:
    # Generates unlabelled examples.
    example_gen_unlabelled = tfx.components.CsvExampleGen(
        input_base=os.path.join(data_root, 'unlabelled')).with_id(
            'CsvExampleGen_Unlabelled')

    # Performs offline batch inference.
    bulk_inferrer = tfx.components.BulkInferrer(
        examples=example_gen_unlabelled.outputs['examples'],
        model=trainer.outputs['model'],
        # Empty data_spec.example_splits will result in using all splits.
        data_spec=tfx.proto.DataSpec(),
        model_spec=tfx.proto.ModelSpec())

  if enable_example_diff:
    skewed_data_example_gen = tfx.components.CsvExampleGen(
        input_base=os.path.join(data_root, 'skewed')).with_id(
            'CsvExampleGen_Skewed')
    example_diff_config = tfx.proto.ExampleDiffConfig(
        paired_example_skew=tfx.proto.PairedExampleSkew(
            skew_sample_size=2, identifier_features=['culmen_length_mm']))
    include_split_pairs = [('train', 'train'), ('train', 'eval')]
    example_diff = tfx.components.ExampleDiff(
        examples_test=example_gen.outputs['examples'],
        examples_base=skewed_data_example_gen.outputs['examples'],
        config=example_diff_config,
        include_split_pairs=include_split_pairs
    )

  components_list = [
      example_gen,
      statistics_gen,
      schema_gen,
      transform,
      trainer,
      model_resolver,
      evaluator,
      pusher,
  ]
  if examplegen_range_config_date:
    components_list.append(input_config_generator)
  if resolver_range_config:
    components_list.append(examples_resolver)
  if enable_transform_input_cache:
    components_list.append(transform_cache_resolver)
  if enable_tuning:
    components_list.append(tuner)
  if enable_bulk_inferrer:
    components_list.extend((example_gen_unlabelled, bulk_inferrer))
  if user_provided_schema_path:
    components_list.append(example_validator)
  if enable_example_diff:
    components_list.extend(
        (skewed_data_example_gen, example_diff))

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components_list,
      enable_cache=True,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python penguin_pipeline_local.py [--model_framework=flax_experimental]
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  absl.flags.FLAGS(sys.argv)

  _pipeline_name = f'penguin_local_{flags.FLAGS.model_framework}'

  # Python module file to inject customized logic into the TFX components. The
  # Transform, Trainer and Tuner all require user-defined functions to run
  # successfully.
  _module_file_name = f'penguin_utils_{flags.FLAGS.model_framework}.py'
  _module_file = os.path.join(_penguin_root, _module_file_name)
  # Path which can be listened to by the model server.  Pusher will output the
  # trained model here.
  _serving_model_dir = os.path.join(_penguin_root, 'serving_model',
                                    _pipeline_name)
  # Pipeline root for artifacts.
  _pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
  # Sqlite ML-metadata db path.
  _metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                                'metadata.db')
  tfx.orchestration.LocalDagRunner().run(
      create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          accuracy_threshold=0.6,
          serving_model_dir=_serving_model_dir,
          metadata_path=_metadata_path,
          user_provided_schema_path=None,
          # TODO(b/180723394): support tuning for Flax.
          enable_tuning=(flags.FLAGS.model_framework == 'keras'),
          enable_bulk_inferrer=True,
          examplegen_input_config=_examplegen_input_config,
          examplegen_range_config_date=_examplegen_range_config_date,
          resolver_range_config=_resolver_range_config,
          beam_pipeline_args=_beam_pipeline_args_by_runner[flags.FLAGS.runner],
          enable_transform_input_cache=True,
          enable_example_diff=False))
