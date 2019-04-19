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
"""Runners for TFX components running as part of a Kubeflow pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import argparse
import json
import os
import re

from future import utils
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tfx
from tfx.utils import import_utils
from tfx.utils import types


def parse_tfx_type(json_str):
  """Parses a list of artifacts and their types from json."""
  json_artifact_list = json.loads(json_str)

  tfx_types = []
  for json_artifact in json_artifact_list:
    tfx_type = types.TfxType.parse_from_json_dict(json_artifact)
    tfx_types.append(tfx_type)

  return tfx_types


def to_snake_case(name):
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class BaseRunner(utils.with_metaclass(abc.ABCMeta), object):
  """Abstract base class for all Kubeflow Pipelines-based TFX components."""

  def __init__(
      self,
      executor_class_path,
      name,
      input_dict,
      outputs,
      exec_properties,
  ):
    raw_args = exec_properties.get('beam_pipeline_args', [])

    # Beam expects str types for it's pipeline args. Ensure unicode type is
    # converted to str if required.
    beam_pipeline_args = []
    for arg in raw_args:
      if isinstance(arg, unicode):
        arg = arg.encode('ascii', 'ignore')
      beam_pipeline_args.append(arg)

    # TODO(zhitaoli): Revisit usage of setup_file here.
    module_dir = os.path.dirname(
        os.path.dirname(tfx.__file__))
    setup_file = os.path.join(module_dir, 'setup.py')
    beam_pipeline_args.append('--setup_file={}'.format(setup_file))

    executor_cls = import_utils.import_class_by_path(executor_class_path)
    self._executor = executor_cls(beam_pipeline_args=beam_pipeline_args)

    self._input_dict = input_dict
    self._output_dict = types.parse_tfx_type_dict(outputs)
    self._exec_properties = exec_properties
    self._component_name = to_snake_case(name)

  def _set_outputs(self):
    output_dir = self._exec_properties['output_dir']
    workflow_id = os.environ['WORKFLOW_ID']
    tf.logging.info('Using workflow id {}'.format(workflow_id))

    max_input_span = 0
    for input_list in self._input_dict.values():
      for single_input in input_list:
        max_input_span = max(max_input_span, single_input.span)
    for output_name, output_artifact_list in self._output_dict.items():
      for output_artifact in output_artifact_list:
        output_artifact.uri = os.path.join(output_dir, self._component_name,
                                           output_name, workflow_id,
                                           output_artifact.split, '')
        output_artifact.span = max_input_span

  def run(self):
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)

    tf.gfile.MakeDirs('/output/ml_metadata')
    for output_name, output_artifact_list in self._output_dict.items():
      filename = os.path.join('/output/ml_metadata', output_name)
      with file_io.FileIO(filename, 'w') as f:
        output_list = [x.json_dict() for x in output_artifact_list]
        f.write(json.dumps(output_list))


# TODO(ajaygopinathan): Get rid of all the individual runner classes below and
# combine them into a single generic runner that constructs the input dict from
# the individual named arguments instead. In the future, the generic runner can
# call into TFX drivers to handle component-specific logic as well.
class CsvExampleGenRunner(BaseRunner):
  """Runner for CSVExampleGen component."""

  def __init__(self, args):
    super(CsvExampleGenRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='CSVExampleGen',
        input_dict={
            'input-base': parse_tfx_type(args.input_base),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_input_artifact_span()
    self._set_outputs()

  def _set_input_artifact_span(self):
    for input_artifact in self._input_dict['input-base']:
      matched = re.match(r'span_([0-9]+)', input_artifact.uri)
      span = matched.group(1) if matched else 1
      input_artifact.span = span


class BigQueryExampleGenRunner(BaseRunner):
  """Runner for BigQueryExampleGen component."""

  def __init__(self, args):
    super(BigQueryExampleGenRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='BigQueryExampleGen',
        input_dict={},
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()


class StatisticsGenRunner(BaseRunner):
  """Runner for StatisticsGen component."""

  def __init__(self, args):
    super(StatisticsGenRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='StatisticsGen',
        input_dict={
            'input_data': parse_tfx_type(args.input_data),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()


class SchemaGenRunner(BaseRunner):
  """Runner for SchemaGen component."""

  def __init__(self, args):
    super(SchemaGenRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='SchemaGen',
        input_dict={
            'stats': parse_tfx_type(args.stats),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()


class ExampleValidatorRunner(BaseRunner):
  """Runner for ExampleValidator component."""

  def __init__(self, args):
    super(ExampleValidatorRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='ExampleValidator',
        input_dict={
            'stats': parse_tfx_type(args.stats),
            'schema': parse_tfx_type(args.schema),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()


class TransformRunner(BaseRunner):
  """Runner for Transform component."""

  def __init__(self, args):
    super(TransformRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='Transform',
        input_dict={
            'input_data': parse_tfx_type(args.input_data),
            'schema': parse_tfx_type(args.schema),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()


class TrainerRunner(BaseRunner):
  """Runner for Trainer component."""

  def __init__(self, args):
    super(TrainerRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='Trainer',
        input_dict={
            'transformed_examples': parse_tfx_type(args.transformed_examples),
            'transform_output': parse_tfx_type(args.transform_output),
            'schema': parse_tfx_type(args.schema),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()

    # TODO(ajaygopinathan): Implement warm starting.
    self._exec_properties['warm_starting'] = False
    self._exec_properties['warm_start_from'] = None


class EvaluatorRunner(BaseRunner):
  """Runner for Evaluator component."""

  def __init__(self, args):
    super(EvaluatorRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='Evaluator',
        input_dict={
            'examples': parse_tfx_type(args.examples),
            'model_exports': parse_tfx_type(args.model_exports),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()


class ModelValidatorRunner(BaseRunner):
  """Runner for ModelValidator component."""

  def __init__(self, args):
    super(ModelValidatorRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='ModelValidator',
        input_dict={
            'examples': parse_tfx_type(args.examples),
            'model': parse_tfx_type(args.model),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()

    # TODO(ajaygopinathan): Implement latest blessed model determination.
    self._exec_properties['latest_blessed_model'] = None
    self._exec_properties['latest_blessed_model_id'] = None


class PusherRunner(BaseRunner):
  """Runner for Pusher component."""

  def __init__(self, args):
    super(PusherRunner, self).__init__(
        executor_class_path=args.executor_class_path,
        name='Pusher',
        input_dict={
            'model_export': parse_tfx_type(args.model_export),
            'model_blessing': parse_tfx_type(args.model_blessing),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()

    # TODO(ajaygopinathan): Implement latest pushed model
    self._exec_properties['latest_pushed_model'] = None
