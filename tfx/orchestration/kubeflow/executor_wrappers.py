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
"""Wrappers for TFX executors running as part of a Kubeflow pipeline."""
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
from typing import Any, Dict, Text

import tfx
from tfx.components.base import base_executor
from tfx.utils import import_utils
from tfx.utils import types


def parse_tfx_type(json_str: Text):
  """Parses a list of artifacts and their types from json."""
  json_artifact_list = json.loads(json_str)

  tfx_types = []
  for json_artifact in json_artifact_list:
    tfx_type = types.TfxArtifact.parse_from_json_dict(json_artifact)
    tfx_types.append(tfx_type)

  return tfx_types


def to_snake_case(name: Text):
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class KubeflowExecutorWrapper(utils.with_metaclass(abc.ABCMeta), object):
  """Abstract base class for all Kubeflow Pipelines-based TFX components."""

  def __init__(
      self,
      executor_class_path: Text,
      name: Text,
      input_dict: Dict[Text, List[types.TfxArtifact]],
      outputs: Text,
      exec_properties: Dict[Text, Any],
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
    self._exec_properties = exec_properties
    self._output_dir = self._exec_properties['output_dir']
    self._workflow_id = os.environ['WORKFLOW_ID']
    # TODO(swoonna): Switch to execution_id when available
    unique_id = '{}_{}'.format(self._component_name, self._workflow_id)
    # TODO(swoonna): Add tmp_dir to additional_pipeline_args
    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=beam_pipeline_args,
        tmp_dir=os.path.join(self._output_dir, '.temp', ''),
        unique_id=unique_id)

    self._executor = executor_cls(executor_context)
    self._input_dict = input_dict
    self._output_dict = types.parse_tfx_type_dict(outputs)
    self._component_name = to_snake_case(name)

  def _set_outputs(self):
    tf.logging.info('Using workflow id {}'.format(self._workflow_id))

    max_input_span = 0
    for input_list in self._input_dict.values():
      for single_input in input_list:
        max_input_span = max(max_input_span, single_input.span)
    for output_name, output_artifact_list in self._output_dict.items():
      for output_artifact in output_artifact_list:
        output_artifact.uri = os.path.join(self._output_dir,
                                           self._component_name, output_name,
                                           self._workflow_id,
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


# TODO(b/132197968): Get rid of all the individual wrapper classes below and
# combine them into a single generic one that constructs the input dict from
# the individual named arguments instead. In the future, the generic wrapper
# can call into TFX drivers to handle component-specific logic as well.
class CsvExampleGenWrapper(KubeflowExecutorWrapper):
  """Wrapper for CSVExampleGen component."""

  def __init__(self, args: argparse.Namespace):
    super(CsvExampleGenWrapper, self).__init__(
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


class BigQueryExampleGenWrapper(KubeflowExecutorWrapper):
  """Wrapper for BigQueryExampleGen component."""

  def __init__(self, args: argparse.Namespace):
    super(BigQueryExampleGenWrapper, self).__init__(
        executor_class_path=args.executor_class_path,
        name='BigQueryExampleGen',
        input_dict={},
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()


class StatisticsGenWrapper(KubeflowExecutorWrapper):
  """Wrapper for StatisticsGen component."""

  def __init__(self, args: argparse.Namespace):
    super(StatisticsGenWrapper, self).__init__(
        executor_class_path=args.executor_class_path,
        name='StatisticsGen',
        input_dict={
            'input_data': parse_tfx_type(args.input_data),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()


class SchemaGenWrapper(KubeflowExecutorWrapper):
  """Wrapper for SchemaGen component."""

  def __init__(self, args: argparse.Namespace):
    super(SchemaGenWrapper, self).__init__(
        executor_class_path=args.executor_class_path,
        name='SchemaGen',
        input_dict={
            'stats': parse_tfx_type(args.stats),
        },
        outputs=args.outputs,
        exec_properties=json.loads(args.exec_properties),
    )
    self._set_outputs()


class ExampleValidatorWrapper(KubeflowExecutorWrapper):
  """Wrapper for ExampleValidator component."""

  def __init__(self, args: argparse.Namespace):
    super(ExampleValidatorWrapper, self).__init__(
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


class TransformWrapper(KubeflowExecutorWrapper):
  """Wrapper for Transform component."""

  def __init__(self, args: argparse.Namespace):
    super(TransformWrapper, self).__init__(
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


class TrainerWrapper(KubeflowExecutorWrapper):
  """Wrapper for Trainer component."""

  def __init__(self, args: argparse.Namespace):
    super(TrainerWrapper, self).__init__(
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


class EvaluatorWrapper(KubeflowExecutorWrapper):
  """Wrapper for Evaluator component."""

  def __init__(self, args: argparse.Namespace):
    super(EvaluatorWrapper, self).__init__(
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


class ModelValidatorWrapper(KubeflowExecutorWrapper):
  """Wrapper for ModelValidator component."""

  def __init__(self, args: argparse.Namespace):
    super(ModelValidatorWrapper, self).__init__(
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


class PusherWrapper(KubeflowExecutorWrapper):
  """Wrapper for Pusher component."""

  def __init__(self, args: argparse.Namespace):
    super(PusherWrapper, self).__init__(
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
