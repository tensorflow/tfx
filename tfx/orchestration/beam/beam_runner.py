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
"""Definition of Beam TFX runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import apache_beam as beam
import tensorflow as tf
from typing import Any, Dict, Iterable, List, Optional, Text

from tfx.components.base import base_component
from tfx.orchestration import component_runner
from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner


# TODO(jyzhao): confirm it's re-executable, add test case.
@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(Any)
class _ComponentAsDoFn(beam.DoFn):
  """Wrap component as beam DoFn."""

  def __init__(self, component: base_component.BaseComponent,
               pipeline_run_id: Text, pipeline_args: Dict[Text, Any]):
    """Initialize the _ComponentAsDoFn.

    Args:
      component: Component that to be executed.
      pipeline_run_id: The unique id of current pipeline run.
      pipeline_args: args dict that contains the following items,
        - pipeline_name: The unique name of this pipeline.
        - pipeline_root: The root path of the pipeline outputs.
        - metadata_connection_config: ML metadata connection config.
        - enable_cache: Whether to enable cache functionality.
        - additional_pipeline_args: Additional pipeline args, includes,
          - beam_pipeline_args: Beam pipeline args for beam jobs within
            executor. Executor will use beam DirectRunner as Default.
    """
    self._component_runner = component_runner.ComponentRunner(
        component, pipeline_run_id, **pipeline_args)
    self._name = component.component_name

  def process(self, element: Any, *signals: Iterable[Any]) -> None:
    """Executes component based on signals.

    Args:
      element: a signal element to trigger the component.
      *signals: side input signals indicate completeness of upstream components.
    """
    for signal in signals:
      assert not list(signal), 'Signal PCollection should be empty.'
    self._run_component()

  def _run_component(self) -> None:
    tf.logging.info('Component %s is running.', self._name)
    self._component_runner.run()
    tf.logging.info('Component %s is finished.', self._name)


class BeamRunner(tfx_runner.TfxRunner):
  """Tfx runner on Beam."""

  def __init__(self, beam_orchestrator_args: Optional[List[Text]] = None):
    """Initializes BeamRunner as a TFX orchestrator.

    Args:
      beam_orchestrator_args: beam args for the beam orchestrator. Note that
        this is different from the beam_pipeline_args within
        additional_pipeline_args, which is for beam pipelines in components.
    """
    super(BeamRunner, self).__init__()
    self._beam_orchestrator_args = beam_orchestrator_args

  def run(self, tfx_pipeline: pipeline.Pipeline) -> None:
    """Deploys given logical pipeline on Beam.

    Args:
      tfx_pipeline: Logical pipeline containing pipeline args and components.
    """
    pipeline_run_id = datetime.datetime.now().isoformat()
    # TODO(jyzhao): remove after driver is supported.
    # Setup output uri in a similar way as Kubeflow before we have driver ready.
    for component in tfx_pipeline.components:
      output_dict = dict(
          (k, v.get()) for k, v in component.outputs.get_all().items())
      for output_name, output_list in output_dict.items():
        for output_artifact in output_list:
          # Last empty string forces this be to a directory.
          output_artifact.uri = os.path.join(
              tfx_pipeline.pipeline_args.get('pipeline_root'),
              tfx_pipeline.pipeline_args.get('pipeline_name'),
              component.component_name, output_name, pipeline_run_id,
              output_artifact.split, '')

    with beam.Pipeline(argv=self._beam_orchestrator_args) as p:
      # Uses for triggering the component DoFns.
      root = p | 'CreateRoot' >> beam.Create([None])

      # Stores mapping of component to its signal.
      signal_map = {}
      # pipeline.components are in topological order.
      for component in tfx_pipeline.components:
        name = component.component_name

        # Signals from upstream components.
        signals_to_wait = []
        if component.upstream_nodes:
          for upstream_node in component.upstream_nodes:
            assert upstream_node in signal_map, ('Components is not in '
                                                 'topological order')
            signals_to_wait.append(signal_map[upstream_node])
        tf.logging.info('Component %s depends on %s.', name,
                        [s.producer.full_label for s in signals_to_wait])

        # Each signal is an empty PCollection. AsIter ensures component will be
        # triggered after upstream components are finished.
        signal_map[component] = (
            root
            | 'Run[%s]' % name >> beam.ParDo(
                _ComponentAsDoFn(component, pipeline_run_id,
                                 tfx_pipeline.pipeline_args),
                *[beam.pvalue.AsIter(s) for s in signals_to_wait]))
        tf.logging.info('Component %s is scheduled.', name)
