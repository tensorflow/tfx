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

import apache_beam as beam
import tensorflow as tf
from typing import Generator, List, Optional, Text, Tuple
from tfx.components.base import base_component
from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner

# Key for the signal element using to trigger component execution.
_SIGNAL_ELEMENT = 'SIGNAL_ELEMENT'
# Value of root signal element using as the start point of beam pipeline.
_ROOT = '_ROOT'


class _ComponentAsDoFn(beam.DoFn):
  """Wrap component as beam DoFn."""

  def __init__(self, component: base_component.BaseComponent):
    self._component = component
    self._name = component.component_name

  def process(self, signal_element: Tuple[Text, List[Text]]
             ) -> Generator[Tuple[Text, Text], None, None]:
    """Use single signal element to trigger component execution.

    Args:
      signal_element: a signal tuple with _SIGNAL_ELEMENT as key and list of
        upstream component name as value. For example, pusher will receive
        signal tuple ('SIGNAL_ELEMENT', ['Trainer', 'ModelValidator']).

    Yields:
      A single signal tuple with component name as value.
    """
    tf.logging.info('Component %s received signal %s.', self._name,
                    signal_element)
    self._run_component()
    yield (_SIGNAL_ELEMENT, self._name)

  def _run_component(self) -> None:
    tf.logging.info('Component %s is running.', self._name)
    # TODO(jyzhao): make it real.
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
    with beam.Pipeline(argv=self._beam_orchestrator_args) as p:
      entrypoint_signal = (
          p |
          'CreateEntryPointSignal' >> beam.Create([(_SIGNAL_ELEMENT, [_ROOT])]))

      # Stores component output signals.
      signal_map = {}
      # pipeline.components are in topological order.
      for component in tfx_pipeline.components:
        name = component.component_name

        signal = None
        if component.upstream_nodes:
          # List of upstream output signal, each signal is a PCollection of
          # single signal element, e.g., ('SIGNAL_ELEMENT', 'Trainer').
          upstream_output_signals = []
          for upstream_node in component.upstream_nodes:
            assert upstream_node in signal_map, ('Components is not in '
                                                 'topological order')
            upstream_output_signals.append(signal_map[upstream_node])
          # Combine all upstream signals, result is a PCollection of single
          # signal, e.g., ('SIGNAL_ELEMENT', ['Trainer', 'ModelValidator']).
          signal = (
              upstream_output_signals
              | 'FlattenUpstreamSignals[%s]' % name >> beam.Flatten()
              | 'CombineUpstreamSignals[%s]' % name >> beam.GroupByKey())
        else:
          signal = entrypoint_signal

        # TODO(jyzhao): implement signal as side input.
        # In normal case, component will be executed only once as signal is
        # beam PCollection of single signal element. But DoFn should be
        # re-executable as Beam might schedule the processing of one signal
        # element multiple times, especially in distributed environment.
        signal_map[component] = (
            signal
            | 'Run[%s]' % name >> beam.ParDo(_ComponentAsDoFn(component)))
        tf.logging.info('Component %s is scheduled.', name)
