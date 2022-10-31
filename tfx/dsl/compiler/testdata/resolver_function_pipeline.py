# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Sample pipeline with a resolver function."""

from typing import Sequence

from tfx.components import CsvExampleGen
from tfx.components import Trainer
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration import pipeline
from tfx.proto import trainer_pb2
import tfx.types
from tfx.utils import typing_utils


class LatestSpans(
    resolver_op.ResolverOp,
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP):
  """Sample resolver operator."""

  n = resolver_op.Property(type=int)

  def _select_latest_spans(self, examples: Sequence[tfx.types.Artifact]):
    examples_by_span = {e.span: e for e in examples}
    latest_spans = sorted(examples_by_span)[-self.n:]
    return [examples_by_span[span] for span in latest_spans]

  def apply(self, inputs: typing_utils.ArtifactMultiMap):
    return {'examples': self._select_latest_spans(inputs['examples'])}


@resolver_function.resolver_function
def resolve_trainer_inputs(inputs, *, n: int):
  return LatestSpans(inputs, n=n)


def create_test_pipeline():
  """Create a test pipeline with resolver function."""

  example_gen = CsvExampleGen(input_base='/data/mydummy_dataset')

  trainer_inputs = resolve_trainer_inputs({
      'examples': example_gen.outputs['examples'],
  }, n=1)

  trainer = Trainer(
      module_file='/src/train.py',
      examples=trainer_inputs['examples'],
      train_args=trainer_pb2.TrainArgs(num_steps=2000),
  )

  return pipeline.Pipeline(
      pipeline_name='resolver-function',
      pipeline_root='/tmp',
      components=[example_gen, trainer],
  )
