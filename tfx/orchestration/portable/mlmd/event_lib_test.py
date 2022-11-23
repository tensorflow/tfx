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
"""Tests for tfx.orchestration.portable.mlmd.event_lib."""
import itertools

from absl.testing import parameterized
import tensorflow as tf
from tfx.orchestration.portable.mlmd import event_lib

from ml_metadata.proto import metadata_store_pb2

Artifact = metadata_store_pb2.Artifact
Event = metadata_store_pb2.Event
Path = metadata_store_pb2.Event.Path
Step = metadata_store_pb2.Event.Path.Step


def valid_path(*args):
  steps = []
  while args:
    key, index, *args = args
    steps.extend([Step(key=key), Step(index=index)])
  return Path(steps=steps)


class EventLibTest(tf.test.TestCase, parameterized.TestCase):

  def testIsValidEvent_Type(self):
    self.assertTrue(
        event_lib.is_valid_input_event(Event(type='INPUT')))
    self.assertTrue(
        event_lib.is_valid_input_event(Event(type='DECLARED_INPUT')))
    self.assertTrue(
        event_lib.is_valid_input_event(Event(type='INTERNAL_INPUT')))
    self.assertFalse(
        event_lib.is_valid_input_event(Event(type='OUTPUT')))
    self.assertFalse(
        event_lib.is_valid_input_event(Event(type='DECLARED_OUTPUT')))
    self.assertFalse(
        event_lib.is_valid_input_event(Event(type='INTERNAL_OUTPUT')))

    self.assertFalse(
        event_lib.is_valid_output_event(Event(type='INPUT')))
    self.assertFalse(
        event_lib.is_valid_output_event(Event(type='DECLARED_INPUT')))
    self.assertFalse(
        event_lib.is_valid_output_event(Event(type='INTERNAL_INPUT')))
    self.assertTrue(
        event_lib.is_valid_output_event(Event(type='OUTPUT')))
    self.assertTrue(
        event_lib.is_valid_output_event(Event(type='DECLARED_OUTPUT')))
    self.assertTrue(
        event_lib.is_valid_output_event(Event(type='INTERNAL_OUTPUT')))

  @parameterized.parameters(
      ('INPUT', event_lib.is_valid_input_event),
      ('INTERNAL_INPUT', event_lib.is_valid_input_event),
      ('DECLARED_INPUT', event_lib.is_valid_input_event),
      ('OUTPUT', event_lib.is_valid_output_event),
      ('INTERNAL_OUTPUT', event_lib.is_valid_output_event),
      ('DECLARED_OUTPUT', event_lib.is_valid_output_event),
  )
  def testIsValidEvent_Path(self, event_type, is_valid_event):
    with self.subTest('No path.'):
      self.assertFalse(
          is_valid_event(Event(type=event_type), 'foo'))

    with self.subTest('Valid path.'):
      self.assertTrue(
          is_valid_event(
              Event(
                  type=event_type,
                  path=Path(
                      steps=[
                          Step(key='foo'),
                          Step(index=0),
                      ])),
              'foo'))

    with self.subTest('Key should match.'):
      self.assertFalse(
          is_valid_event(
              Event(
                  type=event_type,
                  path=Path(
                      steps=[
                          Step(key='foo'),
                          Step(index=0),
                      ])),
              'bar'))

    with self.subTest('Index does not matter.'):
      self.assertTrue(
          is_valid_event(
              Event(
                  type=event_type,
                  path=Path(
                      steps=[
                          Step(key='foo'),
                          Step(index=123),
                      ])),
              'foo'))

    with self.subTest('Multiple steps are ok'):
      self.assertTrue(
          is_valid_event(
              Event(
                  type=event_type,
                  path=Path(
                      steps=[
                          Step(key='asdf'),
                          Step(index=0),
                          Step(key='foo'),
                          Step(index=0),
                      ])),
              'foo'))

    with self.subTest('Invalid path: len(steps) is odd.'):
      self.assertFalse(
          is_valid_event(
              Event(
                  type=event_type,
                  path=Path(
                      steps=[
                          Step(key='asdf'),
                          Step(index=0),
                          Step(key='foo'),
                      ])),
              'foo'))

    with self.subTest('Invalid path: steps[even] should be key.'):
      self.assertFalse(
          is_valid_event(
              Event(
                  type=event_type,
                  path=Path(
                      steps=[
                          Step(index=0),
                          Step(key='foo'),
                      ])),
              'foo'))

    with self.subTest('Invalid path: steps[odd] should be index.'):
      self.assertFalse(
          is_valid_event(
              Event(
                  type=event_type,
                  path=Path(
                      steps=[
                          Step(key='foo'),
                          Step(key='bar'),
                      ])),
              'foo'))

  def testReconstructArtifactMultimap(self):
    a1 = Artifact(id=1, name='a1')
    a2 = Artifact(id=2, name='a2')
    b = Artifact(id=3, name='b')
    c = Artifact(id=4, name='c')

    result = event_lib.reconstruct_artifact_multimap(
        artifacts=[a1, a2, b, c],
        events=[
            Event(artifact_id=a1.id, path=valid_path('a', 0)),
            Event(artifact_id=a2.id, path=valid_path('a', 1)),
            Event(artifact_id=b.id, path=valid_path('b', 0)),
            Event(artifact_id=c.id, path=valid_path('c', 0)),
        ]
    )

    self.assertEqual(result, {
        'a': [a1, a2],
        'b': [b],
        'c': [c],
    })

  def testReconstructArtifactMultimap_DuplicationPermitted(self):
    a = Artifact(id=1, name='a')

    result = event_lib.reconstruct_artifact_multimap(
        artifacts=[a],
        events=[
            Event(
                artifact_id=a.id,
                path=valid_path('a', 0, 'a', 1, 'b', 0, 'c', 0)),
        ]
    )

    self.assertEqual(result, {
        'a': [a, a],
        'b': [a],
        'c': [a],
    })

  def testReconstructArtifactMultimap_EmptyEvents(self):
    result = event_lib.reconstruct_artifact_multimap(artifacts=[], events=[])
    self.assertEmpty(result)

  def testReconstructArtifactMultimap_ArtifactNotIncluded(self):
    a = Artifact(id=1, name='a')
    b = Artifact(id=2, name='a')

    with self.assertRaises(KeyError):
      event_lib.reconstruct_artifact_multimap(
          artifacts=[a],
          events=[
              Event(artifact_id=a.id, path=valid_path('a', 0)),
              Event(artifact_id=b.id, path=valid_path('b', 0)),
          ]
      )

  def testReconstructArtifactMultimap_IndexShouldBeConsecutive(self):
    a1 = Artifact(id=1, name='a1')
    a2 = Artifact(id=2, name='a2')

    with self.subTest('Starts from nonzero'):
      with self.assertRaisesRegex(
          ValueError, 'Index values for key "a" are not consecutive'):
        event_lib.reconstruct_artifact_multimap(
            artifacts=[a1],
            events=[
                Event(artifact_id=a1.id, path=valid_path('a', 1)),
            ])

    with self.subTest('Missing the middle'):
      with self.assertRaisesRegex(
          ValueError, 'Index values for key "a" are not consecutive'):
        event_lib.reconstruct_artifact_multimap(
            artifacts=[a1, a2],
            events=[
                Event(artifact_id=a1.id, path=valid_path('a', 0)),
                Event(artifact_id=a2.id, path=valid_path('a', 2)),
            ])

    with self.subTest('Duplicated index'):
      with self.assertRaisesRegex(
          ValueError, 'Index values for key "a" are not consecutive'):
        event_lib.reconstruct_artifact_multimap(
            artifacts=[a1, a2],
            events=[
                Event(artifact_id=a1.id, path=valid_path('a', 0)),
                Event(artifact_id=a2.id, path=valid_path('a', 0)),
            ])

  def testReconstructArtifactMultimap_HeterogeneousExecutions(self):
    a1 = Artifact(id=1, name='a1')
    a2 = Artifact(id=2, name='a2')

    with self.assertRaisesRegex(
        ValueError, 'All events should be from the same execution'):
      event_lib.reconstruct_artifact_multimap(
          artifacts=[a1, a2],
          events=[
              Event(
                  artifact_id=a1.id,
                  execution_id=123,
                  path=valid_path('a', 0)),
              Event(
                  artifact_id=a2.id,
                  execution_id=321,
                  path=valid_path('a', 1)),
          ])

  def testReconstructArtifactMultimap_HeterogenousArtifactTypes(self):
    a = Artifact(id=1, type_id=1, name='a')
    b = Artifact(id=2, type_id=2, name='b')

    with self.assertRaisesRegex(
        ValueError, 'Artifact type of key "a" is heterogeneous'):
      event_lib.reconstruct_artifact_multimap(
          artifacts=[a, b],
          events=[
              Event(
                  artifact_id=a.id,
                  path=valid_path('a', 0)),
              Event(
                  artifact_id=b.id,
                  path=valid_path('a', 1)),
          ]
      )

  def testReconstructArtifactMultimap_HeterogenousEventTypes(self):
    a1 = Artifact(id=1, name='a1')
    a2 = Artifact(id=2, name='a2')

    with self.assertRaisesRegex(
        ValueError, 'Event type of key "a" is heterogeneous'):
      event_lib.reconstruct_artifact_multimap(
          artifacts=[a1, a2],
          events=[
              Event(
                  artifact_id=a1.id,
                  type='INPUT',
                  path=valid_path('a', 0)),
              Event(
                  artifact_id=a2.id,
                  type='OUTPUT',
                  path=valid_path('a', 1)),
          ]
      )

  @parameterized.parameters(
      *itertools.product(
          ('INPUT', 'INTERNAL_INPUT', 'DECLARED_INPUT'),
          ('OUTPUT', 'INTERNAL_OUTPUT', 'DECLARED_OUTPUT'),
      ))
  def testReconstructInputsAndOutputs(self, input_type, output_type):
    a1 = Artifact(id=1, name='a1')
    a2 = Artifact(id=2, name='a2')
    b = Artifact(id=3, name='b')
    c1 = Artifact(id=4, name='c1')
    c2 = Artifact(id=5, name='c2')
    d = Artifact(id=6, name='d')

    inputs, outputs = event_lib.reconstruct_inputs_and_outputs(
        artifacts=[a1, a2, b, c1, c2, d],
        events=[
            Event(
                type=input_type,
                artifact_id=a1.id,
                path=valid_path('a', 0)),
            Event(
                type=input_type,
                artifact_id=a2.id,
                path=valid_path('a', 1)),
            Event(
                type=input_type,
                artifact_id=b.id,
                path=valid_path('b', 0)),
            Event(
                type=output_type,
                artifact_id=c1.id,
                path=valid_path('c', 0)),
            Event(
                type=output_type,
                artifact_id=c2.id,
                path=valid_path('c', 1)),
            Event(
                type=output_type,
                artifact_id=d.id,
                path=valid_path('d', 0)),
        ])
    self.assertEqual(inputs, {'a': [a1, a2], 'b': [b]})
    self.assertEqual(outputs, {'c': [c1, c2], 'd': [d]})

  def testGenerateEvent(self):
    self.assertProtoEquals(
        """
        type: INPUT
        path {
          steps {
            key: 'key'
          }
          steps {
            index: 1
          }
        }
        artifact_id: 2
        execution_id: 3
        """,
        event_lib.generate_event(
            event_type=metadata_store_pb2.Event.INPUT,
            key='key',
            index=1,
            artifact_id=2,
            execution_id=3))


if __name__ == '__main__':
  tf.test.main()
