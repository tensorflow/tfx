# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import json
import os

from absl.testing import parameterized
from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
from tfx.dsl.io import fileio
from tfx.orchestration.kubeflow.v2 import compiler_utils
from tfx.orchestration.kubeflow.v2.file_based_example_gen import driver
from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils import test_case_utils

from google.protobuf import json_format

_TEST_OUTPUT_METADATA_JSON = 'output/outputmetadata.json'
_TEST_INPUT_DIR = 'input_base'


def _build_executor_invocation(
    use_legacy: bool = False, with_span: bool = False
):
  executor_invocation = pipeline_pb2.ExecutorInput()
  executor_invocation.outputs.output_file = _TEST_OUTPUT_METADATA_JSON
  input_with_span = example_gen_pb2.Input(
      splits=[
          example_gen_pb2.Input.Split(name='s1', pattern='span{SPAN}/split1/*'),
          example_gen_pb2.Input.Split(name='s2', pattern='span{SPAN}/split2/*'),
      ]
  )
  input_without_span = example_gen_pb2.Input(
      splits=[
          example_gen_pb2.Input.Split(name='s1', pattern='split1/*'),
          example_gen_pb2.Input.Split(name='s2', pattern='split2/*'),
      ]
  )
  if with_span:
    input_config = json_format.MessageToJson(input_with_span)
  else:
    input_config = json_format.MessageToJson(input_without_span)

  if use_legacy:
    executor_invocation.inputs.parameters['input_base'].string_value = (
        _TEST_INPUT_DIR
    )
    executor_invocation.inputs.parameters['output_config'].string_value = '{}'
    executor_invocation.inputs.parameters['input_config'].string_value = (
        input_config
    )
  else:
    executor_invocation.inputs.parameter_values['input_base'].string_value = (
        _TEST_INPUT_DIR
    )
    executor_invocation.inputs.parameter_values[
        'output_config'
    ].string_value = '{}'
    executor_invocation.inputs.parameter_values['input_config'].string_value = (
        input_config
    )
  executor_invocation.outputs.artifacts['examples'].artifacts.append(
      pipeline_pb2.RuntimeArtifact(
          type=pipeline_pb2.ArtifactTypeSchema(
              instance_schema=compiler_utils.get_artifact_schema(
                  standard_artifacts.Examples
              )
          )
      )
  )
  return executor_invocation


def _load_test_file(filename: str):
  return fileio.open(
      os.path.join(os.path.dirname(__file__), 'testdata', filename),
      'r',
  ).read()


class RunDriverTest(test_case_utils.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Change working directory after all the testdata files have been read.
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

    fileio.makedirs(os.path.dirname(_TEST_INPUT_DIR))

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testDriverWithoutSpan(self, use_pipeline_spec_2_1):
    split1 = os.path.join(_TEST_INPUT_DIR, 'split1', 'data')
    io_utils.write_string_file(split1, 'testing')
    os.utime(split1, (0, 1))
    split2 = os.path.join(_TEST_INPUT_DIR, 'split2', 'data')
    io_utils.write_string_file(split2, 'testing2')
    os.utime(split2, (0, 3))

    executor_invocation = _build_executor_invocation(
        use_legacy=not use_pipeline_spec_2_1, with_span=False
    )
    serialized_args = [
        '--json_serialized_invocation_args',
        json_format.MessageToJson(message=executor_invocation),
    ]

    if use_pipeline_spec_2_1:
      inputs_spec = pipeline_pb2.ComponentInputsSpec()
      inputs_spec.parameters['input_config'].parameter_type = (
          pipeline_pb2.ParameterType.STRING
      )
      serialized_args.extend([
          '--json_serialized_inputs_spec_args',
          json_format.MessageToJson(message=inputs_spec),
      ])
    # Invoke the driver
    driver.main(driver._parse_flags(serialized_args))

    # Check the output metadata file for the expected outputs
    with fileio.open(_TEST_OUTPUT_METADATA_JSON, 'rb') as output_meta_json:
      output_metadata = pipeline_pb2.ExecutorOutput()
      json_format.Parse(
          output_meta_json.read(), output_metadata, ignore_unknown_fields=True)
      self.assertEqual(output_metadata.parameter_values['span'].number_value, 0)
      self.assertEqual(
          output_metadata.parameter_values['input_fingerprint'].string_value,
          'split:s1,num_files:1,total_bytes:7,xor_checksum:1,sum_checksum:1\n'
          'split:s2,num_files:1,total_bytes:8,xor_checksum:3,sum_checksum:3',
      )
      self.assertEqual(
          output_metadata.parameter_values['input_config'].string_value,
          json_format.MessageToJson(
              example_gen_pb2.Input(
                  splits=[
                      example_gen_pb2.Input.Split(
                          name='s1', pattern='split1/*'
                      ),
                      example_gen_pb2.Input.Split(
                          name='s2', pattern='split2/*'
                      ),
                  ]
              )
          ),
      )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testDriverWithSpan(self, use_pipeline_spec_2_1):
    # Test align of span number.
    span1_split1 = os.path.join(_TEST_INPUT_DIR, 'span1', 'split1', 'data')
    io_utils.write_string_file(span1_split1, 'testing11')
    span1_split2 = os.path.join(_TEST_INPUT_DIR, 'span1', 'split2', 'data')
    io_utils.write_string_file(span1_split2, 'testing12')
    span2_split1 = os.path.join(_TEST_INPUT_DIR, 'span2', 'split1', 'data')
    io_utils.write_string_file(span2_split1, 'testing21')

    executor_invocation = _build_executor_invocation(
        use_legacy=not use_pipeline_spec_2_1, with_span=True
    )
    serialized_args = [
        '--json_serialized_invocation_args',
        json_format.MessageToJson(message=executor_invocation),
    ]

    if use_pipeline_spec_2_1:
      inputs_spec = pipeline_pb2.ComponentInputsSpec()
      inputs_spec.parameters['input_config'].parameter_type = (
          pipeline_pb2.ParameterType.STRING
      )
      serialized_args.extend([
          '--json_serialized_inputs_spec_args',
          json_format.MessageToJson(message=inputs_spec),
      ])
    with self.assertRaisesRegex(
        ValueError, 'Latest span should be the same for each split'):
      driver.main(driver._parse_flags(serialized_args))

    # Test if latest span is selected when span aligns for each split.
    span2_split2 = os.path.join(_TEST_INPUT_DIR, 'span2', 'split2', 'data')
    io_utils.write_string_file(span2_split2, 'testing22')

    driver.main(driver._parse_flags(serialized_args))

    # Check the output metadata file for the expected outputs
    with fileio.open(_TEST_OUTPUT_METADATA_JSON, 'rb') as output_meta_json:
      output_metadata = pipeline_pb2.ExecutorOutput()
      json_format.Parse(
          output_meta_json.read(), output_metadata, ignore_unknown_fields=True)
      self.assertEqual(output_metadata.parameter_values['span'].number_value, 2)
      self.assertEqual(
          output_metadata.parameter_values['input_config'].string_value,
          json_format.MessageToJson(
              example_gen_pb2.Input(
                  splits=[
                      example_gen_pb2.Input.Split(
                          name='s1', pattern='span2/split1/*'
                      ),
                      example_gen_pb2.Input.Split(
                          name='s2', pattern='span2/split2/*'
                      ),
                  ]
              )
          ),
      )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testDriverJsonContract(self, use_pipeline_spec_2_1):
    # This test is identical to testDriverWithoutSpan, but uses raw JSON strings
    # for inputs and expects against the raw JSON output of the driver, to
    # better illustrate the JSON I/O contract of the driver.
    split1 = os.path.join(_TEST_INPUT_DIR, 'split1', 'data')
    io_utils.write_string_file(split1, 'testing')
    os.utime(split1, (0, 1))
    split2 = os.path.join(_TEST_INPUT_DIR, 'split2', 'data')
    io_utils.write_string_file(split2, 'testing2')
    os.utime(split2, (0, 3))

    expected_result_from_file = _load_test_file('expected_output_metadata.json')
    if use_pipeline_spec_2_1:
      executor_invocation = _load_test_file('executor_invocation.json')
    else:
      executor_invocation = _load_test_file('executor_invocation_legacy.json')

    serialized_args = ['--json_serialized_invocation_args', executor_invocation]

    if use_pipeline_spec_2_1:
      inputs_spec = pipeline_pb2.ComponentInputsSpec()
      inputs_spec.parameters['input_config'].parameter_type = (
          pipeline_pb2.ParameterType.STRING
      )
      serialized_args.extend([
          '--json_serialized_inputs_spec_args',
          json_format.MessageToJson(message=inputs_spec),
      ])

    # Invoke the driver
    driver.main(driver._parse_flags(serialized_args))

    # Check the output metadata file for the expected outputs
    with fileio.open(_TEST_OUTPUT_METADATA_JSON, 'rb') as output_meta_json:
      self.assertEqual(
          json.dumps(
              json.loads(output_meta_json.read()), indent=2, sort_keys=True
          ),
          json.dumps(
              json.loads(expected_result_from_file), indent=2, sort_keys=True
          ),
      )
