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
"""Tests for Kubeflow V2 CLI handler."""

import os
import sys
from typing import Mapping, Optional, Sequence, Text

import click
from googleapiclient import http
import mock
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration.kubeflow.v2.proto import pipeline_pb2
from tfx.tools.cli import labels
from tfx.tools.cli.kubeflow_v2 import labels as kubeflow_labels
from tfx.tools.cli.kubeflow_v2.handler import kubeflow_v2_handler
from tfx.utils import io_utils
from tfx.utils import test_case_utils

from google.protobuf import json_format

_TEST_PIPELINE_NAME = 'chicago_taxi_kubeflow'
_TEST_PIPELINE_JOB_NAME = 'chicago_taxi_kubeflow_20200101000000'
_TEST_PROJECT_1 = 'gcp_project_1'
_TEST_PROJECT_2 = 'gcp_project_2'  # _TEST_PROJECT_2 is assumed to have no runs.
_TEST_TFX_IMAGE = 'gcr.io/tfx-oss-public/tfx:latest'
_DUMMY_APIKEY = 'dummy-api-key'

# A good pipeline run JSON dict.
_VALID_RUN = {
    'name':
        'projects/{}/pipelineJobs/{}'.format(_TEST_PROJECT_1,
                                             _TEST_PIPELINE_JOB_NAME),
    'createTime':
        '2020-01-01T00:00:00.000000Z',
    'endTime':
        '2020-01-01T00:01:00.000000Z',
    'displayName':
        'dummy_pipeline',
    'state':
        'SUCCEEDED',
    'spec': {
        'pipelineContext': _TEST_PIPELINE_NAME
    }
}

# Expected job detail page link associated with _VALID_RUN.
_VALID_LINK = 'https://console.cloud.google.com/ai-platform/pipelines/runs/chicago_taxi_kubeflow_20200101000000?project=gcp_project_1'

# A pipeline run JSON dict with invalid name.
_ILLEGALLY_NAMED_RUN = {
    'name': 'ThisIsNotAValidName',
    'createTime': '2020-01-01T00:00:00.000000Z',
    'endTime': '2020-01-01T00:01:00.000000Z',
    'displayName': 'dummy_pipeline',
    'state': 'SUCCEEDED',
    'spec': {
        'pipelineContext': _TEST_PIPELINE_NAME
    }
}

# Mock response for get job request.
_GET_RESPONSES = {
    'projects/{}/'
    'pipelineJobs/{}'.format(_TEST_PROJECT_1, _TEST_PIPELINE_JOB_NAME):
        _VALID_RUN
}

# Mock response for list job request
_LIST_RESPONSES = {
    'projects/{}'.format(_TEST_PROJECT_1): {
        'pipelineJobs': [{
            'name':
                'projects/{}/pipelineJobs/{}'.format(_TEST_PROJECT_1,
                                                     _TEST_PIPELINE_JOB_NAME),
            'createTime':
                '2020-01-01T00:00:00.000000Z',
            'endTime':
                '2020-01-01T00:01:00.000000Z',
            'displayName':
                'dummy_pipeline',
            'spec': {
                'pipelineContext': _TEST_PIPELINE_NAME
            },
        }],
    },
    'projects/{}'.format(_TEST_PROJECT_2): {}
}


# The following mock is used when the side effect of subprocess call is not
# critical.
def _mock_subprocess_noop(cmd, env):
  del env  # Unused for this mock, but is expected to be passed.
  click.echo(cmd)
  return 0


# The following mock is used when we need to compile the pipeline by calling its
# DSL file.
# The subprocess call will be mostly in the following format
# [sys.executable, pipeline_dsl_path] where in the tests pipeline_dsl_path can
# be pointed to test_pipeline_(1|2|bad).py
# when it's test_pipeline_(1|2).py a pipeline spec json file is written
# under cwd/pipeline.json;
# when it's calling test_pipeline_bad.py the process exists with
# code 1.
def _mock_subprocess_call(cmd: Sequence[Optional[Text]],
                          env: Mapping[Text, Text]) -> int:
  """Mocks the subprocess call."""
  assert len(cmd) == 2, 'Unexpected number of commands: {}'.format(cmd)
  del env
  dsl_path = cmd[1]

  if dsl_path.endswith('test_pipeline_bad.py'):
    sys.exit(1)
  if not dsl_path.endswith(
      'test_pipeline_1.py') and not dsl_path.endswith(
          'test_pipeline_2.py'):
    raise ValueError('Unexpected dsl path: {}'.format(dsl_path))

  spec_pb = pipeline_pb2.PipelineSpec(
      pipeline_info=pipeline_pb2.PipelineInfo(name='chicago_taxi_kubeflow'))
  runtime_pb = pipeline_pb2.PipelineJob.RuntimeConfig(
      gcs_output_directory=os.path.join(os.environ['HOME'], 'tfx', 'pipelines',
                                        'chicago_taxi_kubeflow'))
  job_pb = pipeline_pb2.PipelineJob(runtime_config=runtime_pb)
  job_pb.pipeline_spec.update(json_format.MessageToDict(spec_pb))
  io_utils.write_string_file(
      file_name='pipeline.json',
      string_value=json_format.MessageToJson(message=job_pb, sort_keys=True))
  return 0


# Mock the Python API client class for testing purpose.
class _MockClient(object):
  """Mocks Python Google API client."""

  def projects(self):  # pylint: disable=invalid-name
    return _MockProjectsResource()


class _MockProjectsResource(object):
  """Mocks API Resource returned by projects()."""

  def pipelineJobs(self):  # pylint: disable=invalid-name
    return _MockPipelineJobsResource()


class _MockPipelineJobsResource(object):
  """Mocks API Resource returned by pipelineJobs()."""

  class _MockListRequest(http.HttpRequest):

    def __init__(self, parent: Text):
      self._parent = parent

    def execute(self):
      return _LIST_RESPONSES.get(self._parent)

  class _MockGetRequest(http.HttpRequest):

    def __init__(self, name: Text):
      self._name = name

    def execute(self):
      return _GET_RESPONSES.get(self._name)

  def list(self, parent: Text):  # pylint: disable=invalid-name
    """Mocks the list request."""
    return self._MockListRequest(parent=parent)

  def get(self, name: Text):  # pylint: disable=invalid-name
    """Mocks get job request."""
    return self._MockGetRequest(name=name)


class KubeflowV2HandlerTest(test_case_utils.TfxTest):

  def setUp(self):
    super(KubeflowV2HandlerTest, self).setUp()
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')

    self._home = self.tmp_dir
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))
    self.enter_context(test_case_utils.override_env_var('HOME', self._home))
    self._kubeflow_v2_home = os.path.join(self._home, 'kubeflow_v2')
    self.enter_context(
        test_case_utils.override_env_var('KUBEFLOW_V2_HOME',
                                         self._kubeflow_v2_home))

    # Flags for handler.
    self.engine = 'kubeflow_v2'
    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'test_pipeline_1.py')
    self.bad_pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                          'test_pipeline_bad.py')
    self.pipeline_name = _TEST_PIPELINE_NAME
    self.pipeline_root = os.path.join(self._home, 'tfx', 'pipelines',
                                      self.pipeline_name)
    self.run_id = 'dummyID'

    # Pipeline args for mocking subprocess
    self.pipeline_args = {
        'pipeline_name': _TEST_PIPELINE_NAME,
        'pipeline_dsl_path': self.pipeline_path
    }

    # Setting up Mock for API client, so that this Python test is hermatic.
    # subprocess Mock will be setup per-test.
    self.addCleanup(mock.patch.stopall)

  def testGetJobName(self):
    self.assertEqual(_TEST_PIPELINE_JOB_NAME,
                     kubeflow_v2_handler._get_job_name(_VALID_RUN))

  def testGetJobNameInvalidName(self):
    with self.assertRaisesRegex(RuntimeError, 'Invalid job name is received.'):
      kubeflow_v2_handler._get_job_name(_ILLEGALLY_NAMED_RUN)

  def testGetJobLink(self):
    self.assertEqual(
        _VALID_LINK,
        kubeflow_v2_handler._get_job_link(
            job_name=_TEST_PIPELINE_JOB_NAME, project_id=_TEST_PROJECT_1))

  @mock.patch('subprocess.call', _mock_subprocess_call)
  def testSavePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    pipeline_args = handler._extract_pipeline_args()
    handler._save_pipeline(pipeline_args)
    self.assertTrue(
        fileio.exists(
            os.path.join(handler._handler_home_dir,
                         self.pipeline_args[labels.PIPELINE_NAME])))

  @mock.patch('subprocess.call', _mock_subprocess_call)
  def testCreatePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    handler.create_pipeline()
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertTrue(
        fileio.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))

  @mock.patch('subprocess.call', _mock_subprocess_call)
  def testCreatePipelineExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    handler.create_pipeline()
    # Run create_pipeline again to test.
    with self.assertRaises(SystemExit) as err:
      handler.create_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" already exists.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('subprocess.call', _mock_subprocess_call)
  def testUpdatePipeline(self):
    # First create pipeline with test_pipeline.py
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_1.py')
    flags_dict_1 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_1
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict_1)
    handler.create_pipeline()

    # Update test_pipeline and run update_pipeline
    pipeline_path_2 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_2.py')
    flags_dict_2 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_2
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict_2)
    handler.update_pipeline()
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertTrue(
        fileio.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))

  @mock.patch('subprocess.call', _mock_subprocess_call)
  def testUpdatePipelineNoPipeline(self):
    # Update pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.update_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('subprocess.call', _mock_subprocess_call)
  def testCompilePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        kubeflow_labels.TFX_IMAGE_ENV: _TEST_TFX_IMAGE,
        kubeflow_labels.GCP_PROJECT_ID_ENV: _TEST_PROJECT_1
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      _ = handler.compile_pipeline()
    self.assertIn('Pipeline compiled successfully', captured.contents())

  @mock.patch('subprocess.call', _mock_subprocess_call)
  def testCompilePipelineNoPipelineArgs(self):
    # Test against a ill-formed pipeline DSL.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.bad_pipeline_path,
        kubeflow_labels.TFX_IMAGE_ENV: _TEST_TFX_IMAGE,
        kubeflow_labels.GCP_PROJECT_ID_ENV: _TEST_PROJECT_1
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    # Compilation will fail and a SystemExist will be thrown by subprocess.
    with self.assertRaises(SystemExit):
      _ = handler.compile_pipeline()

  def testListPipelinesNonEmpty(self):
    # First create two pipelines in the dags folder.
    handler_pipeline_path_1 = os.path.join(os.environ['KUBEFLOW_V2_HOME'],
                                           'pipeline_1')
    handler_pipeline_path_2 = os.path.join(os.environ['KUBEFLOW_V2_HOME'],
                                           'pipeline_2')
    fileio.makedirs(handler_pipeline_path_1)
    fileio.makedirs(handler_pipeline_path_2)

    # Now, list the pipelines
    flags_dict = {labels.ENGINE_FLAG: kubeflow_labels.KUBEFLOW_V2_ENGINE}
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)

    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('pipeline_1', captured.contents())
    self.assertIn('pipeline_2', captured.contents())

  def testListPipelinesEmpty(self):
    flags_dict = {labels.ENGINE_FLAG: kubeflow_labels.KUBEFLOW_V2_ENGINE}
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('No pipelines to display.', captured.contents())

  @mock.patch('subprocess.call', _mock_subprocess_call)
  def testDeletePipeline(self):
    # First create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    handler.create_pipeline()

    # Now delete the pipeline created aand check if pipeline folder is deleted.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    handler.delete_pipeline()
    handler_pipeline_path = os.path.join(
        handler._handler_home_dir, self.pipeline_args[labels.PIPELINE_NAME], '')
    self.assertFalse(fileio.exists(handler_pipeline_path))

  def testDeletePipelineNonExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = kubeflow_v2_handler.KubeflowV2Handler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.delete_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            flags_dict[labels.PIPELINE_NAME]))


# TODO(b/169095387): re-surrect the tests related with run commandwhen the
# a unified client becomes vailable.


if __name__ == '__main__':
  tf.test.main()
