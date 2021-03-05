# Copyright 2021 Google LLC. All Rights Reserved.
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
"""An abstract class for the runner for both CAIP and uCAIP."""

import abc
import datetime
import json
from typing import Any, Dict, List, Optional, Text

from absl import logging
from google.cloud.aiplatform import aiplatform
from google.cloud.aiplatform.aiplatform_v1beta1.types import job_state
from googleapiclient import discovery
from googleapiclient import http

from tfx import types
from tfx.types import artifact_utils
from tfx.utils import telemetry_utils
from tfx.utils import version_utils

# Default contaier image being used for CAIP training jobs.
_TFX_IMAGE = 'gcr.io/tfx-oss-public/tfx:{}'.format(
    version_utils.get_image_version())

# Entrypoint of cloud AI platform training. The module comes from `tfx`
# package installation into a default location of 'python'.
_CONTAINER_COMMAND = ['python', '-m', 'tfx.scripts.run_executor']

# When uCAIP CustomJob is launched in GA, consider changing this
# region-specific endpoint into user-configurable.
_UCAIP_ENDPOINT = 'us-central1-aiplatform.googleapis.com'

_UCAIP_JOB_STATE_SUCCEEDED = job_state.JobState.JOB_STATE_SUCCEEDED
_UCAIP_JOB_STATE_FAILED = job_state.JobState.JOB_STATE_FAILED
_UCAIP_JOB_STATE_CANCELLED = job_state.JobState.JOB_STATE_CANCELLED


class AbstractJobClient(abc.ABC):
  """Abstract class interacting with CAIP CMLE job or uCAIP CustomJob."""
  JOB_STATES_COMPLETED = ()  # Job states for success, failure or cancellation
  JOB_STATES_FAILED = ()  # Job states for failure or cancellation

  def __init__(self):
    self.create_client()
    self._init_var()

  @abc.abstractmethod
  def _init_var(self) -> None:
    """Initializes class variables."""
    pass

  @abc.abstractmethod
  def create_client(self) -> None:
    """Creates the job client.

    Can also be used for recreating the job client (e.g. in the case of
    communication failure).

    Multiple job requests can be done in parallel if needed, by creating an
    instance of the class for each job. Note that one class instance should
    only be used for one job, as each instance stores variables (e.g. job_id)
    specific to each job.
    """
    pass

  @abc.abstractmethod
  def create_training_args(self, input_dict, output_dict, exec_properties,
                           executor_class_path, training_inputs,
                           job_id) -> Dict[Text, Any]:
    """Get training args for runner._launch_aip_training.

    The training args contain the inputs/outputs/exec_properties to the
    tfx.scripts.run_executor module.

    Args:
      input_dict: Passthrough input dict for tfx.components.Trainer.executor.
      output_dict: Passthrough input dict for tfx.components.Trainer.executor.
      exec_properties: Passthrough input dict for
        tfx.components.Trainer.executor.
      executor_class_path: class path for TFX core default trainer.
      training_inputs: Training input argument for AI Platform training job.
      job_id: Job ID for AI Platform Training job. If not supplied,
        system-determined unique ID is given.

    Returns:
      A dict containing the training arguments
    """
    pass

  @abc.abstractmethod
  def _create_job_spec(
      self,
      job_id: Text,
      training_input: Dict[Text, Any],
      job_labels: Optional[Dict[Text, Text]] = None) -> Dict[Text, Any]:
    """Creates the job spec.

    Args:
      job_id: The job ID of the AI Platform training job.
      training_input: Training input argument for AI Platform training job.
      job_labels: The dict of labels that will be attached to this job.

    Returns:
      The job specification.
    """
    pass

  @abc.abstractmethod
  def launch_job(self,
                 job_id: Text,
                 parent: Text,
                 training_input: Dict[Text, Any],
                 job_labels: Optional[Dict[Text, Text]] = None) -> None:
    """Launches a long-running job.

    Args:
      job_id: The job ID of the AI Platform training job.
      parent: The project name in the form of 'projects/{project_id}'
      training_input: Training input argument for AI Platform training job. See
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
          for the detailed schema.
      job_labels: The dict of labels that will be attached to this job.
    """
    pass

  @abc.abstractmethod
  def get_job_request(self) -> http.HttpRequest:
    """Gets the job request for the long-running job."""
    pass


class CAIPJobClient(AbstractJobClient):
  """Class for interacting with CAIP CMLE job."""

  JOB_STATES_COMPLETED = ('SUCCEEDED', 'FAILED', 'CANCELLED')
  JOB_STATES_FAILED = ('FAILED', 'CANCELLED')

  def create_client(self) -> None:
    """Creates the discovery job client.

    Can also be used for recreating the job client (e.g. in the case of
    communication failure).

    Multiple job requests can be done in parallel if needed, by creating an
    instance of the class for each job. Note that one class instance should
    only be used for one job, as each instance stores variables (e.g. job_id)
    specific to each job.
    """
    self._client = discovery.build('ml', 'v1')

  def _init_var(self) -> None:
    """Initializes class variables."""
    self._job_id = ''  # Assigned in self.launch_job()
    self._project_id = ''  # Assigned in self.launch_job()

  def create_training_args(self, input_dict: Dict[Text, List[types.Artifact]],
                           output_dict: Dict[Text, List[types.Artifact]],
                           exec_properties: Dict[Text, Any],
                           executor_class_path: Text,
                           training_inputs: Dict[Text, Any],
                           job_id: Optional[Text]) -> Dict[Text, Any]:
    """Get training args for runner._launch_aip_training.

    The training args contain the inputs/outputs/exec_properties to the
    tfx.scripts.run_executor module.

    Args:
      input_dict: Passthrough input dict for tfx.components.Trainer.executor.
      output_dict: Passthrough input dict for tfx.components.Trainer.executor.
      exec_properties: Passthrough input dict for
        tfx.components.Trainer.executor.
      executor_class_path: class path for TFX core default trainer.
      training_inputs: Training input argument for AI Platform training job.
        'pythonModule', 'pythonVersion' and 'runtimeVersion' will be inferred.
        For the full set of parameters, refer to
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
      job_id: Job ID for AI Platform Training job. If not supplied,
        system-determined unique ID is given. Refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#resource-job

    Returns:
      A dict containing the training arguments
    """
    training_inputs = training_inputs.copy()

    json_inputs = artifact_utils.jsonify_artifact_dict(input_dict)
    logging.info('json_inputs=\'%s\'.', json_inputs)
    json_outputs = artifact_utils.jsonify_artifact_dict(output_dict)
    logging.info('json_outputs=\'%s\'.', json_outputs)
    json_exec_properties = json.dumps(exec_properties, sort_keys=True)
    logging.info('json_exec_properties=\'%s\'.', json_exec_properties)

    # We use custom containers to launch training on AI Platform, which invokes
    # the specified image using the container's entrypoint. The default
    # entrypoint for TFX containers is to call scripts/run_executor.py. The
    # arguments below are passed to this run_executor entry to run the executor
    # specified in `executor_class_path`.
    container_command = _CONTAINER_COMMAND + [
        '--executor_class_path',
        executor_class_path,
        '--inputs',
        json_inputs,
        '--outputs',
        json_outputs,
        '--exec-properties',
        json_exec_properties,
    ]

    if not training_inputs.get('masterConfig'):
      training_inputs['masterConfig'] = {
          'imageUri': _TFX_IMAGE,
      }

    # Always use our own entrypoint instead of relying on container default.
    if 'containerCommand' in training_inputs['masterConfig']:
      logging.warn('Overriding custom value of containerCommand')
    training_inputs['masterConfig']['containerCommand'] = container_command

    # Pop project_id so AIP doesn't complain about an unexpected parameter.
    # It's been a stowaway in aip_args and has finally reached its destination.
    project = training_inputs.pop('project')
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      job_labels = telemetry_utils.get_labels_dict()

    # 'tfx_YYYYmmddHHMMSS' is the default job ID if not explicitly specified.
    job_id = job_id or 'tfx_{}'.format(
        datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    training_args = {
        'job_id': job_id,
        'project': project,
        'training_input': training_inputs,
        'job_labels': job_labels
    }

    return training_args

  def _create_job_spec(
      self,
      job_id: Text,
      training_input: Dict[Text, Any],
      job_labels: Optional[Dict[Text, Text]] = None) -> Dict[Text, Any]:
    """Creates the job spec.

    Args:
      job_id: The job ID of the AI Platform training job.
      training_input: Training input argument for AI Platform training job. See
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
          for the detailed schema.
      job_labels: The dict of labels that will be attached to this job.

    Returns:
      The job specification. See
      https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs
    """

    job_spec = {
        'jobId': job_id,
        'trainingInput': training_input,
        'labels': job_labels,
    }
    return job_spec

  def launch_job(self,
                 job_id: Text,
                 parent: Text,
                 training_input: Dict[Text, Any],
                 job_labels: Optional[Dict[Text, Text]] = None) -> None:
    """Launches a long-running job.

    Args:
      job_id: The job ID of the AI Platform training job.
      parent: The project name in the form of 'projects/{project_id}'
      training_input: Training input argument for AI Platform training job. See
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
          for the detailed schema.
      job_labels: The dict of labels that will be attached to this job.
    """

    job_spec = self._create_job_spec(job_id, training_input, job_labels)

    # Submit job to AIP Training
    logging.info('TrainingInput=%s', training_input)
    logging.info('Submitting job=\'%s\', project=\'%s\' to AI Platform.',
                 job_id, parent)
    request = self._client.projects().jobs().create(
        body=job_spec, parent=parent)
    self._job_id = job_id
    self._project_id = parent
    request.execute()

  def get_job_request(self) -> http.HttpRequest:
    """Gets the job request for the long-running job."""
    job_name = '{}/jobs/{}'.format(self._project_id, self._job_id)
    request = self._client.projects().jobs().get(name=job_name)
    return request


class UCAIPJobClient(AbstractJobClient):
  """Class for interacting with uCAIP CustomJob."""

  JOB_STATES_COMPLETED = (_UCAIP_JOB_STATE_SUCCEEDED, _UCAIP_JOB_STATE_FAILED,
                          _UCAIP_JOB_STATE_CANCELLED)
  JOB_STATES_FAILED = (_UCAIP_JOB_STATE_FAILED, _UCAIP_JOB_STATE_CANCELLED)

  def create_client(self) -> None:
    """Creates the Gapic job client.

    Can also be used for recreating the job client (e.g. in the case of
    communication failure).

    Multiple job requests can be done in parallel if needed, by creating an
    instance of the class for each job. Note that one class instance should
    only be used for one job, as each instance stores variables (e.g. job_id)
    specific to each job.
    """
    self._client = aiplatform.gapic.JobServiceClient(
        client_options=dict(api_endpoint=_UCAIP_ENDPOINT))

  def _init_var(self) -> None:
    """Initializes class variables."""
    self._job_name = ''  # Assigned in self.launch_job()

  def create_training_args(self, input_dict: Dict[Text, List[types.Artifact]],
                           output_dict: Dict[Text, List[types.Artifact]],
                           exec_properties: Dict[Text, Any],
                           executor_class_path: Text,
                           training_inputs: Dict[Text, Any],
                           job_id: Optional[Text]) -> Dict[Text, Any]:
    """Get training args for runner._launch_aip_training.

    The training args contain the inputs/outputs/exec_properties to the
    tfx.scripts.run_executor module.

    Args:
      input_dict: Passthrough input dict for tfx.components.Trainer.executor.
      output_dict: Passthrough input dict for tfx.components.Trainer.executor.
      exec_properties: Passthrough input dict for
        tfx.components.Trainer.executor.
      executor_class_path: class path for TFX core default trainer.
      training_inputs: Training input argument for AI Platform (Unified)
        custom training job.
        For the full set of parameters, refer to
        https://cloud.google.com/ai-platform-unified/docs/reference/rest/v1/projects.locations.customJobs
      job_id: Display name for AI Platform (Unified) custom training job. If not
        supplied, system-determined unique ID is given. Refer to
        https://cloud.google.com/ai-platform-unified/docs/reference/rest/v1/projects.locations.customJobs

    Returns:
      A dict containing the training arguments
    """
    training_inputs = training_inputs.copy()

    json_inputs = artifact_utils.jsonify_artifact_dict(input_dict)
    logging.info('json_inputs=\'%s\'.', json_inputs)
    json_outputs = artifact_utils.jsonify_artifact_dict(output_dict)
    logging.info('json_outputs=\'%s\'.', json_outputs)
    json_exec_properties = json.dumps(exec_properties, sort_keys=True)
    logging.info('json_exec_properties=\'%s\'.', json_exec_properties)

    # We use custom containers to launch training on AI Platform (unified),
    # which invokes the specified image using the container's entrypoint. The
    # default entrypoint for TFX containers is to call scripts/run_executor.py.
    # The arguments below are passed to this run_executor entry to run the
    # executor specified in `executor_class_path`.
    container_command = _CONTAINER_COMMAND + [
        '--executor_class_path',
        executor_class_path,
        '--inputs',
        json_inputs,
        '--outputs',
        json_outputs,
        '--exec-properties',
        json_exec_properties,
    ]

    if not training_inputs['worker_pool_specs'].get('container_spec'):
      training_inputs['worker_pool_specs']['container_spec'] = {
          'image_uri': _TFX_IMAGE,
      }

    # Always use our own entrypoint instead of relying on container default.
    if 'containerCommand' in training_inputs['worker_pool_specs'][
        'container_spec']:
      logging.warn('Overriding custom value of containerCommand')
    training_inputs['worker_pool_specs']['container_spec'][
        'command'] = container_command

    # Pop project_id so AIP doesn't complain about an unexpected parameter.
    # It's been a stowaway in aip_args and has finally reached its destination.
    project = training_inputs.pop('project')
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      job_labels = telemetry_utils.get_labels_dict()

    # 'tfx_YYYYmmddHHMMSS' is the default job display name if not explicitly
    # specified.
    job_id = job_id or 'tfx_{}'.format(
        datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    training_args = {
        'job_id': job_id,
        'project': project,
        'training_input': training_inputs,
        'job_labels': job_labels
    }

    return training_args

  def _create_job_spec(
      self,
      job_id: Text,
      training_input: Dict[Text, Any],
      job_labels: Optional[Dict[Text, Text]] = None) -> Dict[Text, Any]:
    """Creates the job spec.

    Args:
      job_id: The display name of the AI Platform (Unified) custom training job.
      training_input: Training input argument for AI Platform training job. See
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
          for the detailed schema.
      job_labels: The dict of labels that will be attached to this job.

    Returns:
      The job specification. See
      https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs
    """

    job_spec = {
        'display_name': job_id,
        'job_spec': training_input,
        'labels': job_labels,
    }
    return job_spec

  def launch_job(self,
                 job_id: Text,
                 parent: Text,
                 training_input: Dict[Text, Any],
                 job_labels: Optional[Dict[Text, Text]] = None) -> None:
    """Launches a long-running job.

    Args:
      job_id: The job ID of the AI Platform training job.
      parent: The project name in the form of 'projects/{project_id}'
      training_input: Training input argument for AI Platform training job. See
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
          for the detailed schema.
      job_labels: The dict of labels that will be attached to this job.
    """

    job_spec = self._create_job_spec(job_id, training_input, job_labels)

    # Submit job to AIP Training
    logging.info('TrainingInput=%s', training_input)
    logging.info('Submitting custom job=\'%s\', project=\'%s\' to AI Platform.',
                 job_id, parent)
    response = self._client.create_custom_job(parent=parent,
                                              custom_job=job_spec)
    self._job_name = response.name

  def get_job_request(self) -> http.HttpRequest:
    """Gets the job request for the long-running job."""
    return self._client.get_custom_job(self._job_name)


def get_job_client(enable_ucaip: bool = False):
  if enable_ucaip:
    return UCAIPJobClient()
  return CAIPJobClient()
