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
"""E2E tests for Kubeflow V2 runner with Vertex Pipelines."""

import datetime
import os
import subprocess
from typing import Any, Dict, List, Optional

from absl import logging

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

from tfx.dsl.components.base import base_node
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.orchestration.kubeflow.v2 import vertex_client_utils
from tfx.utils import io_utils
from tfx.utils import test_case_utils

_POLLING_INTERVAL_IN_SECONDS = 60
_MAX_JOB_EXECUTION_TIME = datetime.timedelta(minutes=90)


class BaseKubeflowV2Test(test_case_utils.TfxTest):
  """Defines testing harness for pipeline on KubeflowV2DagRunner."""

  # The following environment variables need to be set prior to calling the test
  # in this file. All variables are required and do not have a default.

  # The src path to use to build docker image
  _REPO_BASE = os.environ.get('KFP_E2E_SRC')

  # The base container image name to use when building the image used in tests.
  _BASE_CONTAINER_IMAGE = os.environ.get('KFP_E2E_BASE_CONTAINER_IMAGE')

  # The project id to use to run tests.
  _GCP_PROJECT_ID = os.environ.get('KFP_E2E_GCP_PROJECT_ID')

  # The GCP region used to call the service.
  _GCP_REGION = os.environ.get('KFP_E2E_GCP_REGION')

  # The GCP bucket to use to write output artifacts.
  _BUCKET_NAME = os.environ.get('KFP_E2E_BUCKET_NAME')

  # The location of test user module file.
  # - Retrieved from inside the container subject to testing.
  # - Depends on the install path of TFX in the docker image.
  _MODULE_FILE = '/opt/conda/lib/python3.10/site-packages/tfx/examples/chicago_taxi_pipeline/taxi_utils.py'

  @classmethod
  def setUpClass(cls):
    super(BaseKubeflowV2Test, cls).setUpClass()

    if ':' not in cls._BASE_CONTAINER_IMAGE:
      # Generate base container image for the test if tag is not specified.
      cls.container_image = '{}:{}'.format(cls._BASE_CONTAINER_IMAGE,
                                           test_utils.random_id())

      # Create a container image for use by test pipelines.
      test_utils.build_and_push_docker_image(cls.container_image,
                                             cls._REPO_BASE)
    else:  # Use the given image as a base image.
      cls.container_image = cls._BASE_CONTAINER_IMAGE

  @classmethod
  def tearDownClass(cls):
    super(BaseKubeflowV2Test, cls).tearDownClass()

    if cls.container_image != cls._BASE_CONTAINER_IMAGE:
      # Delete container image used in tests.
      logging.info('Deleting image %s', cls.container_image)
      subprocess.run(
          ['gcloud', 'container', 'images', 'delete', cls.container_image],
          check=True)

  def setUp(self):
    super().setUp()
    self._test_id = test_utils.random_id()
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))
    self._test_output_dir = 'gs://{}/test_output'.format(self._BUCKET_NAME)
    self._test_data_dir = 'gs://{}/test_data/{}'.format(self._BUCKET_NAME,
                                                        self._test_id)
    self._output_filename = 'pipeline.json'
    self._serving_model_dir = os.path.join(self._test_output_dir, 'output')

    aiplatform.init(
        project=self._GCP_PROJECT_ID,
        location=self._GCP_REGION,
    )

  def _pipeline_root(self, pipeline_name: str):
    return os.path.join(self._test_output_dir, pipeline_name)

  def _delete_pipeline_output(self, pipeline_name: str):
    """Deletes output produced by the named pipeline."""
    io_utils.delete_dir(self._pipeline_root(pipeline_name))

  def _create_pipeline(
      self,
      pipeline_name: str,
      pipeline_components: List[base_node.BaseNode],
      beam_pipeline_args: Optional[List[str]] = None) -> tfx_pipeline.Pipeline:
    """Creates a pipeline given name and list of components."""
    return tfx_pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        components=pipeline_components,
        beam_pipeline_args=beam_pipeline_args)

  def _run_pipeline(self,
                    pipeline: tfx_pipeline.Pipeline,
                    parameter_values: Optional[Dict[str, Any]] = None,
                    exit_handler: Optional[base_node.BaseNode] = None) -> None:
    """Trigger the pipeline execution with a specific job ID."""
    # Ensure cleanup regardless of whether pipeline succeeds or fails.
    self.addCleanup(self._delete_pipeline_output,
                    pipeline.pipeline_info.pipeline_name)

    # Create DAG runner and add exit handler if present.
    v2_dag_runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
        default_image=self.container_image)
    v2_dag_runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=v2_dag_runner_config, output_filename=self._output_filename)
    if exit_handler:
      v2_dag_runner.set_exit_handler(exit_handler)
    v2_dag_runner.run(pipeline, write_out=True)

    # Create and submit Vertex job.
    self._job_id = pipeline.pipeline_name
    job = pipeline_jobs.PipelineJob(
        display_name=self._job_id,
        job_id=self._job_id,
        template_path=self._output_filename,
        parameter_values=parameter_values)
    job.submit()

    # Monitor job status.
    vertex_client_utils.poll_job_status(self._job_id, _MAX_JOB_EXECUTION_TIME,
                                        _POLLING_INTERVAL_IN_SECONDS)
