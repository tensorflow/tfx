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
"""Helper class to start TFX training jobs on CMLE."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import os
import time
from googleapiclient import discovery
from tfx.utils import io_utils
from tfx.utils import logging_utils
from tfx.utils.types import jsonify_tfx_type_dict


# TODO(ajaygopinathan): Add pydocs once this interface is finalized.
def start_cmle_training(input_dict, output_dict, exec_properties,
                        training_inputs):
  """Start a trainer job on CMLE."""
  training_inputs = training_inputs.copy()
  logger = logging_utils.get_logger(exec_properties['log_root'], 'exec')
  # Remove cmle_args from exec_properties so CMLE trainer doesn't call itself
  exec_properties['custom_config'].pop('cmle_training_args')

  json_inputs = jsonify_tfx_type_dict(input_dict)
  logger.info('json_inputs=\'%s\'.', json_inputs)
  json_outputs = jsonify_tfx_type_dict(output_dict)
  logger.info('json_outputs=\'%s\'.', json_outputs)
  json_exec_properties = json.dumps(exec_properties)
  logger.info('json_exec_properties=\'%s\'.', json_exec_properties)

  # Configure CMLE job
  api_client = discovery.build('ml', 'v1')
  job_args = [
      '--executor', 'Trainer', '--inputs', json_inputs, '--outputs',
      json_outputs, '--exec-properties', json_exec_properties
  ]
  training_inputs['args'] = job_args
  training_inputs['pythonModule'] = 'tfx.scripts.run_executor'

  # Pop project_id so CMLE doesn't complain about an unexpected parameter.
  # It's been a stowaway in cmle_args and has finally reached its destination.
  project = training_inputs.pop('project')
  project_id = 'projects/{}'.format(project)

  # Create TFX dist and add it to training_inputs
  local_package = io_utils.build_package()
  cloud_package = os.path.join(training_inputs['jobDir'],
                               os.path.basename(local_package))
  io_utils.copy_file(local_package, cloud_package, True)

  if 'packageUris' in training_inputs:
    training_inputs['packageUris'].append(cloud_package)
  else:
    training_inputs['packageUris'] = cloud_package

  job_name = 'tfx_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  job_spec = {'jobId': job_name, 'trainingInput': training_inputs}

  # Submit job to CMLE
  logger.info('Submitting job=\'{}\', project=\'{}\' to CMLE.'.format(
      job_name, project))
  request = api_client.projects().jobs().create(
      body=job_spec, parent=project_id)
  request.execute()

  # Wait for CMLE job to finish
  job_id = '{}/jobs/{}'.format(project_id, job_name)
  request = api_client.projects().jobs().get(name=job_id)
  response = request.execute()
  while response['state'] not in ('SUCCEEDED', 'FAILED'):
    time.sleep(60)
    response = request.execute()

  if response['state'] == 'FAILED':
    err_msg = 'Job \'{}\' did not succeed.  Detailed response {}.'.format(
        job_name, response)
    logger.error(err_msg)
    raise RuntimeError(err_msg)

  # CMLE training complete
  logger.info('Job \'{}\' successful.'.format(job_name))
