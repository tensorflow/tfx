# Lint as: python2, python3
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
"""Abstract TFX executor class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text, Type
from tfx import types
from tfx.types.artifact import Artifact
from tfx.components.base import base_executor 

import json
import absl
import os
import filecmp
# import shutil
# expected_input_dict: Dict[Text, List[types.Artifact]],
# expected_output_dict: Dict[Text, List[types.Artifact]])
# # take in expected inputs/outputs from recorded contents

def make_recorder_executor(original_executor: base_executor.BaseExecutor, record_dir: Text, component_id: Text) -> base_executor.BaseExecutor:
  
  class Recorder(base_executor.BaseExecutor):
    
    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
       output_dict: Dict[Text, List[types.Artifact]],
       exec_properties: Dict[Text, Any]) -> None:
      # read input contents to the record dir.
      absl.logging.info("Recorder, recording to %s", record_dir)
      record_fn = open(os.path.join(record_dir, "{}.json".format(component_id)), "w")
      # self.copy(input_dict)
      content_dict = {}
      input_uri_dict, output_uri_dict = {}, {}
      absl.logging.info("input_dict", input_dict)
      for input_component_id, in_artifact_list in input_dict.items():
        input_uri_dict[input_component_id] = {}
        for artifact in in_artifact_list:
          input_uri_dict[input_component_id][artifact.type_name] = artifact.uri

      content_dict['input_dict'] = input_uri_dict
      
      # original_executor = original_executor_class()
      original_executor.Do(input_dict, output_dict, exec_properties)
      absl.logging.info("output_dict", output_dict)

      for output_component_id, out_artifact_list in output_dict.items():
        output_uri_dict[output_component_id]  = {}
        for artifact in out_artifact_list:
          output_uri_dict[output_component_id] [artifact.type_name] = artifact.uri
        
      content_dict['output_dict'] = output_uri_dict

      record_fn.write(json.dumps(content_dict))
      # record output contents to the record dir.
      # self.copy(output_dict)
      record_fn.close()
  return Recorder()
# '''def copy(self, artifact_dict):
#       for component_id, in_artifact_list in artifact_dict.items():
#                           for in_artifact in in_artifact_list:
#                             src_path = in_artifact.uri
#                             dest_path = src_path.replace("/usr/local/google/home/sujip/tfx/pipelines/", "")
#                             dest_path = dest_path[:dest_path.rfind('/')] # remove trailing number
#                             dest_path = os.path.join(record_dir, dest_path)
#                             if not os.path.exists(dest_path):
#                               os.makedirs(dest_path)
#                             shutil.copytree(src_path, dest_path)'''