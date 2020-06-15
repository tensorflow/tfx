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
from distutils.dir_util import copy_tree
# import shutil
class DummyExecutor(base_executor.BaseExecutor):
  def set_args(self, component_id, record_dir):
    self.component_id = component_id
    self.record_dir = record_dir
    
  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
     output_dict: Dict[Text, List[types.Artifact]],
     exec_properties: Dict[Text, Any]) -> None:
    absl.logging.info("component_id %s", self.component_id)
    # verifier with recorded contents
    with open(os.path.join(self.record_dir, "{}.json".format(self.component_id)), "r") as f:
      content_dict = json.load(f)
    
    input_uri_dict = content_dict['input_dict']
    output_uri_dict = content_dict['output_dict']

    absl.logging.info("input_uri_dict %s", input_uri_dict)
    absl.logging.info("input_dict %s", input_dict)

    absl.logging.info('output_uri_dict %s', output_uri_dict)
    absl.logging.info('output_dict %s', output_dict)
    
    for in_key_name, artifact_list in input_dict.items():
      if in_key_name == 'baseline_model':
          continue
      for artifact in artifact_list:
        assert artifact.type_name in input_uri_dict[in_key_name].keys()
        src = input_uri_dict[in_key_name][artifact.type_name]
        dest = artifact.uri
        if src == dest:
          continue
        copy_tree(src, dest)
    for out_key_name, artifact_list in output_dict.items():
      for artifact in artifact_list:
        assert artifact.type_name in output_uri_dict[out_key_name].keys()
        src = output_uri_dict[out_key_name][artifact.type_name]
        dest = artifact.uri
        if src == dest:
          continue
        copy_tree(src, dest)

# def make_dummy_executor_class(record_dir: Text, component_id: Text) -> base_executor.BaseExecutor:
#   """
#   TODO: figure out passing class reference work, instead of instance
#   """

#   # take in expected inputs/outputs from recorded contents
#   class DummyExecutor(base_executor.BaseExecutor):
#     def set_args(self, component_id, record_dir="/usr/local/google/home/sujip/record"):
#       self.component_id = component_id
#       self.record_dir = record_dir

#     def Do(self, input_dict: Dict[Text, List[types.Artifact]],
#        output_dict: Dict[Text, List[types.Artifact]],
#        exec_properties: Dict[Text, Any]) -> None:
#       # verifier with recorded contents
#       with open(os.path.join(record_dir, "{}.json".format(component_id)), "r") as f:
#         content_dict = json.load(f.read())
      
#       input_uri_dict = content_dict['input_dict']
#       output_uri_dict = content_dict['output_dict']

#       absl.logging.info("input_uri_dict %s", input_uri_dict)
#       absl.logging.info("input_dict %s", input_dict)

#       absl.logging.info('output_uri_dict %s', output_uri_dict)
#       absl.logging.info('output_dict %s', output_dict)

#       assert len(input_dict.keys()) == 1
#       assert len(input_uri_dict.keys()) == 1
#       assert len(output_dict.keys()) == 1
#       assert len(output_uri_dict.keys()) == 1
#       in_key_name = list(input_dict.keys())[0]
#       out_key_name = list(output_dict.keys())[0]

#       # assert set(input_uri_dict.keys()) == set(input_dict.keys())
#       # assert set(output_uri_dict.keys()) == set(output_dict.keys())
      
#       for artifact in input_dict[in_key_name]:
#         assert artifact.type_name in input_uri_dict[in_key_name].keys()
#         shutil.copytree(input_uri_dict[in_key_name][artifact.type_name], artifact.uri)
#       for artifact in output_dict[in_key_name]:
#         assert artifact.type_name in input_uri_dict[out_key_name].keys()
#         shutil.copytree(output_uri_dict[out_key_name][artifact.type_name], artifact.uri)

#   return DummyExecutor 

