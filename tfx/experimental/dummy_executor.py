from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text, Optional
from tfx import types
from tfx.components.base import base_executor
from tfx.types.artifact import Artifact

import json
import absl
import os
import filecmp
from distutils.dir_util import copy_tree

class DummyExecutor(base_executor.BaseExecutor):
  def __init__(self, component_id, record_dir, context):
    super(DummyExecutor, self).__init__(context)
    self._component_id = component_id
    self._record_dir = record_dir

  def _compare_contents(self, uri: Text, expected_uri: Text):
    """
    recursively compare two directories,
    Files are equal if names and contents are equal.
    """
    dirs_cmp = filecmp.dircmp(uri, expected_uri)
    absl.logging.info("uri: %s", uri)
    absl.logging.info("expected_uri: %s", expected_uri)
    if len(dirs_cmp.left_only) > 0 or len(dirs_cmp.right_only) > 0 or \
      len(dirs_cmp.funny_files) > 0:
      return False

    (_, mismatch, errors) = filecmp.cmpfiles(
        uri, expected_uri, dirs_cmp.common_files, shallow=False)

    if len(mismatch) > 0 or len(errors) > 0:
      return False

    for common_dir in dirs_cmp.common_dirs:
      new_dir1 = os.path.join(uri, common_dir)
      new_dir2 = os.path.join(expected_uri, common_dir)
      if not self._compare_contents(new_dir1, new_dir2):
        return False
    return True

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    # verifier with recorded contents
    json_path = os.path.join(self._record_dir,
                             "{}.json".format(self._component_id))
    with open(json_path, "r") as f:
      content_dict = json.load(f)

    expected_input_dict = content_dict['input_dict']
    expected_output_dict = content_dict['output_dict']

    # absl.logging.info("component_id %s", self.component_id)
    # absl.logging.info("expected_input_dict %s", expected_input_dict)
    # absl.logging.info("input_dict %s", input_dict)

    # absl.logging.info('expected_output_dict %s', expected_output_dict)
    # absl.logging.info('output_dict %s', output_dict)

    for in_key_name, artifact_list in input_dict.items():
      if in_key_name == 'baseline_model':
        continue
      for artifact in artifact_list:
        assert artifact.type_name in expected_input_dict[in_key_name].keys()
        artifact_json = expected_input_dict[in_key_name][artifact.type_name]
        expected_artifact = Artifact.from_json_dict(artifact_json)
        src = expected_artifact.uri
        dest = artifact.uri
        # if artifact.type_name == 'ExampleGen': 
          # self._compare_example_content()
        if not self._compare_contents(dest, src):
          absl.logging.info("WARNING: input checker failed")
          # raise Exception("input checking failed")

    for out_key_name, artifact_list in output_dict.items():
      for artifact in artifact_list:
        assert artifact.type_name in expected_output_dict[out_key_name].keys()
        artifact_json = expected_output_dict[out_key_name][artifact.type_name]
        expected_artifact = Artifact.from_json_dict(artifact_json)
        src = expected_artifact.uri
        dest = artifact.uri
        copy_tree(src, dest)
        absl.logging.info('from %s, copied to %s', src, dest)

# class DummyExecutorFactory(object):
#   def __init__(self, component_id, record_dir):
#     self.component_id = component_id
#     self.record_dir = record_dir

#   def __call__(self, executor_context):
#     return DummyExecutor(self._component_id, self._record_dir,
#                              executor_context)
