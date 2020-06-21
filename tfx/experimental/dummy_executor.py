from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text, Optional
from tfx import types
from tfx.components.base import base_executor
from tfx.types.artifact import Artifact
from tfx.orchestration import metadata
from tfx.types import standard_artifacts

import json
import absl
import os
import filecmp
from distutils.dir_util import copy_tree
from tfx.orchestration.metadata import Metadata
class BaseDummyExecutor(base_executor.BaseExecutor):
  def __init__(self, component_id, record_dir, context):
    super(BaseDummyExecutor, self).__init__(context)
    absl.logging.info("Running DummyExecutor, component_id %s", component_id)
    self._component_id = component_id
    self._record_dir = record_dir
    # self._metadata_connection_config = metadata.sqlite_metadata_connection_config(metadata_dir)
    # self._metadata = Metadata(metadata.sqlite_metadata_connection_config(metadata_dir))
    

  def _compare_contents(self, uri: Text, expected_uri: Text):
    """
    TODO: contents change every run
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
    # print("input_dict", input_dict)
    # print("output_dict", output_dict)
    for input_key, artifact_list in input_dict.items():
      for artifact in artifact_list:
        if artifact.type_name == 'ExternalArtifact':
            continue
        dest = artifact.uri
        src = dest.replace(os.path.join(os.environ['HOME'], "tfx/pipelines/chicago_taxi_beam/"), "")
        src = src[:src.rfind('/')] # remove trailing number
        src = os.path.join(self._record_dir, src)
        # print("src {} test_dir {}".format(src, os.path.join(self._record_dir, componentid, input_key)) ) need component id from component info
        if not self._compare_contents(dest, src):
          absl.logging.info("WARNING: input checker failed")

    for output_key, artifact_list in output_dict.items():
      for artifact in artifact_list:
        dest = artifact.uri
        src = dest.replace(os.path.join(os.environ['HOME'], "tfx/pipelines/chicago_taxi_beam/"), "")
        src = src[:src.rfind('/')] # remove trailing number
        src = os.path.join(self._record_dir, src)
        # print("src: {}, path: {}".format(src, os.path.join(self._record_dir,artifact.type_name, output_key)))
        copy_tree(src, dest)
        absl.logging.info('from %s, copied to %s', src, dest)

