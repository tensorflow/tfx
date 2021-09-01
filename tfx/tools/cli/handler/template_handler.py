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
"""Handler for Template related operations.

Handles operations for templates in tfx/experimental/templates/ directory.
"""

import collections
import os
import re
from typing import Dict, Any, List, Pattern, Set
import urllib.request

import click

from tfx.dsl.io import fileio
from tfx.tools.cli import labels

_PLACEHOLDER_PIPELINE_NAME = re.compile('{{PIPELINE_NAME}}')
_PIPELINE_NAME_ESCAPE_CHAR = ['\\', '\'', '"', '/']
_IMPORT_FROM_PACKAGE = re.compile(
    r'from tfx\.experimental\.templates\.[^\.]+\.')
_IMPORT_FROM_LOCAL_DIR = 'from '

_TemplateFilePath = collections.namedtuple('_TemplateFilePath', ['src', 'dst'])
_ADDITIONAL_FILE_PATHS = {
    'taxi': [  # template name
        _TemplateFilePath(
            'examples/chicago_taxi_pipeline/data/simple/data.csv',
            'data/data.csv'),
    ],
    'penguin': [
        _TemplateFilePath(
            'http://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_processed.csv',
            'data/data.csv'),
    ],
}
_IGNORE_FILE_PATHS = {
    'taxi': [  # template name
        'e2e_tests',
    ],
    'penguin': [
        'e2e_tests',
    ],
}


def _tfx_src_dir() -> str:
  """Get tfx directory in the source tree.

    We should find tfx
    from tfx/tools/cli/handler/template_handler.py.
  Returns:
    Path to the directory containing tfx sources.
  """
  return os.path.dirname(  # tfx/
      os.path.dirname(  # tools/
          os.path.dirname(  # cli/
              os.path.dirname(os.path.abspath(__file__)))))  # handler/


def _templates_src_dir() -> str:
  """Get template directory in the source tree.

    We should find tfx/experimental/templates
    from tfx/tools/cli/handler/template_handler.py.
  Returns:
    Path to the directory containing template sources.
  """
  return os.path.join(_tfx_src_dir(), 'experimental', 'templates')


def list_template() -> List[str]:
  """List available templates by inspecting template source directory.

  Returns:
    List of template names which is same as directory name.
  """
  templates_dir = _templates_src_dir()
  names = []
  for f in os.listdir(templates_dir):
    if f.startswith('_'):
      continue
    if not os.path.isdir(os.path.join(templates_dir, f)):
      continue
    names.append(f)
  return names


def _copy_and_replace_placeholder_dir(
    src: str, dst: str, ignore_paths: Set[str],
    replace_dict: Dict[Pattern[str], str]) -> None:
  """Copy a directory to destination path and replace the placeholders."""
  if not os.path.isdir(dst):
    if os.path.exists(dst):
      raise RuntimeError(
          'Cannot copy template directory {}. Already a file exists.'.format(
              src))
    fileio.makedirs(dst)
  for f in os.listdir(src):
    src_path = os.path.join(src, f)
    dst_path = os.path.join(dst, f)
    if src_path in ignore_paths:
      continue

    if os.path.isdir(src_path):
      if f.startswith('_'):  # Excludes __pycache__ and other private folders.
        continue
      _copy_and_replace_placeholder_dir(src_path, dst_path, ignore_paths,
                                        replace_dict)
    else:  # a file.
      if f.endswith('.pyc'):  # Excludes .pyc
        continue
      _copy_and_replace_placeholder_file(src_path, dst_path, replace_dict)


def _copy_and_replace_placeholder_file(
    src: str, dst: str, replace_dict: Dict[Pattern[str], str]) -> None:
  """Copy a file to destination path and replace the placeholders."""
  click.echo('{} -> {}'.format(os.path.basename(src), dst))
  with open(src) as fp:
    contents = fp.read()
  for orig_regex, new in replace_dict.items():
    contents = orig_regex.sub(new, contents)
  with open(dst, 'w') as fp:
    fp.write(contents)


def _sanitize_pipeline_name(name: str) -> str:
  """Escape special characters to make a valid directory name."""
  for escape_char in _PIPELINE_NAME_ESCAPE_CHAR:
    name = name.replace(escape_char, '\\' + escape_char)
  return name


def copy_template(flags_dict: Dict[str, Any]) -> None:
  """Copy template flags_dict["model"] to flags_dict["dest_dir"].

  Copies all *.py and README files in specified template, and replace
  the content of the files.

  Args:
    flags_dict: Should have pipeline_name, model and dest_dir.
  """
  model = flags_dict[labels.MODEL]
  pipeline_name = _sanitize_pipeline_name(flags_dict[labels.PIPELINE_NAME])
  template_dir = os.path.join(_templates_src_dir(), model)
  if not os.path.isdir(template_dir):
    raise ValueError('Model {} does not exist.'.format(model))
  destination_dir = flags_dict[labels.DESTINATION_PATH]

  ignore_paths = {
      os.path.join(template_dir, x) for x in _IGNORE_FILE_PATHS.get(model, [])
  }
  replace_dict = {
      _IMPORT_FROM_PACKAGE: _IMPORT_FROM_LOCAL_DIR,
      _PLACEHOLDER_PIPELINE_NAME: pipeline_name,
  }
  _copy_and_replace_placeholder_dir(template_dir, destination_dir, ignore_paths,
                                    replace_dict)
  for additional_file in _ADDITIONAL_FILE_PATHS.get(model, []):
    dst_path = os.path.join(destination_dir, additional_file.dst)
    fileio.makedirs(os.path.dirname(dst_path))

    if additional_file.src.startswith(('http://', 'https://')):
      urllib.request.urlretrieve(additional_file.src, dst_path)
    else:
      src_path = os.path.join(_tfx_src_dir(), additional_file.src)
      fileio.copy(src_path, dst_path)
