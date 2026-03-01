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
"""Wrapper for pip commands."""

import subprocess
import sys
from typing import List


def get_package_names() -> List[str]:
  freeze_output = subprocess.check_output(
      [sys.executable, '-m', 'pip', 'freeze', '--local']).decode('utf-8')
  result = []
  for line in freeze_output.split():
    line = line.lstrip()
    if not line or line[0] == '#' or '==' not in line:
      continue
    result.append(line.split('==')[0])
  return result
