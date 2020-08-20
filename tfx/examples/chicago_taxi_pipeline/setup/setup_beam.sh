#!/bin/bash
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -u

WORK_DIR="/tmp/beam"
BEAM_DIR="$WORK_DIR/beam"
GIT_COMMAND=`which git`
BEAM_REPO="https://github.com/apache/beam"
# LINT.IfChange
BEAM_BRANCH="release-2.22.0"
# LINT.ThenChange(../../../dependencies.py)


# TODO(BEAM-2530): Support Java 11 when BEAM-2530 is done.
function check_java() {
  if type -p java; then
    _java=java
  elif [[ -n "$JAVA_HOME" ]] && [[ -x "$JAVA_HOME/bin/java" ]];  then
    _java="$JAVA_HOME/bin/java"
  else
    echo "No java found. Please install Java 1.8"
    exit 1
  fi

  if [[ "$_java" ]]; then
    version=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
    if [[ ! `echo $version | sed "s/1\.8\..*/1.8/"` == "1.8" ]]; then
      echo "Java version $version. The script requires Java 1.8."
      exit 1
    fi
  fi
}

# TODO(b/139747527): Use artifacts instead of building from source once they are
# published.
function install_beam(){
  echo "Installing Beam from source code at $BEAM_REPO Branch $BEAM_BRANCH"
  echo "Using work directory $WORK_DIR"
  mkdir -p $WORK_DIR
  if [ -z "$GIT_COMMAND" ]; then
    echo "ERROR: GIT is needed to download and build Beam."
  else
    echo "Using $GIT_COMMAND to download Beam source code."
    cd $WORK_DIR && $GIT_COMMAND clone $BEAM_REPO && cd $BEAM_DIR && $GIT_COMMAND checkout $BEAM_BRANCH
    echo "Beam cloned in $BEAM_DIR"
  fi
}

function update_beam(){
  mkdir -p $WORK_DIR
  if [ ! -z "$GIT_COMMAND" ]; then
    echo "Using $GIT_COMMAND to update Beam source code."
    cd $BEAM_DIR && $GIT_COMMAND checkout $BEAM_BRANCH && $GIT_COMMAND pull --rebase
  fi
}
