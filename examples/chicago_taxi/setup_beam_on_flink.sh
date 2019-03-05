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

if [ "${VIRTUAL_ENV:-unset}" == "unset" ]; then
  echo "Please run the setup script from a vritual environment and make sure environment variable\
  VIRTUAL_ENV is set correctly."
  exit 1
fi

WORK_DIR="/tmp/beam"
BEAM_DIR="$WORK_DIR/beam"
GIT_COMMAND=`which git`
BEAM_REPO="https://github.com/apache/beam"
BEAM_BRANCH="release-2.11.0"

SETUP_FLINK=1
FLINK_VERSION="1.5.6"
FLINK_NAME="flink-$FLINK_VERSION"
FLINK_BINARY="$FLINK_NAME-bin-scala_2.11.tgz"
FLINK_DOWNLOAD_URL="http://archive.apache.org/dist/flink/flink-$FLINK_VERSION/$FLINK_BINARY"

echo "Setup Beam from source code at $BEAM_REPO Branch $BEAM_BRANCH"
echo "Using work directory $WORK_DIR"


# TODO(BEAM-6763): Use artifacts instead of building from source once they are published.
function install_beam(){
  mkdir -p $WORK_DIR
  if [ -z "$GIT_COMMAND" ]; then
    echo "ERROR: GIT is needed to download and build Beam."
  else
    echo "Using $GIT_COMMAND to download Beam source code."
    cd $WORK_DIR && $GIT_COMMAND clone $BEAM_REPO && $GIT_COMMAND checkout $BEAM_BRANCH
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

function setup_flink() {
  if [ $SETUP_FLINK == 1 ]; then
    if [ ! -d $WORK_DIR/$FLINK_NAME ]; then
      echo "SETUP FLINK at $WORK_DIR/$FLINK_NAME"
      cd $WORK_DIR && wget -P $WORK_DIR $FLINK_DOWNLOAD_URL && tar -xvf $FLINK_BINARY
      echo "FLINK SETUP DONE at $WORK_DIR/$FLINK_NAME"
    fi
  fi
}

function start_flink() {
  echo "Starting flink at $WORK_DIR/$FLINK_NAME"
  cd $WORK_DIR/$FLINK_NAME && ./bin/stop-cluster.sh && ./bin/start-cluster.sh
  echo "Flink running from $WORK_DIR/$FLINK_NAME"
}

function start_job_server() {
  echo "Starting Beam jobserver"
  cd $BEAM_DIR
  ./gradlew beam-runners-flink_2.11-job-server:runShadow -PflinkMasterUrl=localhost:8081
}

function main(){
  # Check and create the relevant directory
  if [ ! -d "$WORK_DIR" ]; then
    install_beam
  else
    echo "Work directory $WORK_DIR already exists."
    echo "Please delete $WORK_DIR in case of issue."
    update_beam
  fi
  setup_flink
  start_flink
  start_job_server
}

main $@
