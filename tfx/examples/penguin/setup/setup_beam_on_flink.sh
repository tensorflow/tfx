#!/bin/bash
# Copyright 2019 Google LLC
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
# LINT.IfChange
FLINK_VERSION="1.12.1"
# LINT.ThenChange(../penguin_pipeline_local.py)
FLINK_NAME="flink-$FLINK_VERSION"
FLINK_BINARY="$FLINK_NAME-bin-scala_2.12.tgz"
FLINK_DOWNLOAD_URL="http://archive.apache.org/dist/flink/flink-$FLINK_VERSION/$FLINK_BINARY"

function setup_flink() {
  if [ ! -d $WORK_DIR/$FLINK_NAME ]; then
    echo "SETUP FLINK at $WORK_DIR/$FLINK_NAME"
    cd $WORK_DIR && curl $FLINK_DOWNLOAD_URL -o $WORK_DIR/$FLINK_BINARY  && tar -xvf $FLINK_BINARY
    if [ $? != 0 ]; then
      echo "ERROR: Unable to download Flink from $FLINK_DOWNLOAD_URL." \
            "Please make sure you have working internet and you have" \
            "curl(https://en.wikipedia.org/wiki/CURL) on your machine." \
            "Alternatively, you can also manually download Flink archive"\
            "and place it at $FLINK_DOWNLOAD_URL and extract Flink"\
            "to $WORK_DIR/$FLINK_NAME"
      exit 1
    fi
    echo "FLINK SETUP DONE at $WORK_DIR/$FLINK_NAME"
  fi
}

function start_flink() {
  echo "Starting flink at $WORK_DIR/$FLINK_NAME"
  local flink_conf=$WORK_DIR/$FLINK_NAME/conf/flink-conf.yaml
  local parallelism=$(get_parallelism)
  sed -i "s/taskmanager.numberOfTaskSlots: [0-9]*/taskmanager.numberOfTaskSlots: $parallelism/g" $flink_conf
  sed -i "s/#*rest.bind-address: .*/rest.bind-address: localhost/g" $flink_conf

  cd $WORK_DIR/$FLINK_NAME && ./bin/stop-cluster.sh && ./bin/start-cluster.sh
  echo "Flink running from $WORK_DIR/$FLINK_NAME"
}

# LINT.IfChange
function get_parallelism(){
   python -c "
import multiprocessing
try:
  parallelism = multiprocessing.cpu_count()
except NotImplementedError:
  parallelism = 1
print(parallelism)
"
}
# LINT.ThenChange(../penguin_pipeline_local.py)

function main(){
  setup_flink
  start_flink
}

main $@
