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

source `dirname "$(readlink -f "$0")"`/setup_beam.sh

if [ "${VIRTUAL_ENV:-unset}" == "unset" ]; then
  echo "Please run the setup script from a vritual environment and make sure environment variable\
  VIRTUAL_ENV is set correctly."
  exit 1
fi

SPARK_VERSION="2.4.3"
SPARK_NAME="spark-$SPARK_VERSION"
SPARK_ROOT="$SPARK_NAME-bin-hadoop2.7"
SPARK_BINARY="$SPARK_ROOT.tgz"
SPARK_DOWNLOAD_URL="http://archive.apache.org/dist/spark/$SPARK_NAME/$SPARK_BINARY"

SPARK_HOST=`hostname`
SPARK_PORT=7077
SPARK_MASTER_URL=spark://$SPARK_HOST:$SPARK_PORT
SPARK_CONF=spark.conf
SPARK_SECRET=`openssl rand -hex 20`

function setup_spark() {
  if [ ! -d $WORK_DIR/$SPARK_ROOT ]; then
    echo "SETUP SPARK at $WORK_DIR/$SPARK_ROOT"
    cd $WORK_DIR && curl $SPARK_DOWNLOAD_URL -o $WORK_DIR/$SPARK_BINARY  && tar -xvf $SPARK_BINARY
    if [ $? != 0 ]; then
      echo "ERROR: Unable to download Spark from $SPARK_DOWNLOAD_URL." \
            "Please make sure you have working internet and you have" \
            "curl(https://en.wikipedia.org/wiki/CURL) on your machine." \
            "Alternatively, you can also manually download Spark archive"\
            "and place it at $SPARK_DOWNLOAD_URL and extract Spark"\
            "to $WORK_DIR/$SPARK_ROOT"
      exit 1
    fi
    echo "SPARK SETUP DONE at $WORK_DIR/$SPARK_ROOT"
  fi
}

function start_spark() {
  echo "Starting Spark at $WORK_DIR/$SPARK_ROOT"
  cd $WORK_DIR/$SPARK_ROOT
  ./sbin/stop-all.sh
  # Authenticate to avoid CVE-2018-17190
  # https://spark.apache.org/security.html
  echo -e "spark.authenticate true\nspark.authenticate.secret $SPARK_SECRET" > $SPARK_CONF
  # default web UI port (8080) is also used by Airflow
  ./sbin/start-master.sh -h $SPARK_HOST -p $SPARK_PORT --webui-port 8081 --properties-file $SPARK_CONF
  ./sbin/start-slave.sh $SPARK_MASTER_URL --properties-file $SPARK_CONF
  echo "Spark running from $WORK_DIR/$SPARK_ROOT"
}


function start_job_server() {
  echo "Starting Beam Spark jobserver"
  cd $BEAM_DIR
  ./gradlew :runners:spark:job-server:runShadow \
      -PsparkMasterUrl=$SPARK_MASTER_URL \
      -Dspark.authenticate=true \
      -Dspark.authenticate.secret=$SPARK_SECRET
}

function main(){
  check_java
  # Check and create the relevant directory
  if [ ! -d "$WORK_DIR" ]; then
    install_beam
  else
    echo "Work directory $WORK_DIR already exists."
    echo "Please delete $WORK_DIR in case of issue."
    update_beam
  fi
  setup_spark
  start_spark
  start_job_server
}

main $@
