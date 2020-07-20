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
"""Base class for running Apache Beam pipeline based benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx.benchmarks import benchmark_base
from tfx.benchmarks.datasets.chicago_taxi import dataset
from tfx.benchmarks.tfma_benchmark_chicago_taxi import TFMABenchmarkChicagoTaxi
from tfx.benchmarks.tfma_v2_benchmark_chicago_taxi import TFMAV2BenchmarkChicagoTaxi
from tfx.benchmarks.tft_benchmark_chicago_taxi import TFTBenchmarkChicagoTaxi
from tfx.benchmarks.big_shuffle_benchmark import BigShuffleBenchmarkBase

import subprocess
import tempfile
import time
import yaml
import csv
import os

TFMA = "tfma"
TFMA_V2 = "tfma_v2"
TFT = "tft"
BIG_SHUFFLE = "big_shuffle"

TFMA_BENCHMARK_MINI_PIPELINE = "TFMABenchmarkChicagoTaxi.benchmarkMiniPipeline"
TFMA_V2_BENCHMARK_MINI_PIPELINE_UNBATCHED = "TFMAV2BenchmarkChicagoTaxi.benchmarkMiniPipelineUnbatched"
TFMA_V2_BENCHMARK_MINI_PIPELINE_BATCHED = "TFMAV2BenchmarkChicagoTaxi.benchmarkMiniPipelineBatched"
TFT_BENCHMARK_ANALYZE_AND_TRANSFORM_DATASET = "TFTBenchmarkChicagoTaxi.benchmarkAnalyzeAndTransformDataset"
BIG_SHUFFLE_BENCHMARK = "BigShuffleBenchmarkBase.benchmarkBigShuffle with file size: "
BIG_SHUFFLE_EMPTY_PIPELINE_BENCHMARK = "BigShuffleBenchmarkBase.benchmarkEmptyPipeline"

class BeamPipelineBenchmarkBase():
  """Beam Pipeline benchmarks base class."""

  def __init__(self, min_num_workers, max_num_workers, base_dir,
               cloud_dataflow_temp_loc, big_shuffle_input_file,
               big_shuffle_output_file):

    self.wall_times = {}
    self.wall_times[TFMA] = {}
    self.wall_times[TFMA_V2] = {}
    self.wall_times[TFT] = {}
    self.wall_times[BIG_SHUFFLE] = {}

    self.wall_times_list = []

    self.min_num_workers = min_num_workers
    self.max_num_workers = max_num_workers
    self.num_workers = self.min_num_workers

    self.yaml_tf = tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w+t", delete=False)
    self.yaml_path = self.yaml_tf.name # "tmp_flink_on_k8s_cluster.yaml"

    with open("flink_on_k8s_cluster.yaml") as f2:
      with open(self.yaml_path, "w+t") as f1:
        for line in f2:
          f1.write(line)

    self.dataset = dataset.get_dataset(base_dir=base_dir)

    benchmark_base.set_cloud_dataflow_temp_loc(cloud_dataflow_temp_loc)
    self.min_file_size = 1e6 # 1 MB
    self.max_file_size = 1e9 # 1 GB
    self.big_shuffle_input_file = big_shuffle_input_file
    self.big_shuffle_output_file = big_shuffle_output_file

  def _run_tfma_benchmarks(self):
    tfma_benchmark_chicago_taxi = TFMABenchmarkChicagoTaxi(dataset=self.dataset)
    self.wall_times[TFMA][TFMA_BENCHMARK_MINI_PIPELINE] = tfma_benchmark_chicago_taxi.benchmarkMiniPipeline()

  def _run_tfma_v2_benchmarks(self):
    tfma_v2_benchmark_chicago_taxi = TFMAV2BenchmarkChicagoTaxi(dataset=self.dataset)
    self.wall_times[TFMA_V2][TFMA_V2_BENCHMARK_MINI_PIPELINE_UNBATCHED] = tfma_v2_benchmark_chicago_taxi.benchmarkMiniPipelineUnbatched()
    self.wall_times[TFMA_V2][TFMA_V2_BENCHMARK_MINI_PIPELINE_BATCHED] = tfma_v2_benchmark_chicago_taxi.benchmarkMiniPipelineBatched()

  def _run_tft_benchmarks(self):
    tft_benchmark_chicago_taxi = TFTBenchmarkChicagoTaxi(dataset=self.dataset)
    self.wall_times[TFT][TFT_BENCHMARK_ANALYZE_AND_TRANSFORM_DATASET] = tft_benchmark_chicago_taxi.benchmarkAnalyzeAndTransformDataset()

  def _run_big_shuffle_benchmark(self):
    big_shuffle_benchmark = BigShuffleBenchmarkBase(
        input_file=self.big_shuffle_input_file,
        output_file=self.big_shuffle_output_file)

    file_size = self.min_file_size
    while file_size <= self.max_file_size:
      big_shuffle_benchmark.regenerate_data(file_size)
      self.wall_times[BIG_SHUFFLE][BIG_SHUFFLE_BENCHMARK + str(file_size)] = big_shuffle_benchmark.benchmarkBigShuffle()
      file_size *= 10

    self.wall_times[BIG_SHUFFLE][BIG_SHUFFLE_EMPTY_PIPELINE_BENCHMARK] = big_shuffle_benchmark.benchmarkEmptyPipeline()

  def _run_all_benchmarks(self):
    self._run_tfma_benchmarks()
    self._run_tfma_v2_benchmarks()
    self._run_tft_benchmarks()
    self._run_big_shuffle_benchmark()

  def _post_process(self):
    # Add test names if dataset is empty
    if self.wall_times_list == []:
      test_names = ["Number of Replicas"]
      for base_file in self.wall_times:
        for test in self.wall_times[base_file]:
          test_names.append(test)

      self.wall_times_list.append(test_names)

    # Add wall times to dataset
    row = [self.num_workers]
    for tf_module in self.wall_times:
      for test, wall_time in self.wall_times[tf_module].items():
        row.append(wall_time)

    self.wall_times_list.append(row)

  def _update_yaml(self, num_replicas):
    with open(self.yaml_path) as f:
      yaml_file = yaml.load(f, Loader=yaml.FullLoader)
      yaml_file["spec"]["taskManager"]["replicas"] = num_replicas
      yaml.dump(yaml_file, self.yaml_tf)

  def _write_to_csv(self, csv_filename):
    with open(csv_filename, "w+") as  my_csv:
      csv_writer = csv.writer(my_csv, delimiter=',')
      csv_writer.writerows(self.wall_times_list)

  def _increment_num_workers(self):
    if self.num_workers < 4:
      self.num_workers += 1
    else:
      self.num_workers *= 2

  def benchmarkFlinkOnK8s(self):
    """Utilizes the flink-on-k8s-operator to run Beam pipelines"""

    benchmark_base.set_beam_pipeline_mode("flink_on_k8s")
    self.num_workers = self.min_num_workers

    # Delete any leftover kubectl clusters
    subprocess.call("kubectl delete -f " + self.yaml_path, shell=True)
    subprocess.call("kubectl delete -f flink_on_k8s_job_server.yaml",
                    shell=True)
    time.sleep(180)

    while self.num_workers <= self.max_num_workers:

      # Update the .yaml file replicas value
      self._update_yaml(self.num_workers)
      time.sleep(10)

      # Apply the kubectl clusters
      subprocess.call("kubectl apply -f " + self.yaml_path, shell=True)
      subprocess.call("kubectl apply -f flink_on_k8s_job_server.yaml",
                      shell=True)
      time.sleep(180)

      # Set up port forwarding
      subprocess.call("pkill kubectl -9", shell=True)
      subprocess.Popen(
          "kubectl port-forward service/flink-on-k8s-job-server 8098:8098",
          shell=True)
      subprocess.Popen(
          "kubectl port-forward service/flink-on-k8s-job-server 8099:8099",
          shell=True)
      time.sleep(20)

      # Run the benchmarks
      benchmark_base.set_num_workers(self.num_workers)
      self._run_all_benchmarks()
      self._post_process()

      # Write to csv
      self._write_to_csv(csv_filename=
                         "beam_pipeline_benchmark_results_flink_on_k8s.csv")

      self._increment_num_workers()

      # Cleanup kubectl processes
      subprocess.call("kubectl delete -f " + self.yaml_path, shell=True)
      subprocess.call("kubectl delete -f flink_on_k8s_job_server.yaml",
                      shell=True)
      subprocess.call("pkill kubectl -9", shell=True)
      time.sleep(180)

    self.yaml_tf.close()
    os.remove(self.yaml_path)

  def benchmarkLocalScaled(self):
    """Utilizes the local machine to run Beam pipelines"""

    benchmark_base.set_beam_pipeline_mode("local_scaled_execution")
    self.num_workers = self.min_num_workers

    while self.num_workers <= self.max_num_workers:
      # Run the benchmarks
      benchmark_base.set_num_workers(self.num_workers)
      self._run_all_benchmarks()

      # Write to csv
      self._post_process()
      self._write_to_csv(csv_filename=
                         "beam_pipeline_benchmark_results_local.csv")

      self._increment_num_workers()

  def benchmarkCloudDataflow(self):
    """Utilizes Cloud Dataflow to run Beam pipelines"""

    benchmark_base.set_beam_pipeline_mode("cloud_dataflow")
    self.num_workers = self.min_num_workers

    while self.num_workers <= self.max_num_workers:

      # Run the benchmarks
      benchmark_base.set_num_workers(self.num_workers)
      self._run_all_benchmarks()

      # Write to csv
      self._post_process()
      self._write_to_csv(csv_filename=
                         "beam_pipeline_benchmark_results_cloud_dataflow.csv")

      self._increment_num_workers()
