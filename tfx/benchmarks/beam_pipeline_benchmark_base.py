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

from tfx.benchmarks import mode_config
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

_TFMA = "tfma"
_TFMA_V2 = "tfma_v2"
_TFT = "tft"
_BIG_SHUFFLE = "big_shuffle"

_TFMA_BENCHMARK_MINI_PIPELINE = "TFMABenchmarkChicagoTaxi.benchmarkMiniPipeline"
_TFMA_V2_BENCHMARK_MINI_PIPELINE_UNBATCHED = "TFMAV2BenchmarkChicagoTaxi.benchmarkMiniPipelineUnbatched"
_TFMA_V2_BENCHMARK_MINI_PIPELINE_BATCHED = "TFMAV2BenchmarkChicagoTaxi.benchmarkMiniPipelineBatched"
_TFT_BENCHMARK_ANALYZE_AND_TRANSFORM_DATASET = "TFTBenchmarkChicagoTaxi.benchmarkAnalyzeAndTransformDataset"
_BIG_SHUFFLE_BENCHMARK = "BigShuffleBenchmarkBase.benchmarkBigShuffle with file size: "
_BIG_SHUFFLE_EMPTY_PIPELINE_BENCHMARK = "BigShuffleBenchmarkBase.benchmarkEmptyPipeline"


class BeamPipelineBenchmarkBase(object):
  """Beam Pipeline benchmarks base class."""

  def __init__(self, min_num_workers, max_num_workers, base_dir,
               big_shuffle_input_file, big_shuffle_output_file):

    self._wall_times = {}
    self._wall_times[_TFMA] = {}
    self._wall_times[_TFMA_V2] = {}
    self._wall_times[_TFT] = {}
    self._wall_times[_BIG_SHUFFLE] = {}

    self._wall_times_list = []

    self.min_num_workers = min_num_workers
    self.max_num_workers = max_num_workers

    self._dataset = dataset.get_dataset(base_dir=base_dir)

    self._big_shuffle_input_file = big_shuffle_input_file
    self._big_shuffle_output_file = big_shuffle_output_file

    self.cloud_dataflow_project = None
    self.cloud_dataflow_temp_loc = None

  def _set_benchmark_class_parameters(self, benchmark_class, num_workers,
                                      beam_pipeline_mode):
    benchmark_class.set_num_workers(num_workers)
    benchmark_class.set_beam_pipeline_mode(beam_pipeline_mode)
    if beam_pipeline_mode == mode_config.CLOUD_DATAFLOW_MODE:
      assert  self.cloud_dataflow_project
      assert self.cloud_dataflow_temp_loc
      benchmark_class.set_cloud_dataflow_temp_loc(self.cloud_dataflow_temp_loc)
      benchmark_class.set_cloud_dataflow_project(self.cloud_dataflow_project)

  def _run_tfma_benchmarks(self, num_workers, beam_pipeline_mode):
    tfma_benchmark_chicago_taxi = TFMABenchmarkChicagoTaxi(
        dataset=self._dataset)
    self._set_benchmark_class_parameters(tfma_benchmark_chicago_taxi,
                                         num_workers, beam_pipeline_mode)

    self._wall_times[_TFMA][_TFMA_BENCHMARK_MINI_PIPELINE] = tfma_benchmark_chicago_taxi.benchmarkMiniPipeline()

  def _run_tfma_v2_benchmarks(self, num_workers, beam_pipeline_mode):
    tfma_v2_benchmark_chicago_taxi = TFMAV2BenchmarkChicagoTaxi(dataset=self._dataset)
    self._set_benchmark_class_parameters(tfma_v2_benchmark_chicago_taxi,
                                         num_workers, beam_pipeline_mode)

    self._wall_times[_TFMA_V2][_TFMA_V2_BENCHMARK_MINI_PIPELINE_UNBATCHED] = tfma_v2_benchmark_chicago_taxi.benchmarkMiniPipelineUnbatched()
    self._wall_times[_TFMA_V2][_TFMA_V2_BENCHMARK_MINI_PIPELINE_BATCHED] = tfma_v2_benchmark_chicago_taxi.benchmarkMiniPipelineBatched()

  def _run_tft_benchmarks(self, num_workers, beam_pipeline_mode):
    tft_benchmark_chicago_taxi = TFTBenchmarkChicagoTaxi(dataset=self._dataset)
    self._set_benchmark_class_parameters(tft_benchmark_chicago_taxi,
                                         num_workers, beam_pipeline_mode)

    self._wall_times[_TFT][_TFT_BENCHMARK_ANALYZE_AND_TRANSFORM_DATASET] = tft_benchmark_chicago_taxi.benchmarkAnalyzeAndTransformDataset()

  def _run_big_shuffle_benchmarks(self, num_workers, beam_pipeline_mode):
    big_shuffle_benchmark = BigShuffleBenchmarkBase(
        input_file=self._big_shuffle_input_file,
        output_file=self._big_shuffle_output_file)
    self._set_benchmark_class_parameters(big_shuffle_benchmark,
                                         num_workers, beam_pipeline_mode)

    min_file_size = 1e6 # 1 MB
    max_file_size = 1e9 # 1 GB
    file_size = min_file_size

    while file_size <= max_file_size:
      big_shuffle_benchmark.regenerate_data(file_size)
      self._wall_times[_BIG_SHUFFLE][_BIG_SHUFFLE_BENCHMARK + str(file_size)] = big_shuffle_benchmark.benchmarkBigShuffle()
      file_size *= 10

    self._wall_times[_BIG_SHUFFLE][_BIG_SHUFFLE_EMPTY_PIPELINE_BENCHMARK] = big_shuffle_benchmark.benchmarkEmptyPipeline()

  def _run_all_benchmarks(self, num_workers, beam_pipeline_mode):
    self._run_tfma_benchmarks(num_workers, beam_pipeline_mode)
    self._run_tfma_v2_benchmarks(num_workers, beam_pipeline_mode)
    self._run_tft_benchmarks(num_workers, beam_pipeline_mode)
    self._run_big_shuffle_benchmarks(num_workers, beam_pipeline_mode)

  def _post_process(self, num_workers):
    # Add test names if dataset is empty
    if self._wall_times_list == []:
      test_names = ["Number of Replicas"]
      for base_file in self._wall_times:
        for test in self._wall_times[base_file]:
          test_names.append(test)

      self._wall_times_list.append(test_names)

    # Add wall times to dataset
    row = [num_workers]
    for tf_module in self._wall_times:
      for test, wall_time in self._wall_times[tf_module].items():
        row.append(wall_time)

    self._wall_times_list.append(row)

  def _generate_temp_yaml(self):
    yaml_tf = tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w+t", delete=False)
    yaml_path = yaml_tf.name

    with open("flink_on_k8s_cluster.yaml") as f1:
      with open(yaml_path, "w+t") as f2:
        for line in f1:
          f2.write(line)

    return yaml_path, yaml_tf

  def _update_yaml(self, yaml_path, yaml_tf, num_replicas):
    with open(yaml_path) as f:
      yaml_file = yaml.load(f, Loader=yaml.FullLoader)
      yaml_file["spec"]["taskManager"]["replicas"] = num_replicas
      yaml.dump(yaml_file, yaml_tf)

  def _write_to_csv(self, csv_filename):
    with open(csv_filename, "w+") as  my_csv:
      csv_writer = csv.writer(my_csv, delimiter=',')
      csv_writer.writerows(self._wall_times_list)

  def _increment_num_workers(self, num_workers):
    if num_workers < 4:
      num_workers += 1
    else:
      num_workers *= 2

    return num_workers

  def benchmarkFlinkOnK8s(self):
    """Utilizes the flink-on-k8s-operator to run Beam pipelines"""

    beam_pipeline_mode = mode_config.FLINK_ON_K8S_MODE
    num_workers = self.min_num_workers
    yaml_path, yaml_tf = self._generate_temp_yaml()

    # Delete any leftover kubectl clusters
    subprocess.call("kubectl delete -f " + yaml_path, shell=True)
    subprocess.call("kubectl delete -f flink_on_k8s_job_server.yaml",
                    shell=True)
    time.sleep(180)

    while num_workers <= self.max_num_workers:
      # Update the .yaml file replicas value
      self._update_yaml(yaml_path, yaml_tf, num_workers)
      time.sleep(10)

      # Apply the kubectl clusters
      subprocess.call("kubectl apply -f " + yaml_path, shell=True)
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
      self._run_all_benchmarks(num_workers, beam_pipeline_mode)
      self._post_process(num_workers)

      # Write to csv
      self._write_to_csv(csv_filename=
                         "beam_pipeline_benchmark_results_flink_on_k8s.csv")

      num_workers = self._increment_num_workers(num_workers)

      # Cleanup kubectl processes
      subprocess.call("kubectl delete -f " + yaml_path, shell=True)
      subprocess.call("kubectl delete -f flink_on_k8s_job_server.yaml",
                      shell=True)
      subprocess.call("pkill kubectl -9", shell=True)
      time.sleep(180)

    yaml_tf.close()
    os.remove(yaml_path)

  def benchmarkLocalScaled(self):
    """Utilizes the local machine to run Beam pipelines"""

    beam_pipeline_mode = mode_config.LOCAL_SCALED_EXECUTION_MODE
    num_workers = self.min_num_workers

    while num_workers <= self.max_num_workers:
      # Run the benchmarks
      self._run_all_benchmarks(num_workers, beam_pipeline_mode)

      # Write to csv
      self._post_process(num_workers)
      self._write_to_csv(csv_filename=
                         "beam_pipeline_benchmark_results_local.csv")

      num_workers = self._increment_num_workers(num_workers)

  def benchmarkCloudDataflow(self, cloud_dataflow_project,
                             cloud_dataflow_temp_loc):
    """Utilizes Cloud Dataflow to run Beam pipelines"""

    self.cloud_dataflow_project = cloud_dataflow_project
    self.cloud_dataflow_temp_loc = cloud_dataflow_temp_loc

    beam_pipeline_mode = mode_config.CLOUD_DATAFLOW_MODE
    num_workers = self.min_num_workers

    while num_workers <= self.max_num_workers:
      # Run the benchmarks
      self._run_all_benchmarks(num_workers, beam_pipeline_mode)

      # Write to csv
      self._post_process(num_workers)
      self._write_to_csv(csv_filename=
                         "beam_pipeline_benchmark_results_cloud_dataflow.csv")

      num_workers = self._increment_num_workers(num_workers)
