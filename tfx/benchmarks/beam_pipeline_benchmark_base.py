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

import subprocess
import time
import yaml
import csv

class BeamPipelineBenchmarkBase():
  """Beam Pipeline benchmarks base class."""

  def __init__(self, min_num_workers, max_num_workers, base_dir,
               cloud_dataflow_temp_loc=""):

    self.wall_times = {}
    self.wall_times["tfma"] = {}
    self.wall_times["tfma_v2"] = {}
    self.wall_times["tft"] = {}

    self.wall_times_list = []

    self.min_num_workers = min_num_workers
    self.max_num_workers = max_num_workers
    self.num_workers = self.min_num_workers

    self.yaml_path = "beam_flink_cluster.yaml"
    self.dataset = dataset.get_dataset(base_dir=base_dir)

    self.tfma_test_names = []
    self.tfma_v2_test_names = []
    self.tft_test_names = []

    benchmark_base.set_cloud_dataflow_temp_loc(cloud_dataflow_temp_loc)

  def _run_tfma_benchmarks(self):
    tfma_benchmark_chicago_taxi = TFMABenchmarkChicagoTaxi(dataset=self.dataset)
    self.wall_times["tfma"]["TFMABenchmarkChicagoTaxi.benchmarkMiniPipeline"] = tfma_benchmark_chicago_taxi.benchmarkMiniPipeline()

  def _run_tfma_v2_benchmarks(self):
    tfma_v2_benchmark_chicago_taxi = TFMAV2BenchmarkChicagoTaxi(dataset=self.dataset)
    self.wall_times["tfma_v2"]["TFMAV2BenchmarkChicagoTaxi.benchmarkMiniPipelineUnbatched"] = tfma_v2_benchmark_chicago_taxi.benchmarkMiniPipelineUnbatched()
    self.wall_times["tfma_v2"]["TFMAV2BenchmarkChicagoTaxi.benchmarkMiniPipelineBatched"] = tfma_v2_benchmark_chicago_taxi.benchmarkMiniPipelineBatched()

  def _run_tft_benchmarks(self):
    tft_benchmark_chicago_taxi = TFTBenchmarkChicagoTaxi(dataset=self.dataset)
    self.wall_times["tft"]["TFTBenchmarkChicagoTaxi.benchmarkAnalyzeAndTransformDataset"] = tft_benchmark_chicago_taxi.benchmarkAnalyzeAndTransformDataset()

  def _run_all_benchmarks(self):
    self._run_tfma_benchmarks()
    self._run_tfma_v2_benchmarks()
    self._run_tft_benchmarks()

  def _post_process(self):

    # Add test names if dataset is empty
    if self.wall_times_list == []:
      test_names = ["Number of Replicas"]
      for tf_module in self.wall_times:
        for test in self.wall_times[tf_module]:
          test_names.append(test)

      self.wall_times_list.append(test_names)

    # Add wall times to dataset
    row = [self.num_workers]
    for tf_module in self.wall_times:
      for test, wall_time in self.wall_times[tf_module].items():
        row.append(wall_time)

    self.wall_times_list.append(row)
    print(self.wall_times_list)

  def _update_yaml(self, num_replicas):
    with open(self.yaml_path) as f:
      yaml_file = yaml.load(f)

      yaml_file["spec"]["taskManager"]["replicas"] = num_replicas

      with open(self.yaml_path, "w") as f:
        yaml.dump(yaml_file, f)

  def _write_to_csv(self, csv_filename):
    with open(csv_filename, "w+") as  my_csv:
      csv_writer = csv.writer(my_csv, delimiter=',')
      csv_writer.writerows(self.wall_times_list)

  def benchmarkFlinkOnK8s(self):
    """Utilizes the flink-on-k8s-operator to run Beam pipelines"""

    benchmark_base.set_beam_pipeline_mode("flink_on_k8s")
    self.num_workers = self.min_num_replicas

    while self.num_workers <= self.max_num_replicas:

      # Update the .yaml file replicas value
      self._update_yaml(self.num_workers)
      time.sleep(10)

      # Delete and reapply the kubectl clusters
      subprocess.call("kubectl delete -f beam_flink_cluster.yaml", shell=True)
      subprocess.call("kubectl delete -f beam_job_server.yaml", shell=True)
      time.sleep(180)

      subprocess.call("kubectl apply -f beam_flink_cluster.yaml", shell=True)
      subprocess.call("kubectl apply -f beam_job_server.yaml", shell=True)
      time.sleep(180)

      # Set up port forwarding
      subprocess.call("pkill kubectl -9", shell=True)
      subprocess.Popen("kubectl port-forward service/beam-job-server 8098:8098",
                       shell=True)
      subprocess.Popen("kubectl port-forward service/beam-job-server 8099:8099",
                       shell=True)
      time.sleep(30)

      # Run the benchmarks
      benchmark_base.set_num_workers(self.num_workers)
      self._run_all_benchmarks()
      self._post_process()

      # Write to csv
      self._write_to_csv(csv_filename=
                         "beam_pipeline_benchmark_results_flink_on_k8s.csv")

      if self.num_workers < 4:
        self.num_workers += 1
      else:
        self.num_workers *= 2

      # Cleanup kubectl processes
      subprocess.call("kubectl delete -f beam_flink_cluster.yaml", shell=True)
      subprocess.call("kubectl delete -f beam_job_server.yaml", shell=True)
      subprocess.call("pkill kubectl -9", shell=True)

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

      if self.num_workers < 4:
        self.num_workers += 1
      else:
        self.num_workers *= 2

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

      if self.num_workers < 4:
        self.num_workers += 1
      else:
        self.num_workers *= 2
