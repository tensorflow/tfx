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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import subprocess
import time
import yaml
from absl import flags

from datasets.chicago_taxi import dataset
from tfma_benchmark_chicago_taxi import TFMABenchmarkChicagoTaxi
from tfma_v2_benchmark_chicago_taxi import TFMAV2BenchmarkChicagoTaxi
from tft_benchmark_chicago_taxi import TFTBenchmarkChicagoTaxi
import benchmark_base

class BeamPipelineBenchmarkBase():

    def __init__(self, min_num_workers, max_num_workers, base_dir,
                 temp_location_for_cloud_dataflow=""):
        self.wall_times = {}
        self.wall_times["tfma"] = {}
        self.wall_times["tfma_v2"] = {}
        self.wall_times["tft"] = {}

        self.dataset = []

        self.min_num_workers = min_num_workers
        self.max_num_workers = max_num_workers

        self.yaml_path = "beam_flink_cluster.yaml"
        self.dataset = dataset.get_dataset(base_dir=dataset_dir)

        benchmark_base.set_temp_location_for_cloud_dataflow = temp_location_for_cloud_dataflow

    def _run_tfma_benchmarks(self):
        tfma_benchmark_chicago_taxi = TFMABenchmarkChicagoTaxi(dataset=self.dataset)
        self.wall_times["tfma"]["benchmarkMiniPipeline"] = tfma_benchmark_chicago_taxi.benchmarkMiniPipeline()

    def _run_tfma_v2_benchmarks(self):
        tfma_v2_benchmark_chicago_taxi = TFMAV2BenchmarkChicagoTaxi(dataset=self.dataset)
        self.wall_times["tfma_v2"]["benchmarkMiniPipelineUnbatched"] = tfma_v2_benchmark_chicago_taxi.benchmarkMiniPipelineUnbatched()
        self.wall_times["tfma_v2"]["benchmarkMiniPipelineBatched"] = tfma_v2_benchmark_chicago_taxi.benchmarkMiniPipelineBatched()

    def _run_tft_benchmarks(self):
        tft_benchmark_chicago_taxi = TFTBenchmarkChicagoTaxi(dataset=self.dataset)
        self.wall_times["tft"]["benchmarkAnalyzeAndTransformDataset"] = tft_benchmark_chicago_taxi.benchmarkAnalyzeAndTransformDataset()

    def _run_all_benchmarks(self):
        self.run_tfma_benchmarks()
        self.run_tfma_v2_benchmarks()
        self.run_tft_benchmarks()

    def _post_process(self):

        # Add test names if dataset is empty
        if (self.dataset == []):
            test_names = ["Number of Replicas"]
            for tf_module in self.wall_times:
                for test in self.wall_times[tf_module]:
                    test_names.append(test)

            self.dataset.append(test_names)

        # Add wall times to dataset
        row = [self.num_replicas]
        for tf_module in self.wall_times:
            for test, wall_time in self.wall_times[tf_module].items():
                row.append(wall_time)

        self.dataset.append(row)
        print(self.dataset)

    def _update_yaml(num_replicas):

        with open(self.yaml_path) as f:
            yaml_file = yaml.load(f)

        yaml_file["spec"]["taskManager"]["replicas"] = num_replicas

        with open(self.yaml_path, "w") as f:
            yaml.dump(yaml_file, f)

    def _write_to_csv(self, csv_filename):
        with open(csv_filename, "w+") as  my_csv:
            csv_writer = csv.writer(my_csv,delimiter=',')
            csv_writer.writerows(self.dataset)

    def benchmarkFlinkOnK8s():

        benchmark_base.set_beam_pipeline_mode("flink_on_k8s")
        num_replicas = self.min_num_replicas

        while(num_replicas <= self.max_num_replicas):
            # Update the .yaml file replicas value
            self._update_yaml(num_replicas)
            time.sleep(10)

            # Delete and reapply the kubectl clusters
            subprocess.call("kubectl delete -f beam_flink_cluster.yaml", shell = True)
            subprocess.call("kubectl delete -f beam_job_server.yaml", shell = True)
            time.sleep(180)

            subprocess.call("kubectl apply -f beam_flink_cluster.yaml", shell = True)
            subprocess.call("kubectl apply -f beam_job_server.yaml", shell = True)
            time.sleep(180)

            # Set up port forwarding
            subprocess.call("pkill kubectl -9", shell = True)
            subprocess.Popen("kubectl port-forward service/beam-job-server 8098:8098", shell = True)
            subprocess.Popen("kubectl port-forward service/beam-job-server 8099:8099", shell = True)
            time.sleep(30)

            # Run the benchmarks
            benchmark_base.set_num_workers(num_replicas)
            self._run_all_benchmarks()
            self._post_process()

            # Write to csv
            self._write_to_csv(csv_filename="beam_pipeline_benchmark_results_flink_on_k8s.csv")

            if (num_replicas < 4):
                num_replicas += 1
            else:
                num_replicas *= 2

        # Cleanup kubectl processes
        subprocess.call("kubectl delete -f beam_flink_cluster.yaml", shell = True)
        subprocess.call("kubectl delete -f beam_job_server.yaml", shell = True)
        subprocess.call("pkill kubectl -9", shell = True)

    def benchmarkLocalScaled():

        benchmark_base.set_beam_pipeline_mode("local_scaled_execution")
        num_workers = self.min_num_workers

        while(num_workers <= self.max_num_workers):
            # Run the benchmarks
            benchmark_base.set_num_workers(num_workers)
            self._run_all_benchmarks()

            # Write to csv
            self._post_process()
            self._write_to_csv(csv_filename="beam_pipeline_benchmark_results_local.csv")

            if (num_workers < 4):
                num_workers += 1
            else:
                num_workers *= 2

    def benchmarkCloudDataflow():

        benchmark_base.set_beam_pipeline_mode("cloud_dataflow")
        num_workers = self.min_num_workers

        while(num_workers <= self.max_num_workers):

            # Run the benchmarks
            benchmark_base.set_num_workers(num_workers)
            self._run_all_benchmarks()

            # Write to csv
            self._post_process()
            self._write_to_csv(csv_filename="beam_pipeline_benchmark_results_cloud_dataflow.csv")

            if (num_workers < 4):
                num_workers += 1
            else:
                num_workers *= 2
