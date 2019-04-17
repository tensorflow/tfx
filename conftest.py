import sys

collect_ignore = []
if sys.version_info.major == 2:
    collect_ignore.append('tfx/examples/chicago_taxi_pipeline/taxi_pipeline_kubeflow_test.py')
    collect_ignore.append('tfx/orchestration/kubeflow')