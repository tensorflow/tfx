# Chicago Taxi Dataset

This dataset was generated in the same way that it was generated in the
[TFX Chicago Taxi example][0].

To regenerate the dataset (including the derived TFRecords dataset,
intermediate outputs for the benchmarks, and trained models), use the following
steps:

1. Generate a CSV dataset. See the [TFX Chicago Taxi example][0] for details
as to how to do so. In particular, [`taxi_pipeline_kubeflow_gcp.py`][1] has the
BigQuery query used to generate the dataset. You can run this query on BigQuery
and export the results to a CSV file.

2. Run `regenerate_datasets.py` as follows, changing the path to the CSV dataset
as appropriate:

```
python regenerate_datasets.py -- --dataset=chicago_taxi --generate_dataset_args=/path/to/dataset.csv
```

This will regenerate the derived TFRecords dataset, as well as all intermediate
outputs required for running the benchmarks.

[0]: https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/README.md
[1]: https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_kubeflow_gcp.py
