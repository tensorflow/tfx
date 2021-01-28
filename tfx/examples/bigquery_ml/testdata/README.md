This is an instruction file for generating the test data for the sample BigQuery
ML publisher pipeline.

TODO(b/145624746): Converge TFX test data once optional dense is supported.
Note: TFT's feature spec generation currently does not support parsing spec for
"optional dense" input (i.e. dense tensor inputs with missing values). Any input
with missing values is interpreted as a sparse tensor. This does not work for
BigQuery as it only supports dense input for model serving. for additional
details please refer to
https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-tensorflow#inputs
Here we fill in the missing values before TFX pipeline starts as a result.
Corresponding changes in taxi_utils_bqml.py file result in the need for a new
test data set.

The test data in this hierarchy is an exact replica of output directory of
the chicago taxi pipeline that results from executing
bigquery_ml/taxi_pipeline_kubeflow_gcp_bqml.py with a few minor modifications as
follows.

1. To generate a new set of test data set the number of rows / samples returned
   from BigQuery by modifying the Query in
   bigquery_ml/taxi_pipeline_kubeflow_gcp_bqml.py to limit the output to 1000.
   sample query is as follows.

```sql
 SELECT
   IFNULL(pickup_community_area, -1) as pickup_community_area,
   IFNULL(fare, -1) as fare,
   EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
   EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
   EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
   UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
   IFNULL(pickup_latitude, -1) as pickup_latitude,
   IFNULL(pickup_longitude, -1) as pickup_longitude,
   IFNULL(dropoff_latitude, -1) as dropoff_latitude,
   IFNULL(dropoff_longitude, -1) as dropoff_longitude,
   IFNULL(trip_miles, -1) as trip_miles,
   IFNULL(pickup_census_tract, -1) as pickup_census_tract,
   IFNULL(dropoff_census_tract, -1) as dropoff_census_tract,
   IFNULL(payment_type, '') as payment_type,
   IFNULL(company, '') as company,
   IFNULL(trip_seconds, -1) as trip_seconds,
   IFNULL(dropoff_community_area, -1) as dropoff_community_area,
   IFNULL(tips,0) as tips
 FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
 where (ABS(FARM_FINGERPRINT(unique_key)) / 0x7FFFFFFFFFFFFFFF) < 0.00001
 LIMIT 1000
```
* where 0x7FFFFFFFFFFFFFFF represents Max_INT 64 and 0.00001 is the sample rate.

2. To ensure consistency we will generate a CSV input.Run the query in BigQuery
   and save the results as a CSV file under testdata/csv
3. Change the pipeline to accept a CSV input instead of BigQuery query by using
   the sample below, where data_root contains the CSV file generated in step 2.

```python
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=data_root)
```

4. Run the pipeline and copy all generated output from GCS to testdata. Output
   path is defined in taxi_pipeline_kubeflow_gcp_bqml.py as follows.

```python
  _output_bucket = 'gs://my-bucket'
  _tfx_root = os.path.join(_output_bucket, 'tfx')
```

5. Remove all *.gstmp and events.out.tfevents.*.cmle-training-* files
6. Add the following to the last line of .../testdata/schema_gen/schema.pbtxt
   generate_legacy_feature_spec: false

This completes the test data generation step.
