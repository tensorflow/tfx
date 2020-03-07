# The StatisticsGen TFX Pipeline Component

The StatisticsGen TFX pipeline component generates features statistics
over both training and serving data, which can be used by other pipeline
components.
StatisticsGen uses Beam to scale to large datasets.

* Consumes: datasets created by an ExampleGen pipeline component.
* Emits: Dataset statistics.

## StatisticsGen and TensorFlow Data Validation

StatisticsGen makes extensive use of [TensorFlow Data Validation](tfdv.md) for
generating statistics from your dataset.

## Using the StatsGen Component

A StatisticsGen pipeline component is typically very easy to deploy and
requires little
customization. Typical code looks like this:

```python
from tfx import components

...

compute_eval_stats = components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```

## Using the StatsGen Component With a Schema

For the first run of a pipeline, the output of StatisticsGen will be used to
infer a schema. However, on subsequent runs you may have a manually curated
schema that contains additional information about your data set. By providing
this schema to StatisticsGen, TFDV can provide more useful statistics based on
declared properties of your data set.

In this setting, you will invoke StatisticsGen with a curated schema that has
been imported by an ImporterNode like this:

```python
from tfx import components
from tfx.types import standard_artifacts

...

user_schema_importer = components.ImporterNode(
    instance_name='import_user_schema',
    source_uri=user_schema_path,
    artifact_type=standard_artifcats.Schema)

compute_eval_stats = components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      schema=user_schema_importer.outputs['result'],
      name='compute-eval-stats'
      )
```
