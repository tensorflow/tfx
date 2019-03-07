# The ExampleGen TFX Pipeline Component

The ExampleGen TFX Pipeline component ingests
data into TFX pipelines. It consumes external files/services to generate
Examples which will be read by other TFX components.

* Consumes: Data from external data sources such as CSV and BigQuery
* Emits: tf.Example records

## ExampleGen and Other Components

ExampleGen provides data to components that make use of the
[TensorFlow Data Validation](tfdv.md) library, such as [SchemaGen](schemagen.md),
[StatisticsGen](statsgen.md), and [Example Validator](exampleval.md).  It also
provides data to [Transform](transform.md), which makes use of the
[TensorFlow Transform](tft.md) library, and ultimately to deployment targets
during inference.

## Developing an ExampleGen Component

For supported data sources (currently, CSV files and results of BigQuery
queries) the ExampleGen pipeline component is typically very easy to deploy and
requires little customization. Typical code looks like this:

```python
from tfx.utils.dsl_utils import csv_input
from tfx.components import ExamplesGen

examples = csv_input(os.path.join(base_dir, 'no_split/span_1'))
examples_gen = ExamplesGen(input_data=examples)
```
