# The ExampleGen TFX Pipeline Component

The ExampleGen TFX Pipeline component is an API for getting tf.Example
records into TFX pipelines. It consumes external files/services to generate
Examples which will be read by other TFX components.

* Consumes: Data from external data sources such as CSV and BigQuery
* Emits: tf.Example records

## ExampleGen and Other Components

ExampleGen provides data to components that make use of the
[TensorFlow Data Validation](tfdv.md) library, such as [SchemaGen](schemagen.md),
[StatsGen](statsgen.md), and [Example Validator](exampleval.md).  It also
provides data to [Transform](transform.md), which makes use of the
[TensorFlow Transform](tft.md) library, and ultimately to deployment targets
during inference.

## Developing an ExampleGen Component

For supported data sources (currently CSV only) the ExampleGen pipeline
component is typically very easy to develop and requires little customization,
since all of the work is done by the ExampleGen TFX component. Typical code looks
like this:

```python
from tfx.utils.dsl_utils import csv_inputs
from tfx.components import ExamplesGen

examples = csv_inputs(os.path.join(base_dir, 'no_split/span_1'))
examples_gen = ExamplesGen(input_data=examples)
```
