# TFX/TFJS next page prediction example

This example demonstrates:

  * How Apache Beam can be used to convert Google
    Analytics events into data used for training (see
    `bigquery_beam_data_generation.py`).

<!--TODO(b/187088244): Update field names to be more descriptive. -->

The TFX pipeline expects gzipped TFRecords containing TF Examples. Each example
should contain the following fields:

  * cur_page: A string field containing the current page.
  * session_index: An integer field specifying where in the session the cur_page
    was visited.
  * label: A string field containing the next page.

The data folder in this directory contains synthetic training data containing
100 examples to help you get started.
