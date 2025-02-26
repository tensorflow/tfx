# Community-developed components, examples, and tools for TFX

Developers helping developers. TFX-Addons is a collection of community
projects to build new components, examples, libraries, and tools for TFX.
The projects are organized under the auspices of the special interest group,
SIG TFX-Addons.

[Join the community and share your work with the world!](http://goo.gle/tfx-addons-group)

---

TFX-Addons is available on PyPI for all OS. To install the latest version, run:

```shell
pip install tfx-addons
```

You can then use TFX-Addons like this:

```python
from tfx import v1 as tfx
import tfx_addons as tfxa

# Then you can easily load projects tfxa.{project_name}. For example:
tfxa.feast_examplegen.FeastExampleGen(...)
```

<div class="grid cards" markdown>

-   [__Feast ExampleGen Component__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/feast_examplegen)

	---

	An [ExampleGen](./examplegen.md) component for ingesting datasets from a [Feast Feature Store](https://feast.dev/).

	[:octicons-arrow-right-24: Feast ExampleGen](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/feast_examplegen)

-   [__Feature Selection Component__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/feature_selection)

	---

	Perform feature selection using various algorithms with this TFX component.

	[:octicons-arrow-right-24: Feature Selection](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/feature_selection)

-   [__Firebase Publisher Component__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/firebase_publisher)

	---

	A TFX component to publish/update ML models to [Firebase ML](https://firebase.google.com/products/ml).

	[:octicons-arrow-right-24: Firebase Publisher](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/firebase_publisher)

-   [__Hugging Face Pusher Component__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/huggingface_pusher)

	---

	[Hugging Face Model Hub](https://huggingface.co/models). Optionally pushes the application to the [Hugging Face Spaces Hub](https://huggingface.co/spaces).

	[:octicons-arrow-right-24: Hugging Face Pusher](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/huggingface_pusher)

-   [__Message Exit Handler Component__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/message_exit_handler)

	---

	Handle the completion or failure of a pipeline by notifying users, including any error messages.

	[:octicons-arrow-right-24: Message Exit Handler](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/message_exit_handler)

-   [__MLMD Client Library__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/mlmd_client)

	---

	Client library to inspect content in [ML Metadata](mlmd.md) populated by TFX pipelines.

	[:octicons-arrow-right-24: MLMD Client](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/mlmd_client)

-   [__Model Card Generator__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/model_card_generator)

	---

	The ModelCardGenerator takes [dataset statistics](statsgen.md), [model evaluation](evaluator.md), and a [pushed model](pusher.md) to automatically populate parts of a model card.

	[:octicons-arrow-right-24: Model Card Generator](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/model_card_generator)

-   [__Pandas Transform Component__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/pandas_transform)

	---

	Use [Pandas dataframes](https://pandas.pydata.org/) instead of the standard Transform component for your feature engineering. Processing is distributed using [Apache Beam](https://beam.apache.org/) for scalability.

	[:octicons-arrow-right-24: Pandas Transform](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/pandas_transform)

-   [__Sampling Component__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/sampling)

	---

	A TFX component to sample data from examples, using probabilistic estimation.

	[:octicons-arrow-right-24: Sampling](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/sampling)

-   [__Schema Curation Component__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/schema_curation)

	---

	Apply user code to a schema produced by the [SchemaGen component](schemagen.md), and curate it based on domain knowledge.

	[:octicons-arrow-right-24: Schema Curation](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/schema_curation)

-   [__XGBoost Evaluator Component__](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/xgboost_evaluator)

	---

	Evaluate [XGBoost](https://xgboost.ai/) models by extending the standard [Evaluator component](evaluator.md).

	[:octicons-arrow-right-24: XGBoost Evaluator](https://github.com/tensorflow/tfx-addons/tree/main/tfx_addons/xgboost_evaluator)

</div>
