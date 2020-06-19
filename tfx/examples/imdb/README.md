# IMDB Sentiment Analaysis Example

* The data is downloaded from tensorflow_dataset.
* The example dataset imdb_small.csv that comes within /data is the first 100 entries of the original dataset.
* To fetch the entire dataset, please consult `imdb_dataset_utils.py`
* And please adjust the corresponding hyperparameters to account for the larger dataset.

```
# Run this file to download and preprocess then entire imdb dataset.
# Remove the imdb_small.csv that comes natively in the repo/data folder.
# Make sure imdb.csv is present in the /data folder.
# Change the hyperparameters to better suit the bigger dataset.
# The configurations that were found reasonable are listed below:
#     imdb_pipeline_native_kears.py:
#         tfma.GenericValueThreshold(lower_bound={'value':0.8}
#         trainer_pb2.TrainArgs(num_steps=5120)
#         trainer_pb2.EvalArgs(num_steps=2560)
#     imdb_utils_native_keras.py:
#         _TRAIN_BATCH_SIZE=64
#         _EVAL_BATCH_SIZE=64

import os
import pandas as pd
import tensorflow_datasets as tfds

if __name__ == '__main__':
  ds = tfds.load('imdb_reviews', split='train+test')
  numpy_ds = tfds.as_numpy(ds)
  df = pd.DataFrame(numpy_ds)
  df['text'] = df['text'].str.decode("utf-8")
  dst_path = os.getcwd() + '/data/imdb.csv'
  df.to_csv(dst_path, index=False)
```
