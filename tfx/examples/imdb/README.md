# IMDB Sentiment Analaysis Example

*   The data is downloaded from tensorflow_dataset.
*   The example dataset imdb_small_with_labels.csv that comes within /data is
    the first 100 entries of the original dataset.
*   To fetch the entire dataset, please use the snippet below
*   And please adjust the corresponding hyperparameters to account for the
    larger dataset.

```python
# Run this file to download and preprocess then entire imdb dataset.
# Remove the imdb_small_with_labels.csv that comes natively in the repo/data
# folder. Make sure imdb.csv is present in the /data folder.
# Change the hyperparameters to better suit the bigger dataset.
# The configurations that were found reasonable are listed below:
# imdb_pipeline_native_keras.py:
#    tfma.GenericValueThreshold(lower_bound={'value':0.85}
#    trainer_pb2.TrainArgs(num_steps=7000)
#    trainer_pb2.EvalArgs(num_steps=800)
# imdb_utils_native_keras.py:
#    _TRAIN_BATCH_SIZE=64
#    _EVAL_BATCH_SIZE=64

import os import pandas as pd
import tensorflow_datasets as tfds

if __name__ == '__main__':
  ds = tfds.load('imdb_reviews', split='train+test')
  numpy_ds = tfds.as_numpy(ds)
  df = pd.DataFrame(numpy_ds)
  df['text'] = df['text'].str.decode("utf-8")
  dst_path = os.getcwd() + '/data/imdb.csv'
  df.to_csv(dst_path, index=False)
```

# Acknowledge Data Source

```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```
