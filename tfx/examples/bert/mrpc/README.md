# MRPC Bert Sentence Pair Classification Example

*   The data is downloaded from tensorflow_dataset.
*   The example datasets in /data are the first 100 entries of the original
    datasets.
*   To fetch the entire dataset, please use the snippet below
*   And please adjust the corresponding hyperparameters to account for the
    larger dataset.

```python
# Remove the train_small.csv and validation_small.csv in
# /data/train and /data/validation that comes natively in the /data
# folder.
# Run this snippet to download the MRPC dataset from tfds.
# Change the hyperparameters to better suit the bigger dataset.
# The configurations that were found reasonable are listed below:
# bert_mrpc_pipeline_native_keras.py:
#    tfma.GenericValueThreshold(lower_bound={'value':0.80}
#    trainer_pb2.TrainArgs(num_steps=350)
#    trainer_pb2.EvalArgs(num_steps=38)
# bert_mrpc_utils_native_keras.py:
#    _TRAIN_BATCH_SIZE=32
#    _EVAL_BATCH_SIZE=32

import os
import pandas as pd
import tensorflow_datasets as tfds

if __name__ == '__main__':
  train_path = os.getcwd() + '/data/train/train.csv'
  validation_path = os.getcwd() + '/data/validation/validation.csv'

  ds = tfds.load('glue/mrpc', split='train')
  numpy_ds = tfds.as_numpy(ds)
  df = pd.DataFrame(numpy_ds)
  df['sentence1'] = df['sentence1'].str.decode("utf-8")
  df['sentence2'] = df['sentence2'].str.decode("utf-8")
  df.to_csv(train_path, index=False)

  ds = tfds.load('glue/mrpc', split='validation')
  numpy_ds = tfds.as_numpy(ds)
  df = pd.DataFrame(numpy_ds)
  df['sentence1'] = df['sentence1'].str.decode("utf-8")
  df['sentence2'] = df['sentence2'].str.decode("utf-8")
  df.to_csv(validation_path, index=False)
```

# Acknowledge Data Source

```
@inproceedings{dolan2005automatically,
  title={Automatically constructing a corpus of sentential paraphrases},
  author={Dolan, William B and Brockett, Chris},
  booktitle={Proceedings of the Third International Workshop on Paraphrasing (IWP2005)},
  year={2005}
}
@inproceedings{wang2019glue,
  title={ {GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
  note={In the Proceedings of ICLR.},
  year={2019}
}
```
