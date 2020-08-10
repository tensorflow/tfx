# Squad Bert Single Sentence Classification Example

*   The data is downloaded from tensorflow_dataset.
*   The example datasets in /data are the first 100 entries of the original datasets.
*   To fetch the entire dataset, please use the snippet below
*   And please adjust the corresponding hyperparameters to account for the
    larger dataset.

```python
# Remove the train_small.tfrecord and validation_small.tfrecord in
# /data/train and /data/validation that comes natively in the /data
# folder.
# Run this snippet to download the squad dataset from tfds.
# Change the hyperparameters to better suit the bigger dataset.
# The configurations that were found reasonable are listed below:
# bert_squad_pipeline_native_keras.py:
#    tfma.GenericValueThreshold(lower_bound={'value':0.80}
#    trainer_pb2.TrainArgs(num_steps=1600)
#    trainer_pb2.EvalArgs(num_steps=72)
# bert_squad_utils_native_keras.py:
#    _TRAIN_BATCH_SIZE=16
#    _EVAL_BATCH_SIZE=16

import os
import pandas as pd
import tensorflow_datasets as tfds

if __name__ == '__main__':
  train_path = os.getcwd() + '/data/train/train.tfrecord'
  validation_path = os.getcwd() + '/data/validation/validation.tfrecord'
  tfds.load('squad', data_dir=os.getcwd(), split=['train', 'validation'])
  file_path = os.path.join(os.getcwd(), 'squad', 'plain_text', '1.0.0')
  os.rename(os.path.join(file_path, 'squad-train.tfrecord-00000-of-00001'), train_path)
  os.rename(os.path.join(file_path, 'squad-validation.tfrecord-00000-of-00001'), validation_path)
```

# Acknowledge Data Source

```
@article{2016arXiv160605250R,
       author = { {Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
```