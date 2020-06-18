import pandas as pd
import tensorflow_datasets as tfds
import os
"""
Run this file to download and preprocess then entire imdb dataset.
Remove the imdb_small.csv that comes natively in the repo/data folder.
Make sure imdb.csv is present in the /data folder.
Change the hyperparameters to better suit the bigger dataset.
The configurations that were found reasonable are listed below:
    imdb_pipeline_native_kears.py:
        tfma.GenericValueThreshold(lower_bound={'value':0.8}
        trainer_pb2.TrainArgs(num_steps=512)
        trainer_pb2.EvalArgs(num_steps=256)
       
    imdb_utils_native_keras.py:
        _TRAIN_BATCH_SIZE=64
        _EVAL_BATCH_SIZE=64
        _TRAIN_EPOCHS=10 
"""
if __name__ == '__main__':
    ds = tfds.load('imdb_reviews', split='train+test')
    numpy_ds = tfds.as_numpy(ds)
    df = pd.DataFrame(numpy_ds)
    df['text'] = df['text'].str.decode("utf-8")
    path = os.getcwd()
    df.to_csv(path + '/data/imdb.csv', index=False)
