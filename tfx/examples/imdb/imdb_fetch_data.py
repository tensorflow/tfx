import os
import pandas as pd
import tensorflow_datasets as tfds


# Example use in another file of this directory:
# import imdb_fetch_data as full_data
# full_data.fetch_data()


def fetch_data():
  """This downloads the full dataset to pwd/data/imdb.csv"""

  ds = tfds.load('imdb_reviews', split='train+test')
  numpy_ds = tfds.as_numpy(ds)
  df = pd.DataFrame(numpy_ds)
  df['text'] = df['text'].str.decode("utf-8")
  dst_path = os.getcwd() + '/data/imdb.csv'
  df.to_csv(dst_path, index=False)

if __name__ == '__main__':
  fetch_data()
