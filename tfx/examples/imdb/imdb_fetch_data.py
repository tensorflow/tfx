import os
import pandas as pd
import tensorflow_datasets as tfds


# Example use in another file of this directory:
# import imdb_fetch_data as full_data
# full_data.fetch_data()


"""
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
"""


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
