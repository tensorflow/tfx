""" Utility script to add dataset split info to the dataset """

import tensorflow as tf

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def add_split_info_to_dataset(filenames, split):
  """ Add split information as an extra feature to the dataset """
  dataset = tf.data.TFRecordDataset(filenames)
  examples = []
  for record in dataset:
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    new_feature = {
        'is_train': _int64_feature(split == 'train')
    }
    for key in example.features.feature:
      new_feature[key] = example.features.feature[key]
    example_proto = tf.train.Example(features=tf.train.\
                        Features(feature=new_feature))
    example_proto_serialized = example_proto.SerializeToString()
    examples.append(example_proto_serialized)

  # Write the `tf.Example` observations to the file.
  out_path = 'data/{0}_tiny/cifar10_with_split_info_{0}.tfrecord'.format(split)
  with tf.io.TFRecordWriter(out_path) as writer:
    for example in examples:
      writer.write(example)

add_split_info_to_dataset(
    ['data/cifar10/3.0.2/cifar10-train.tfrecord-00000-of-00001'],
    'train')
add_split_info_to_dataset(
    ['data/cifar10/3.0.2/cifar10-test.tfrecord-00000-of-00001'],
    'test')
