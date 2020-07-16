import tensorflow as tf
import tensorflow_datasets as tfds

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def add_split_info_to_dataset(filenames, split):
    filenames = ['data/train_whole/cifar10-train.tfrecord-00000-of-00001']
    dataset = tf.data.TFRecordDataset(filenames)

    examples = []
    for record in dataset:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        keys = example.features.feature
        new_feature = {
            'is_train': _int64_feature(split=='train')
        }
        for key in example.features.feature:
            new_feature[key] = example.features.feature[key]
        example_proto = tf.train.Example(features=tf.train.Features(feature=new_feature))
        example_proto_serialized = example_proto.SerializeToString()
        examples.append(example_proto_serialized)

    # Write the `tf.Example` observations to the file.
    with tf.io.TFRecordWriter('data/{0}_with_split_whole/cifar10_with_split_info_{0}.tfrecord'.format(split)) as writer:
        for example in examples:
            writer.write(example)

add_split_info_to_dataset(['data/train_whole/cifar10-train.tfrecord-00000-of-00001'], 'train')
add_split_info_to_dataset(['data/test_whole/cifar10-test.tfrecord-00000-of-00001'], 'test')

# filenames = ['data/train_with_split/cifar10_tiny_with_split_info_train.tfrecord']
# dataset = tf.data.TFRecordDataset(filenames)
# for record in dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(record.numpy())
#     print(example)

# filenames = ['data/test_with_split/cifar10_tiny_with_split_info_test.tfrecord']
# dataset = tf.data.TFRecordDataset(filenames)
# for record in dataset.take(1):
#     example = tf.train.Example()
#     example.ParseFromString(record.numpy())
#     print(example)