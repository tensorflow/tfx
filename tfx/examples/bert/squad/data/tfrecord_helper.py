import tensorflow as tf
import sys

def make_small_tfrecord(
    input_file,
    output_dir,
    num_examples):
  ds = tf.data.TFRecordDataset(input_file)
  writer = tf.data.experimental.TFRecordWriter(output_dir)
  writer.write(ds.take(num_examples))

if __name__ == "__main__":
  input_file, output_dir, num_examples = sys.argv[1:4]
  make_small_tfrecord(
      input_file,
      output_dir,
      int(num_examples)
  )
