import tensorflow as tf

from os.path import join


flags = tf.app.flags

flags.DEFINE_string('tfrecord_dir', './tfrecords', '')

FLAGS = flags.FLAGS


tfrecord = 'tt0780571.sf-021027.ef-024026.video.tfrecord'


def check_tfrecord():
    it = tf.python_io.tf_record_iterator(join(FLAGS.tfrecord_dir, tfrecord))
    serialized_example = next(it)
    context_features = {
        "number": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frame_feats": tf.FixedLenSequenceFeature([1536], dtype=tf.float32)
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    with tf.Session() as sess:
        c, s = sess.run([context_parsed, sequence_parsed])
    print(c['number'])
    print(s['frame_feats'].shape)
    print(s['frame_feats'])

def test():
    writer = tf.python_io.TFRecordWriter('./test.tfrecord')



def main(_):
    test()

if __name__ == '__main__':
    tf.app.run()
