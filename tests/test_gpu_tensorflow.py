def test_tensorflow_gpu():
    # Test that tensorflow supports GPUs
    import tensorflow as tf
    with tf.device('gpu:0'):
        a = tf.constant(10)
        b = tf.constant(32)
        sess = tf.Session()
        sess.run(a+b)
