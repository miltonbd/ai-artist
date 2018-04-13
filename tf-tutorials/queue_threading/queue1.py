# use tensorflow tf.Print to show the values flowing through the graph

import tensorflow as tf

tf.set_random_seed(123)


with tf.variable_scope("my"): # also tf.name_scope() works for constant, variable
    dummy_input = tf.random_normal([3])
    dummy_input = tf.Print(dummy_input,[dummy_input],"Dummy input")
    q = tf.FIFOQueue(3,tf.float32)
    enqueue_op = q.enqueue_many(dummy_input)
    threads = 10
    qr=tf.train.QueueRunner(q,[enqueue_op] * threads)
    tf.train.add_queue_runner(qr)
    data = q.dequeue()
    data = tf.Print(data,[q.size(), data],"data")
    fg=data+1


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(fg)
    sess.run(fg)
    sess.run(fg)
    sess.run(fg)

    # we have to request all threads now stop, then we can join the queue runner
    # thread back to the main thread and finish up
    coord.request_stop()
    coord.join(threads)
