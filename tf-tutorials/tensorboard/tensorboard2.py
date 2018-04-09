import tensorflow as tf

"""tensorboard can be used to retrive any information like images in input an inside tensor, also" 
weight, shape, input image etc. write every piece of debug information in tensorflow summary

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('train_accuracy', accuracy)

with tf.name_scope('Cost'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y))
    opt = tf.train.AdamOptimizer()
    optimizer = opt.minimize(cross_entropy)
    grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])
    tf.summary.scalar('cost', cross_entropy)
    tf.summary.scalar('train_cost', cross_entropy)


with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs/mnistlogs/1f', sess.graph)
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge([cost, accuracy])

"""


g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, name="X")

    with tf.name_scope("Layer11"):
        W1 = tf.placeholder(tf.float32, name="W1")
        b1 = tf.placeholder(tf.float32, name="b1")

        a1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    with tf.name_scope("Layer2"):
        W2 = tf.placeholder(tf.float32, name="W2")
        b2 = tf.placeholder(tf.float32, name="b2")

        a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

    with tf.name_scope("Layer3"):
        W3 = tf.placeholder(tf.float32, name="W3")
        b3 = tf.placeholder(tf.float32, name="b3")

        y_hat = tf.matmul(a2, W3) + b3

    for i in range(100):
        tf.summary.scalar("loss",i)
    merged = tf.summary.merge_all()
    sess=tf.Session()
    sess.run(merged)
    tw=tf.summary.FileWriter("logdir",sess.graph).close()
