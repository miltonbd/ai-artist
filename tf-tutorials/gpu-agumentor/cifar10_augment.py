import tensorflow as tf

class Cifar10Augment(object):

    def __init__(self):
        pass


    def augment(self):
        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True))
        sess.run(init)

        pass



if __name__ == '__main__':
    cifar10 = Cifar10Augment()
    cifar10.augment()