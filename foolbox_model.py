import foolbox
import os
from model import Model
import tensorflow as tf

from foolbox import zoo


def create():
    tf.compat.v1.enable_eager_execution()

    with tf.compat.v1.get_default_graph().as_default():
        weights_path = zoo.fetch_weights(
            'https://github.com/MadryLab/mnist_challenge_models/raw/master/secret.zip',
            unzip=True
        )
        weights_path = os.path.join(weights_path, 'models/secret')

        model = Model()

        sess = tf.Session().__enter__()
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(weights_path)
        saver.restore(sess, checkpoint)

    fmodel = foolbox.models.TensorFlowModel(model, bounds=(0, 1))

    return fmodel
