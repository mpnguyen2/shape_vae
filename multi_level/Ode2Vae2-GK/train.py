import os
import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from models.ode2vae_tfv2 import ODE2VAE2D
from tensorflow.keras.utils import plot_model

'''
This is the training script for the TF 2.x model. Since Tensorflow no longer has built-in argument parsing, we need to
use Google absl-py to parse the inputs.
'''
FLAGS = flags.FLAGS

flags.DEFINE_string("data_root", 'data', 'root of the data folder')
flags.DEFINE_string("ckpt_dir", 'checkpoints', 'checkpoints folder')
flags.DEFINE_string("task", 'mocap_many', 'experiment to execute')
flags.DEFINE_integer("f_opt", 2, 'neural network(1), Bayesian neural network(2)')
flags.DEFINE_integer("amort_len", 3, 'the number data points (from time zero) velocity encoder takes as input')
flags.DEFINE_string("activation_fn", 'relu',
                    'activation function used in fully connected layers ("relu","tanh","identity")')
flags.DEFINE_integer("q", 10, 'latent dimensionality')
flags.DEFINE_float("gamma", 1.0, 'constant in front of variational loss penalty (sec. 3.3)')
flags.DEFINE_integer("inst_enc_KL", 1, '(1) if use variational loss penalty (sec. 3.3); (0) otherwise')
flags.DEFINE_integer("Hf", 100, 'number of hidden units in each layer of differential NN')
flags.DEFINE_integer("He", 50, 'number of hidden units in each layer of encoder')
flags.DEFINE_integer("Hd", 50, 'number of hidden units in each layer of decoder')
flags.DEFINE_integer("Nf", 2, 'number of hidden layers in differential NN')
flags.DEFINE_integer("Ne", 2, 'number of hidden layers in encoder')
flags.DEFINE_integer("Nd", 2, 'number of hidden layers in  decoder')
flags.DEFINE_integer("NF_enc", 16, 'number of filters in the first encoder layer')
flags.DEFINE_integer("NF_dec", 32, 'number of filters in the last decoder layer')
flags.DEFINE_integer("KW_enc", 5, 'encoder kernel width')
flags.DEFINE_integer("KW_dec", 5, 'decoder kernel width')
flags.DEFINE_float("eta", 0.001, 'learning rate')
flags.DEFINE_integer("batch_size", 10, 'number of sequences in each training mini-batch')
flags.DEFINE_integer("num_epoch", 1000, 'number of training epochs')
flags.DEFINE_integer("subject_id", 0, 'subject ID in mocap_single experiments')


def main(_):
    logging.info(FLAGS.__flags)

    if not os.path.exists(FLAGS.ckpt_dir):
        os.makedirs(FLAGS.ckpt_dir)

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    # model = ODE2VAE2D(FLAGS.test_render, FLAGS.ignore_checkpoint, FLAGS.manual, FLAGS.save,
    #                   FLAGS.learning_rate, FLAGS.beta1, FLAGS.beta2, FLAGS.discount_rate, FLAGS.epochs,
    #                   FLAGS.max_steps, FLAGS.games_per_update, FLAGS.save_iterations, FLAGS.test_games,
    #                   FLAGS.checkpoint_dir)

    plot_model(model, to_file='model.png')

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
        )


if __name__ == '__main__':
    app.run(main)
