import os, math, time

import numpy as np
import tensorflow as tf
from scipy.io import savemat

from absl import app, flags, logging

from models.data.wrappers import *
from models.data.utils import plot_latent

from models.ode2vae_tfv1 import ODE2VAE

sess = tf.compat.v1.InteractiveSession()
tf.compat.v1.disable_eager_execution()

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

flags.register_validator(
    'activation_fn',
    lambda val: val == 'relu' or val == 'tanh' or val == 'identity',
    message='--activation_fn must be either relu, tanh, or identity.')


def main(argv):
    del argv  # unused variable

    ########### setup params, data, etc ###########
    # read params
    flags_dict = FLAGS.flag_values_dict()
    for key, value in flags_dict.items():
        print(f'{key}: {value}')

    data_root = FLAGS.data_root
    ckpt_dir = FLAGS.ckpt_dir
    task = FLAGS.task
    f_opt = FLAGS.f_opt  #unused, always set to 2
    amort_len = FLAGS.amort_len
    activation_fn = FLAGS.activation_fn
    q = FLAGS.q
    gamma = FLAGS.gamma
    inst_enc_KL = FLAGS.inst_enc_KL
    Hf = FLAGS.Hf
    He = FLAGS.He
    Hd = FLAGS.Hd
    Nf  = FLAGS.Nf
    Ne = FLAGS.Ne
    Nd = FLAGS.Nd
    NF_enc = FLAGS.NF_enc
    NF_dec = FLAGS.NF_dec
    KW_enc = FLAGS.KW_enc
    KW_dec = FLAGS.KW_dec
    eta = FLAGS.eta
    batch_size = FLAGS.batch_size
    num_epoch = FLAGS.num_epoch
    subject_id = FLAGS.subject_id

    if activation_fn == 'relu':
        activation_fn = tf.nn.relu
    elif activation_fn == 'tanh':
        activation_fn = tf.nn.tanh
    else:
        activation_fn = tf.identity

    if not os.path.exists(os.path.join(ckpt_dir, task)):
        os.makedirs(os.path.join(ckpt_dir, task))
    if not os.path.exists(os.path.join('plots', task)):
        os.makedirs(os.path.join('plots', task))
    # dataset
    dataset, N, T, D = load_data(data_root, task, subject_id=subject_id, plot=True)

    # artificial time points
    dt = 0.1
    t = dt * np.arange(0, T, dtype=np.float32)
    # file extensions
    if task == 'bballs' or 'mnist' in task:
        ext = '{:s}_q{:d}_inst{:d}_fopt{:d}_enc{:d}_dec{:d}'.format(task, q, inst_enc_KL, f_opt, NF_enc, NF_dec)
    elif 'mocap' in task:
        ext = '{:s}_q{:d}_inst{:d}_fopt{:d}_He{:d}_Hf{:d}_Hd{:d}'.format(task, q, inst_enc_KL, f_opt, He, Hf, Hd)
    print('file extensions are {:s}'.format(ext))

    ########### training related stuff ###########
    xval_batch_size = int(batch_size / 2)
    min_val_lhood = -1e15

    xbspl = tf.compat.v1.placeholder(tf.int64, name="tr_batch_size")
    xfpl = tf.compat.v1.placeholder(tf.float32, [None, None, D], name="tr_features")
    xtpl = tf.compat.v1.placeholder(tf.float32, [None, None], name="tr_timepoints")

    def data_map(X, y, W=T, p=0, dt=dt):
        W += tf.random.uniform([1], 0, 1, tf.int32)[0]  # needed for t to be of dim. None
        W = tf.cast(W, tf.int32)
        rng_ = tf.range(0, W)
        t_ = tf.cast(dt, dtype=tf.float32) * tf.cast(rng_, tf.float32)
        X = tf.gather(X, rng_, axis=1)
        y = tf.gather(y, rng_, axis=1)
        return X, y, t_

    xtr_dataset = tf.data.Dataset.from_tensor_slices((xfpl, xtpl)).batch(xbspl).map(data_map, 8).prefetch(2)
    xval_dataset = tf.data.Dataset.from_tensor_slices((xfpl, xtpl)).batch(xbspl).map(data_map, 8).repeat()

    xiter_ = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(xtr_dataset),
                                                       tf.compat.v1.data.get_output_shapes(xtr_dataset))
    X, _, t = xiter_.get_next()

    xtr_init_op = xiter_.make_initializer(xtr_dataset)
    xval_init_op = xiter_.make_initializer(xval_dataset)

    ########### model ###########
    vae = ODE2VAE(q, D, X, t, NF_enc=NF_enc, NF_dec=NF_dec, KW_enc=KW_enc, KW_dec=KW_dec, Nf=Nf, Ne=Ne,
                  Nd=Nd, task=task, eta=eta, L=1, Hf=Hf, He=He, Hd=Hd, activation_fn=activation_fn,
                  inst_enc_KL=inst_enc_KL, amort_len=amort_len, gamma=gamma)

    ########### training loop ###########
    t0 = time.time()

    print('{:>15s}'.format("epoch") + '{:>15s}'.format("total_cost") + '{:>15s}'.format("E[p(x|z)]") + '{:>15s}'.format(
        "E[p(z)]") + '{:>15s}'.format("E[q(z)]") + \
          '{:>16s}'.format("E[KL[ode||enc]]") + '{:>15s}'.format("valid_cost") + '{:>15s}'.format("valid_error"))
    print('{:>15s}'.format("should") + '{:>15s}'.format("decrease") + '{:>15s}'.format("increase") + '{:>15s}'.format(
        "increase") + '{:>15s}'.format("decrease") + \
          '{:>16s}'.format("decrease") + '{:>15s}'.format("decrease") + '{:>15s}'.format("decrease"))

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = vae(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    for epoch in range(num_epoch):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))

    for epoch in range(num_epoch):
        num_iter = 0
        Tss = max(min(T, T // 5 + epoch // 2), vae.amort_len + 1)
        sess.run(xtr_init_op, feed_dict={xfpl: dataset.train.x, xtpl: dataset.train.y, xbspl: batch_size})
        xtr_dataset = xtr_dataset.shuffle(buffer_size=dataset.train.N)
        sess.run(xval_init_op, feed_dict={xfpl: dataset.val.x, xtpl: dataset.val.y, xbspl: xval_batch_size})
        val_lhood = 0
        num_val_iter = 10
        for _ in range(num_val_iter):
            pass  #something with val_lhood += mean_reconstr_lhood
        val_lhood = val_lhood / (num_val_iter * Tss)
        xval_dataset = xval_dataset.shuffle(buffer_size=dataset.val.N)

        if val_lhood > min_val_lhood:
            min_val_lhood = val_lhood
            vae.save_model(ckpt_dir, ext)
            X, ttr = dataset.train.next_batch(5)
            Xrec = vae.reconstruct(X, ttr)
            zt = vae.integrate(X, ttr)
            plot_reconstructions(task, X, Xrec, ttr, show=False, fname='plots/{:s}/rec_tr_{:s}.png'.format(task, ext))
            plot_latent(zt, vae.q, vae.L, show=False, fname='plots/{:s}/latent_tr_{:s}.png'.format(task, ext))
            X, tval = dataset.val.next_batch(5)
            Xrec = vae.reconstruct(X, tval)
            # zt   = vae.integrate(X)
            plot_reconstructions(task, X, Xrec, tval, show=False, fname='plots/{:s}/rec_val_{:s}.png'.format(task, ext))
            # plot_latent(zt,vae.q,vae.L,show=False,fname='plots/{:s}/latent_val_{:s}.png'.format(task,ext))
            val_err = -1
            if 'mnist' in task:
                X1 = X[:, amort_len:, :]
                X2 = Xrec[:, amort_len:, :]
                val_err = np.mean((X1 - X2) ** 2)
            elif task == 'bballs':
                X1 = X[:, amort_len:amort_len + 10, :]
                X2 = Xrec[:, amort_len:amort_len + 10, :]
                val_err = np.sum((X1 - X2) ** 2, 2)
                val_err = np.mean(val_err)
            elif task == 'mocap_single':
                diff = X[0, :, :] - Xrec[0, :, :]
                diff = diff[4 * diff.shape[0] // 5:, :] ** 2
                val_err = np.mean(diff)
            elif task == 'mocap_many':
                val_err = np.mean((X - Xrec) ** 2)

            print(f'cost: {val_err}')

    t1 = time.time()
    print('elapsed time: {:.2f}'.format(t1 - t0))

if __name__ == '__main__':
    app.run(main)