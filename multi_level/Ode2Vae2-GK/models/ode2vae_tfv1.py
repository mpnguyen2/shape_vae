import warnings

from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, BatchNormalization, Activation, Concatenate, \
    LSTMCell, RNN, LSTM, Bidirectional, Permute, Lambda, Reshape, Conv2DTranspose, RepeatVector
from tensorflow.keras.models import Sequential, Model

import numpy as np
import tensorflow as tf
import os

import tensorflow_probability as tfp

import tensorflow_probability.python.distributions as tfd

from tensorflow.python.framework import ops

warnings.filterwarnings("ignore")


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# tf.compat.v1.disable_resource_variables()

class ODE2VAE(Model):
    """
    The following is our attempt to hack the original ODE2VAE code to get it working in Tensorflow 2.x. This proved to
    be intractible from how low-quality and complicated the source code is. We turned the encoders/decoders into models,
    but it's impossible to use these in conjunction with TF 1.x code. A good estimate is that it would take 20 more
    hours at least to completely upgrade this code.
    """

    def __init__(self, q, D, X, t, task='mocap_many', eta=0.001, L=1,
                 Hf=100, He=20, Hd=20, Nf=2, Ne=2, Nd=2, activation_fn=tf.nn.relu, inst_enc_KL=True,
                 amort_len=10, NF_enc=32, NF_dec=32, KW_enc=5, KW_dec=5, gamma=1.0, stiff=False,
                 Tss=1):
        """
        q          - latent dim
        D          - data dimensionality
        X          - pointer to training data batch
        t          - pointer to time steps
        Hf         - number of hidden neurons used in the differential NN
        He         - number of hidden neurons used in the encoder NN
        Hd         - number of hidden neurons used in the decoder NN
        Nf         - number of hidden layers used in the differential NN
        Ne         - number of hidden layers used in the encoder NN
        Nd         - number of hidden layers used in the decoder NN
        task       - network architecture - 'cmu', 'mnist', 'bballs', 'face', 'h36m', 'dog'
        eta        - learning rate
        L          - number of z0 particles (random draws) per input data point
        amort_len  - the number of data points from time 0 used in encoding
        NF_enc     - number of filters in encoder cnn
        NF_dec     - number of filters in decoder cnn
        KW_enc     - kernel width in encoder cnn
        KW_dec     - kernel width in encoder cnn
        beta       - the constant in front of KL term (beta-VAE term)
        """
        self.q = q
        self.D = D
        self.L = L
        self.Hf = Hf
        self.He = He
        self.Hd = Hd
        self.Nf = Nf
        self.Ne = Ne
        self.Nd = Nd
        self.activation_fn = activation_fn
        self.inst_enc_KL = inst_enc_KL
        self.task = task
        self.eta = eta
        self.amort_len = amort_len
        self.Tss = Tss
        self.NF_enc = NF_enc
        self.NF_dec = NF_dec
        self.KW_enc = KW_enc
        self.KW_dec = KW_dec
        self.gamma = gamma
        print('gamma is {:.6f}'.format(self.gamma))
        self.beta = self.q / (Hf * (3 * q + Nf * Hf + Nf + 1) + q)
        print('beta is {:.6f}'.format(self.beta))
        self.is_enc_rnn = False
        self.x = tf.identity(X, name="X")
        self.t = tf.identity(t, name="t")
        self.qt = tf.constant(self.q, dtype=tf.float32, name='q')  # needed while loading a saved model
        self.Dt = tf.constant(self.D, dtype=tf.float32, name='D')  # needed while loading a saved model
        self.Lt = tf.constant(self.L, dtype=tf.float32, name='L')  # needed while loading a saved model
        self.amort_lent = tf.constant(self.amort_len, dtype=tf.int32,
                                      name='amort_len')  # needed while loading a saved model

        # new code: replace deprecated ODE solver
        dt = t[1] - t[0]
        self.solver = tfp.math.ode.BDF(first_step_size=dt) if stiff \
            else tfp.math.ode.DormandPrince(first_step_size=dt / 5)

        self.create_network()
        self.create_loss()
        self.create_optimizers()

    def set_configs(self):
        # network output types, needed for reconstruction likelihood
        if self.task == 'bballs' or self.task == 'mnist':
            self.dec_out = 'bernoulli'
        else:
            self.dec_out = 'normal'
        print(self.dec_out + ' likelihood')

        # Always use f_opt = 2 for the latent acceleration model (var-NN)
        with tf.compat.v1.variable_scope('f', reuse=tf.compat.v1.AUTO_REUSE):
            self.loglambdasq = tf.Variable(np.log(np.array([1e-4])), dtype=tf.float32, name='S')
            tf.compat.v1.summary.scalar('S', tf.exp(self.loglambdasq[0]))
            Ns = [2 * self.q]
            for tmp in range(self.Nf):
                Ns.append(self.Hf)
            Ns.append(self.q)
            # q,H,H,H,q
            self.f_weight_means = []  # weight matrix means
            self.f_bias_means = []  # bias vector means
            for i in range(0, self.Nf + 1):
                w_ = tf.compat.v1.get_variable("w{:d}".format(i), shape=[Ns[i], Ns[i + 1]],
                                               initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                   scale=1.0, mode="fan_avg", distribution="uniform"))
                self.f_weight_means.append(w_)
                b_ = tf.compat.v1.get_variable("b{:d}".format(i), shape=[Ns[i + 1]],
                                               initializer=tf.compat.v1.zeros_initializer())
                self.f_bias_means.append(b_)
            Ws = []  # TF weight samples
            Bs = []  # TF bias samples
            for i in range(0, self.Nf + 1):
                Wi = tf.random.normal([Ns[i], Ns[i + 1]]) * tf.sqrt(tf.exp(self.loglambdasq)) + self.f_weight_means[
                    i]
                Ws.append(Wi)
                Bi = tf.random.normal([Ns[i + 1]]) * tf.sqrt(tf.exp(self.loglambdasq)) + self.f_bias_means[i]
                Bs.append(Bi)

        def f(X, t=[0]):
            with tf.compat.v1.variable_scope('f', reuse=tf.compat.v1.AUTO_REUSE):
                x = tf.reshape(X, [-1, 2 * self.q])
                for i in range(0, self.Nf + 1):
                    if i < self.Nf:
                        x = self.activation_fn(tf.matmul(x, Ws[i]) + Bs[i])
                    elif i == self.Nf:
                        x = tf.matmul(x, Ws[i]) + Bs[i]
                return x

        def fmean(X, t=[0]):
            with tf.compat.v1.variable_scope('f', reuse=tf.compat.v1.AUTO_REUSE):
                x = tf.reshape(X, [-1, 2 * self.q])
                for i in range(0, self.Nf + 1):
                    if i < self.Nf:
                        x = self.activation_fn(tf.matmul(x, self.f_weight_means[i]) + self.f_bias_means[i])
                    elif i == self.Nf:
                        x = tf.matmul(x, self.f_weight_means[i]) + self.f_bias_means[i]
                return x

        self.f = f
        self.fmean = fmean

        # extract data points v0&s0 are computed conditioned on
        v0 = tf.gather(self.x, [i for i in range(self.amort_len)], axis=1)  # N,amort_len,D
        s0 = tf.squeeze(tf.gather(self.x, [0], axis=1), axis=1)  # N,1,D
        # extract the data points for instant encoding
        x0vs = tf.stack([self.x[:, i:-self.amort_len + i] for i in range(self.amort_len)],
                        0)  # amort_len, N, T-amort_len, D
        x0vs = tf.transpose(a=x0vs, perm=[2, 1, 0, 3])  # T-amort_len, N, amort_len, D
        x0vs = tf.reshape(x0vs, [-1, self.amort_len, self.D])  # (T-amort_len)*N, amort_len, D
        x0ss = self.x[:, :-self.amort_len, :]  # N, T-amort_len, D
        x0ss = tf.transpose(a=x0ss, perm=[1, 0, 2])  # T-amort_len, N, D
        x0ss = tf.reshape(x0ss, [-1, self.D])  # (T-amort_len)*N, D

        # TODO tf.split v_enc_fnc and s_enc_fnc
        v_enc_model = None
        s_enc_model = None
        dec_model = None

        # static/dynamic encoders and decoder
        if 'mocap' in self.task:
            s_enc_model = self.enc_dense(tf.shape(s0)[1:])  # exclude batch portion
            dec_model = self.dec_dense()
            if self.is_enc_rnn:
                v_enc_model = self.enc_rnn(tf.shape(v0)[1:])
            else:
                v0 = tf.reshape(v0, [-1, self.amort_len * self.D])
                x0vs = tf.reshape(x0vs, [-1, self.amort_len * self.D])  # (T-amort_len)*N, amort_len*D
                v_enc_model = self.enc_dense((self.amort_len * self.D,))  # enc_dense, enc_rnn
        elif self.task == 'mnist' or self.task == 'bballs':
            dec_model = self.dec_mnist_bball
            if self.is_enc_rnn:
                # v_enc_model = self.enc_rnn TODO: shape????
                # s_enc_model = self.enc_rnn
                pass
            else:
                # TODO fix shape--should be image shape
                v_enc_model = self.create_conv_encoder(tf.shape(self.x), name='v_enc')
                s_enc_model = self.create_conv_encoder(tf.shape(self.x), name='s_enc')

                def channel_stack(x_T):
                    x, T = x_T
                    dim = int(np.sqrt(self.D))
                    x = tf.reshape(x, shape=[-1, T, dim, dim, 1])
                    x = tf.transpose(a=x, perm=[0, 2, 3, 1, 4])
                    x = tf.reshape(x, [-1, dim, dim, T])
                    return x

                v0 = channel_stack((v0, self.amort_len))
                s0 = channel_stack((s0, 1))
                x0vs = tf.map_fn(channel_stack, (x0vs, tf.tile([self.amort_len], [tf.shape(input=x0vs)[0]])),
                                 (tf.float32))  # T-a_len, N, D, D, a_len
                x0ss = tf.map_fn(channel_stack, (x0ss, tf.tile([1], [tf.shape(input=x0ss)[0]])),
                                 (tf.float32))  # T-a_len, N, D, D, 1
                len_inst_enc = tf.shape(input=x0vs)[0]  # T-a_len
                N = tf.shape(input=x0vs)[1]
                D = int(np.sqrt(self.D))
                x0vs = tf.reshape(x0vs, [len_inst_enc * N, D, D, self.amort_len])  # (T-a_len)*N,D,D,amort_len
                x0ss = tf.reshape(x0ss, [len_inst_enc * N, D, D, 1])  # (T-a_len)*N,D,D,1
        return v0, s0, x0vs, x0ss, v_enc_model, s_enc_model, dec_model

    def create_network(self):
        self.x = self.x[:, :self.Tss, :]
        self.t = self.t[:self.Tss]
        x0_v, x0_s, x0vs, x0ss, v_enc_model, s_enc_model, dec_model = self.set_configs()
        self.v0_output = v_enc_model(tf.shape(x0_v), name='v0_enc')(x0_v)
        self.s0_output = s_enc_model(tf.shape(x0_s), name='s0_enc')(x0_s)
        self.v0_mu, self.v0_log_sigma_sq = tf.split(self.v0_output, num_or_size_splits=2, axis=1)
        self.s0_mu, self.s0_log_sigma_sq = tf.split(self.s0_output, num_or_size_splits=2, axis=1)
        self.v0_log_sigma_sq = tf.identity(self.v0_log_sigma_sq, name='v0_log_sigma_sq')
        self.s0_log_sigma_sq = tf.identity(self.s0_log_sigma_sq, name='s0_log_sigma_sq')
        # ODE
        mvn = tfd.MultivariateNormalDiag(loc=tf.zeros(self.q), scale_diag=tf.ones(self.q))

        def _sample_trajectory(xin):
            eps1 = tf.random.normal(tf.shape(input=self.v0_log_sigma_sq), 0, 1, dtype=tf.float32)
            v0 = tf.add(self.v0_mu, tf.multiply(tf.sqrt(tf.exp(self.v0_log_sigma_sq)), eps1))
            eps2 = tf.random.normal(tf.shape(input=self.s0_log_sigma_sq), 0, 1, dtype=tf.float32)
            s0 = tf.add(self.s0_mu, tf.multiply(tf.sqrt(tf.exp(self.s0_log_sigma_sq)), eps2))
            q0 = mvn.log_prob(eps1) + mvn.log_prob(eps2)
            q0 = tf.expand_dims(q0, 1)  # N,1
            vt, st, logpt = self.ode_density(v0, s0, q0, self.f, self.t)  # T, N, q [t0x0,t0x1,...t0xN,t1x0,t1x1,...]
            return vt, st, logpt

        vt_L, st_L, self.logpt_L = tf.map_fn(_sample_trajectory, tf.range(0, self.L),
                                             (tf.float32, tf.float32, tf.float32))  # L,T,N,q & L,T,N
        self.vt_L = tf.transpose(a=vt_L, perm=[1, 2, 0, 3], name='vt_L')  # T, N, L, q
        self.st_L = tf.transpose(a=st_L, perm=[1, 2, 0, 3], name='st_L')  # T, N, L, q
        self.vt_L = tf.reshape(self.vt_L, [-1, self.q])  # T*N*L, q
        self.st_L = tf.reshape(self.st_L, [-1, self.q])  # T*N*L, q
        self.zt_L = tf.concat([self.vt_L, self.st_L], 1)  # T*N*L, 2q
        self.vt_mu, self.st_mu = self.ode(self.v0_mu, self.s0_mu, self.fmean, self.t)  # T, N, q
        self.st_mu = tf.reshape(self.st_mu, [-1, self.q], name='st_mu')  # T*N, q
        self.vt_mu = tf.reshape(self.vt_mu, [-1, self.q], name='vt_mu')  # T*N, q
        # Reconstruct
        self.x_rec_mu_L, self.x_rec_log_sigma_sq_L = self.decode(self.st_L, dec_model)  # T*N*L,D
        self.x_rec_mu_mu, self.x_rec_log_sigma_sq_mu = self.decode(self.st_mu, dec_model)  # T*N,D
        self.x_rec_mu_L = tf.identity(self.x_rec_mu_L, name='x_rec_mu_L')
        self.x_rec_mu_mu = tf.identity(self.x_rec_mu_mu, name='x_rec_mu_mu')
        if self.inst_enc_KL:
            self.inst_enc(x0vs, x0ss, v_enc_fnc, s_enc_fnc)

    def create_conv_encoder(self, shape, name=None):
        enc_inputs = Input(shape=shape)
        enc_intermediate = Conv2D(self.NF_enc, self.KW_enc, use_bias=False, name='h0')(enc_inputs)
        enc_intermediate = BatchNormalization()(enc_intermediate)
        enc_intermediate = Activation("relu")(enc_intermediate)
        enc_intermediate = Conv2D(2 * self.NF_enc, self.KW_enc, use_bias=False, name='h1')(enc_intermediate)
        enc_intermediate = BatchNormalization()(enc_intermediate)
        enc_intermediate = Activation("relu")(enc_intermediate)
        enc_intermediate = Conv2D(4 * self.NF_enc, self.KW_enc, use_bias=False, name='h2')(enc_intermediate)
        enc_intermediate = BatchNormalization()(enc_intermediate)
        enc_intermediate = Activation("relu")(enc_intermediate)
        enc_intermediate = Conv2D(8 * self.NF_enc, self.KW_enc, use_bias=False, name='h3')(enc_intermediate)
        enc_intermediate = BatchNormalization()(enc_intermediate)
        enc_intermediate = Activation("relu")(enc_intermediate)
        enc_intermediate = Flatten(enc_intermediate)
        mu = Dense(self.q, activation=None)(enc_intermediate)
        log_sig_sq = Dense(self.q, activation=None)(enc_intermediate)
        enc_outputs = Concatenate(axis=1)([mu, log_sig_sq])
        return Model(inputs=enc_inputs, outputs=enc_outputs, name=name)

    def inst_enc(self, x0vs, x0ss, v_enc_fnc, s_enc_fnc):  # (T-a_len)*N, enc_expected_input_shape
        self.vt_mu_enc, self.vt_log_sigma_sq_enc = v_enc_fnc(x0vs, name='enc_v0')  # (T-a_len)*N,q
        self.st_mu_enc, self.st_log_sigma_sq_enc = s_enc_fnc(x0ss, name='enc_s0')  # (T-a_len)*N,q
        zt_mu_enc = tf.concat([self.vt_mu_enc, self.st_mu_enc], 1)  # (T-a_len)*N,2q
        zt_log_sigma_sq_enc = tf.concat([self.vt_log_sigma_sq_enc, self.st_log_sigma_sq_enc], 1)  # (T-a_len)*N,2q
        N = tf.shape(input=self.x)[0]
        zt_mu_enc = tf.reshape(zt_mu_enc, [-1, N, 2 * self.q])  # T-a_len,N,2q
        zt_log_sigma_sq_enc = tf.reshape(zt_log_sigma_sq_enc, [-1, N, 2 * self.q])  # T-a_len,N,2q
        self.zt_L_mu_enc = tf.reshape(tf.tile(zt_mu_enc, [1, 1, self.L]), [-1, 2 * self.q])  # (T-a_len)*N*L,2q
        self.zt_L_log_sigma_sq_enc = tf.reshape(tf.tile(zt_log_sigma_sq_enc, [1, 1, self.L]),
                                                [-1, 2 * self.q])  # (T-a_len)*N*L,2q
        T = tf.shape(input=zt_mu_enc)[0]  # T-a_len
        N = tf.shape(input=zt_mu_enc)[1]
        self.zt_L_ode = self.zt_L[:T * N * self.L, :]  # (T-a_len)*N*L,2q

    def enc_dense(self, shape, name=None):
        inputs = Input(shape=shape)
        x = inputs
        for i in range(self.Ne):
            # x = bn_list[i](tf.contrib.layers.fully_connected(x,  self.He, activation_fn=self.activation_fn),train=self.train)
            x = Dense(self.He, activation=self.activation_fn)(x)
        z0_mu = Dense(self.q, activation=None)(x)
        z0_log_sig_sq = Dense(self.q, activation=None)(x)
        outputs = Concatenate(axis=1)([z0_mu, z0_log_sig_sq])
        return Model(inputs=inputs, outputs=outputs, name=name)

    def enc_rnn(self, shape, name=None):
        # rnn_layers = [tf.nn.rnn_cell.LSTMCell(size, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE) for size in [50, 50]]
        # rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        # rnn_cell = tf.nn.rnn_cell.GRUCell(100, activation=tf.nn.tanh)
        # TODO verify this imitates tf.nn.dynamic_rnn properly
        inputs = Input(shape=shape)
        rnn_cell = LSTM(25, activation=self.activation_fn)
        rnn_cell = Bidirectional(rnn_cell)
        rnn_outputs = RNN(rnn_cell)(inputs)
        rnn_outputs = Permute((1, 0, 2))(rnn_outputs)
        rnn_outputs = Lambda(lambda x: x[-1, :, :])(rnn_outputs)
        z0_mu = Dense(self.q, activation=None)(rnn_outputs)
        z0_log_sig_sq = Dense(self.q, activation=None)(rnn_outputs)
        outputs = Concatenate(axis=1)([z0_mu, z0_log_sig_sq])
        return Model(inputs=inputs, outputs=outputs, name=name)

    def dec_mnist_bball(self, shape, name=None):
        inputs = Input(shape=(self.q,))
        intermediate = inputs
        if self.task == 'mnist':
            intermediate = Dense(3 * 3 * 8, activation=None)(intermediate)
            intermediate = Reshape((3, 3, 8))(intermediate)
            intermediate = Conv2DTranspose(self.NF_dec * 4, 3, name='h0')(intermediate)
            intermediate = BatchNormalization()(intermediate)
            intermediate = Activation("relu")(intermediate)
        elif self.task == 'bballs':
            intermediate = Dense(4 * 4 * 8, activation=None)(intermediate)
            intermediate = Reshape((4, 4, 8))(intermediate)
            intermediate = Conv2DTranspose(self.NF_dec * 4, self.KW_dec, name='h0')(intermediate)
            intermediate = BatchNormalization()(intermediate)
            intermediate = Activation("relu")(intermediate)
        else:
            raise ValueError('task is incorrect: {:s}'.format(self.task))
        intermediate = Conv2DTranspose(self.NF_dec * 2, self.KW_dec, name='h1')(intermediate)
        intermediate = BatchNormalization()(intermediate)
        intermediate = Activation("relu")(intermediate)
        intermediate = Conv2DTranspose(self.NF_dec, self.KW_dec, name='h2')(intermediate)
        intermediate = BatchNormalization()(intermediate)
        intermediate = Activation("relu")(intermediate)
        intermediate = Conv2DTranspose(1, self.KW_dec, name='h3')(intermediate)
        logits = Activation(tf.keras.activations.sigmoid)(intermediate)
        logits = Flatten()(logits)
        outputs = Concatenate(axis=1)([logits, logits])
        return Model(inputs=inputs, outputs=outputs, name=name)

    def dec_dense(self, name=None):
        inputs = Input(shape=(self.q,))
        z = inputs
        for i in range(self.Nd):
            z = Dense(self.Hd, activation=self.activation_fn)(z)
        x_mean = Dense(self.D, activation=None)(z)
        x_log_sig_sq = Dense(1, activation='sigmoid')(z)
        x_log_sig_sq = RepeatVector(self.D)(x_log_sig_sq)
        x_log_sig_sq = Reshape((self.D,))(x_log_sig_sq)
        outputs = Concatenate()(x_mean, x_log_sig_sq)
        return Model(inputs=inputs, outputs=outputs, name=name)

    def ode_density(self, v0, s0, logP0, fmom, t, dt=None):
        def ode_density_step(v_s_logP, t):  # N,2q+1
            v_s = v_s_logP[:, :-1]  # N,2q
            f1 = fmom(v_s, t)  # N,q
            f2 = v_s[:, :self.q]  # N,q
            df1dvs = ops.convert_to_tensor([tf.gradients(ys=f1[:, d], xs=v_s)[0] for d in range(self.q)])  # q,N,2q
            df1dv = df1dvs[:, :, :self.q]  # q,N,q
            df1dv = tf.transpose(a=df1dv, perm=[1, 0, 2])  # N,q,q
            dlogP = -tf.linalg.trace(df1dv)[:, None]  # N,1
            dv_ds_dlogP = tf.concat([f1, f2, dlogP], 1)  # N,2q+1
            return dv_ds_dlogP

        if dt is None:
            dt = (t[1] - t[0]) / 5
        v0_s0_logp0 = tf.concat([v0, s0, logP0], 1)  # N,2q+1
        vt_st_logpt = self.solver.solve(ode_density_step, 0, v0_s0_logp0, [t]).states[0]  # T,N,2q+1
        vt = tf.identity(vt_st_logpt[:, :, 0:self.q], name="latent-velocity")
        st = tf.identity(vt_st_logpt[:, :, self.q:2 * self.q], name="latent")
        logpt = tf.identity(vt_st_logpt[:, :, -1], name="latent_density")
        return vt, st, logpt

    def ode(self, v0, s0, fmom, t, dt=None):
        if dt is None:
            dt = (t[1] - t[0]) / 5
        v0_s0 = tf.concat([v0, s0], 1)  # N,2q

        def ode_f_helper(v_s, t):
            f1 = fmom(v_s, t)  # N,q
            f2 = v_s[:, :self.q]  # N,q
            return tf.concat([f1, f2], 1)  # N,2q

        vt_st = self.solver.solve(ode_f_helper, 0, v0_s0, [t]).states[0]  # T,N,2q
        vt = tf.identity(vt_st[:, :, 0:self.q], name="latent-velocity-mean")
        st = tf.identity(vt_st[:, :, self.q:], name="latent-mean")
        return vt, st

    def decode(self, Z, dec_model):
        x_rec = dec_model(Z)
        x_rec_mean, x_rec_log_sig_sq = tf.split(x_rec, num_or_size_splits=2, axis=1)
        x_rec_mean = tf.identity(x_rec_mean, name="rec_mean")
        x_rec_log_sig_sq = tf.identity(x_rec_log_sig_sq, name="rec_logvar")
        return x_rec_mean, x_rec_log_sig_sq

    def gauss_KL(self, mu1, mu2, logsigmasq, loglambdasq):  # KL( N(mu1,diag(sig2)) || N(mu2,lambda2xI) )
        return 0.5 * tf.reduce_sum(
            input_tensor=-1 - logsigmasq + loglambdasq + (tf.square(mu1 - mu2) + tf.exp(logsigmasq)) / tf.exp(
                loglambdasq), axis=1)

    def gauss_KL_I(self, mu, logsigmasq):  # KL( N(mu,sig2) || N(0,I) )
        return self.gauss_KL(mu, 0.0, logsigmasq, 0.0)

    def rec_lhood(self, x, x_rec_mu, x_rec_log_sigma_sq, mean):
        N = tf.shape(input=x)[0]
        x = tf.transpose(a=x, perm=[1, 0, 2])  # T,N,D
        if mean:
            x = tf.reshape(x, [-1, self.D])
        else:
            x = tf.reshape(tf.tile(x, [1, 1, self.L]), [-1, self.D])
        if self.dec_out == 'bernoulli':
            lhood = tf.reduce_sum(
                input_tensor=x * tf.math.log(1e-5 + x_rec_mu) + (1 - x) * tf.math.log(1e-5 + 1 - x_rec_mu), axis=1)
        elif self.dec_out == 'normal':
            mvn = tfd.MultivariateNormalDiag(loc=x_rec_mu, scale_diag=tf.sqrt(1e-10 + tf.exp(x_rec_log_sigma_sq)))
            lhood = tf.reduce_sum(input_tensor=mvn.log_prob(x))
        return tf.reduce_sum(input_tensor=lhood) / tf.cast(N, tf.float32) / self.L  # = 1/N * \sum_{i,t} x_t^i

    def create_loss(self):
        self.KL_w = 0
        for w in self.f_weight_means:
            self.KL_w += tf.reduce_sum(input_tensor=self.gauss_KL_I(w, self.loglambdasq))
        for w in self.f_bias_means:
            self.KL_w += tf.reduce_sum(input_tensor=self.gauss_KL_I(tf.reshape(w, [1, -1]), self.loglambdasq))
        self.mean_reconstr_lhood = self.rec_lhood(self.x, self.x_rec_mu_mu, self.x_rec_log_sigma_sq_mu, True)
        self.reconstr_lhood = self.rec_lhood(self.x, self.x_rec_mu_L, self.x_rec_log_sigma_sq_L, False)
        tf.compat.v1.summary.scalar('beta-KL', self.beta * self.KL_w)
        tf.compat.v1.summary.scalar('reconstr_lhood', self.reconstr_lhood)
        self.inst_enc_loss = tf.constant(0.0)
        N = tf.shape(input=self.x)[0]
        self.log_p = tfd.MultivariateNormalDiag(loc=tf.zeros(2 * self.q), scale_diag=tf.ones(2 * self.q)).log_prob(
            self.zt_L)
        self.log_p = tf.reduce_sum(input_tensor=self.log_p) / tf.cast(N, tf.float32) / self.L
        self.log_q = tf.reduce_sum(input_tensor=self.logpt_L) / tf.cast(N, tf.float32) / self.L
        tf.compat.v1.summary.scalar('log_p', self.log_p)
        tf.compat.v1.summary.scalar('log_q', self.log_q)
        self.elbo = -self.beta * self.KL_w + self.reconstr_lhood - self.log_q + self.log_p
        self.vae_loss = tf.multiply(-1.0, self.elbo, name='loss')
        if self.inst_enc_KL and self.gamma > 0.0:
            mvn = tfd.MultivariateNormalDiag(loc=self.zt_L_mu_enc,
                                             scale_diag=tf.sqrt(tf.exp(self.zt_L_log_sigma_sq_enc)))
            self.qenc_zt_ode = tf.reduce_sum(input_tensor=mvn.log_prob(self.zt_L_ode)) / tf.cast(N, tf.float32) / self.L
            self.inst_enc_loss = self.gamma * (self.log_q - self.qenc_zt_ode)
            self.vae_loss += self.inst_enc_loss
            print('inst_enc_KL loss added to VAE loss')
        tf.compat.v1.summary.scalar('inst_enc_loss', self.inst_enc_loss)

    def create_optimizers(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.expdec = tf.compat.v1.train.exponential_decay(self.eta, self.global_step, 200, 0.995, staircase=True)
        vae_vars = tf.compat.v1.trainable_variables()
        numpars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
        print('{:d} total trainable parameters'.format(numpars))
        for i in vae_vars:
            print(i)
        self.vae_optimizer = tf.compat.v1.train.AdamOptimizer(self.expdec).minimize(self.vae_loss, \
                                                                                    name='adam_vae', var_list=vae_vars,
                                                                                    global_step=self.global_step)
        self.elbo_optimizer = tf.compat.v1.train.AdamOptimizer(self.expdec).minimize(-self.elbo, \
                                                                                     name='adam_vae_elbo',
                                                                                     var_list=vae_vars,
                                                                                     global_step=self.global_step)
        self.vae_rec_optimizer = tf.compat.v1.train.AdamOptimizer(self.expdec).minimize(-self.reconstr_lhood, \
                                                                                        name='adam_vae_rec',
                                                                                        var_list=vae_vars,
                                                                                        global_step=self.global_step)

    def integrate(self, X, t=None, dt=0.1):
        t = dt * np.arange(0, X.shape[1], dtype=np.float32)
        st = self.sess.run((self.st_L), feed_dict={tf.compat.v1.get_default_graph().get_tensor_by_name('X:0'): X, \
                                                   tf.compat.v1.get_default_graph().get_tensor_by_name('t:0'): t,
                                                   tf.compat.v1.get_default_graph().get_tensor_by_name('Tss:0'):
                                                       X.shape[1], \
                                                   self.train: False})
        st = np.reshape(st, (-1, X.shape[0] * self.L, self.q))  # [T,N*L,D]
        st = np.transpose(st, [1, 2, 0])  # [N*L,D,T]
        return st

    def reconstruct(self, X, t=None, dt=0.1):
        t = dt * np.arange(0, X.shape[1], dtype=np.float32)
        Xrec = self.sess.run(self.x_rec_mu_mu, feed_dict={tf.compat.v1.get_default_graph().get_tensor_by_name('X:0'): X, \
                                                          tf.compat.v1.get_default_graph().get_tensor_by_name('t:0'): t,
                                                          tf.compat.v1.get_default_graph().get_tensor_by_name('Tss:0'):
                                                              X.shape[
                                                                  1], \
                                                          self.train: False})
        Xrec = np.reshape(Xrec, (-1, X.shape[0], self.D))  # [T,N,D]
        Xrec = np.transpose(Xrec, [1, 0, 2])  # [N,T,D]
        return Xrec

    def save_model(self, ckpt_dir, ext='noname'):
        saver = tf.compat.v1.train.Saver(save_relative_paths=True)
        saver.save(self.sess, os.path.join(ckpt_dir, self.task, ext))


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='same', name="enc"):
    with tf.compat.v1.variable_scope(name):
        return tf.compat.v1.layers.conv2d(input_, output_dim, [k_h, k_w], [d_h, d_w], padding=padding)


def deconv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='same', name="dec"):
    with tf.compat.v1.variable_scope(name):
        return tf.compat.v1.layers.conv2d_transpose(input_, output_dim, [k_h, k_w], [d_h, d_w], padding=padding)
