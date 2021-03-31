import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Lambda, Concatenate, Conv2DTranspose, InputLayer
from tensorflow.keras.losses import Loss

import tensorflow_probability as tfp


class ODE2VAE2D(Model):
    """
    This class is our attempt to rewrite a simple/minimalistic version of ODE2VAE using Tensorflow 2.x from scratch.
    However, this task proved to be intractible given the complexity of the source code and degree to which it was
    completely undocumented spaghetti.
    """
    def __init__(self, img_width: int, img_height: int, img_channels: int, latent_dim: int = 25, num_timesteps: int = 3,
                 batch_size: int = 8, learning_rate: float = 0.001, beta: float = 1.5, encoding_filters: int = 32,
                 decoding_filters: int = 32, decoder_depth: int = 20, hidden_depth: int = 3, hidden_neurons: int = 100,
                 acc_var: float = 1e-4, fps: int = 60, stiff: bool = True):
        super().__init__()

        # Shape parameters
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels  # together, the product of these three give D, the dimension of the data space
        self.latent_dim = latent_dim  # d, the dimension of the latent space
        self.num_timesteps = num_timesteps  # maximum video length in dataset

        # Model parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta  # for beta-VAE
        self.encoding_filters = encoding_filters
        self.decoding_filters = decoding_filters
        self.decoder_depth = decoder_depth
        self.hidden_depth = hidden_depth
        self.hidden_neurons = hidden_neurons
        self.acc_var = acc_var  # variance of the acceleration BNN (used for initialization and loss)

        # The time delta between frames
        self.dt = 1 / fps

        # If stiff use BDF and if not use Dormand-Prince
        self.ode_solver = tfp.math.ode.BDF() if stiff else tfp.math.ode.DormandPrince()

        input_shape = (self.num_timesteps, self.img_width, self.img_height, self.img_channels)

        self.inputs = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.latest_input = Lambda(lambda x: x[-1, :, :, :], output_shape=input_shape[1:])(self.inputs)

        # Encoders

        # Position encoder
        penc_conv1 = tf.keras.layers.Conv2D(filters=self.encoding_filters, kernel_size=5, strides=(2, 2),
                                            activation='relu')(self.latest_input)
        penc_conv2 = tf.keras.layers.Conv2D(filters=2 * self.encoding_filters, kernel_size=5, strides=(2, 2),
                                            activation='relu')(penc_conv1)
        penc_conv3 = tf.keras.layers.Conv2D(filters=4 * self.encoding_filters, kernel_size=5, strides=(2, 2),
                                            activation='relu')(penc_conv2)
        penc_conv4 = tf.keras.layers.Conv2D(filters=8 * self.encoding_filters, kernel_size=5, strides=(2, 2),
                                            activation='relu')(penc_conv3)
        penc_flatten = tf.keras.layers.Flatten()(penc_conv4)

        self.penc_output = tf.keras.layers.Dense(2 * self.latent_dim)(penc_flatten)

        self.pos_encoder = Model(self.latest_input, self.penc_output, name='position encoder')

        # Velocity encoder
        venc_conv1 = tf.keras.layers.Conv3D(filters=self.decoding_filters * self.num_timesteps, kernel_size=5,
                                            strides=(2, 2, 2), activation='relu')(self.inputs)
        venc_conv2 = tf.keras.layers.Conv3D(filters=2 * self.decoding_filters * self.num_timesteps, kernel_size=5,
                                            strides=(2, 2, 2), activation='relu')(venc_conv1)
        venc_conv3 = tf.keras.layers.Conv3D(filters=4 * self.decoding_filters * self.num_timesteps, kernel_size=5,
                                            strides=(2, 2, 2), activation='relu')(venc_conv2)
        venc_conv4 = tf.keras.layers.Conv3D(filters=8 * self.decoding_filters * self.num_timesteps, kernel_size=5,
                                            strides=(2, 2, 2), activation='relu')(venc_conv3)
        venc_flatten = tf.keras.layers.Flatten()(venc_conv4)

        # No activation
        self.venc_output = tf.keras.layers.Dense(2 * self.latent_dim)(venc_flatten)

        self.velocity_encoder = Model(self.inputs, self.venc_output, name='velocity encoder')

        # Since the latent acceleration is a function of position and velocity the encodings need to be concatenated
        # self.combined = Concatenate([self.pos_encoder, self.velocity_encoder])
        # self.encoder = Model(self.inputs, self.combined)

        # Latent acceleration model

        # Model logvar TODO
        self.acc_logvar = tf.Variable(np.log(self.acc_var), dtype=tf.float32, name='S')

        # Build a fully connected dense model with 2d input and d output with wide hidden layers
        acc_layer_sizes = [2 * self.latent_dim]
        for tmp in range(self.hidden_depth):
            acc_layer_sizes.append(self.hidden_neurons)
        acc_layer_sizes.append(self.latent_dim)

        # The following complicated code is due to the fact that the acceleration model is actually a Bayesian NN.
        acc_weight_initializer = tf.initializers.GlorotUniform()  # same as initialization by the Xavier rule
        acc_bias_initializer = tf.initializers.Zeros()
        self.acc_weight_means = []  # weight matrix means
        self.acc_bias_means = []  # bias vector means
        for i in range(0, self.hidden_depth + 1):
            w_ = tf.Variable(acc_weight_initializer(shape=(acc_layer_sizes[i], acc_layer_sizes[i + 1])), name=f'w{i}')
            self.acc_weight_means.append(w_)
            b_ = tf.Variable(acc_bias_initializer(shape=(acc_layer_sizes[i + 1],)), name=f'b{i}')
            self.acc_bias_means.append(b_)

        self.Ws = []  # TF weight samples
        self.Bs = []  # TF bias samples
        for i in range(0, self.hidden_depth + 1):
            Wi = tf.random.normal((acc_layer_sizes[i], acc_layer_sizes[i + 1])) * tf.sqrt(tf.exp(self.acc_logvar)) \
                 + self.acc_weight_means[i]
            self.Ws.append(Wi)
            Bi = tf.random.normal((acc_layer_sizes[i + 1],)) * tf.sqrt(tf.exp(self.acc_logvar)) + self.acc_bias_means[i]
            self.Bs.append(Bi)

        # TODO use Model.summary to verify model

        # ODE flows TODO
        self.z0_mu, self.z0_logvar = self.encode()
        self.mvn = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(2 * self.latent_dim),
                                                            scale_diag=tf.ones(2 * self.latent_dim))

        # Decoder
        latent_shape = (latent_dim,)
        self.latent_input = InputLayer(input_shape=latent_shape)
        decode_intermediate = self.latent_input

        # TODO vary activation function
        # for _ in range(self.decoder_depth):
        #    decode_intermediate = Dense()(decode_intermediate)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    @tf.function
    def encode(self, x):
        p_mean, p_logvar = tf.split(self.pos_encoder(x), num_or_size_splits=2, axis=0)
        v_mean, v_logvar = tf.split(self.velocity_encoder(x), num_or_size_splits=2, axis=0)
        mean = tf.concat([p_mean, v_mean])
        logvar = tf.concat([p_logvar, v_logvar])
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    # TODO update training step from vanilla VAE loss
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= self.img_width * self.img_height
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def ode_density(self, v0, s0, logP0, fmom, t, T):
        def ode_density_step(t, v_s_logP):  # N,2q+1
            v_s = v_s_logP[:, :-1]  # N,2q
            f1 = fmom(v_s, t)  # N,q
            f2 = v_s[:, :self.q]  # N,q
            df1dvs = tf.convert_to_tensor([tf.gradients(f1[:, d], v_s)[0] for d in range(self.q)])  # q,N,2q
            df1dv = df1dvs[:, :, :self.q]  # q,N,q
            df1dv = tf.transpose(df1dv, [1, 0, 2])  # N,q,q
            dlogP = -tf.linalg.trace(df1dv)[:, None]  # N,1
            dv_ds_dlogP = tf.concat([f1, f2, dlogP], 1)  # N,2q+1
            return dv_ds_dlogP

        v0_s0_logp0 = tf.concat([v0, s0, logP0], 1)  # N,2q+1
        vt_st_logpt = self.ode_solver.solve(ode_density_step, t, v0_s0_logp0, solution_times=t).states  # T,N,2q+1
        vt = tf.identity(vt_st_logpt[:, :, 0:self.q], name="latent-velocity")
        st = tf.identity(vt_st_logpt[:, :, self.q:2 * self.q], name="latent")
        logpt = tf.identity(vt_st_logpt[:, :, -1], name="latent_density")
        return vt, st, logpt

    def ode(self, v0, s0, fmom, t, T):
        v0_s0 = tf.concat([v0, s0], 1)  # N,2q

        def ode_f_helper(v_s, t):
            f1 = fmom(v_s, t)  # N,q
            f2 = v_s[:, :self.q]  # N,q
            return tf.concat([f1, f2], 1)  # N,2q

        vt_st = self.ode_solver.solve(ode_f_helper, t, v0_s0, solution_times=t).states  # T,N,2q
        vt = tf.identity(vt_st[:, :, 0:self.q], name="latent-velocity-mean")
        st = tf.identity(vt_st[:, :, self.q:], name="latent-mean")
        return vt, st

    def latent_acc(self, x):
        for i in range(0, self.hidden_depth + 1):
            if i < self.hidden_depth:
                x = self.activation_fn(tf.matmul(x, self.Ws[i]) + self.Bs[i])
            elif i == self.hidden_depth:
                x = tf.matmul(x, self.Ws[i]) + self.Bs[i]
        return x

    def latent_acc_mean(self, x):
        for i in range(0, self.hidden_depth + 1):
            if i < self.hidden_depth:
                x = self.activation_fn(tf.matmul(x, self.acc_weight_means[i]) + self.acc_bias_means[i])
            elif i == self.hidden_depth:
                x = tf.matmul(x, self.acc_weight_means[i]) + self.acc_bias_means[i]
        return x

    def sample_trajectory(self):
        eps = tf.random.normal((2 * self.latent_dim,), 0, 1, dtype=tf.float32)
        z0 = tf.add(self.z0_mu, tf.multiply(tf.sqrt(tf.exp(self.z0_logvar)), eps))
        s0, v0 = tf.split(z0, num_or_size_splits=2, axis=0)
        q0 = self.mvn.log_prob(eps)
        q0 = tf.expand_dims(q0, 1)
        vt, st, logpt = self.ode_density(v0, s0, q0, self.latent_acc,
                                         self.t)  # T, N, q [t0x0,t0x1,...t0xN,t1x0,t1x1,...]
        return vt, st, logpt

    # TODO which axis is correct?
    # need to use tf.reduce_sum for batches
    @tf.function
    def gauss_KL(self, mu1, mu2, logvar1, logvar2):
        return 0.5 * (-1 - logvar1 + logvar2 + (tf.square(mu1 - mu2) + tf.exp(logvar1)) / tf.exp(logvar2))

    @tf.function
    def gauss_KL_I(self, mu, logvar):
        return self.gauss_KL(mu, 0.0, logvar, 0.0)
