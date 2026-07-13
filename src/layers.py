from numpy import std
from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
import tensorflow as tf

from tensorflow.keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from keras.layers import LeakyReLU
from keras import layers

from tensorflow.keras import layers, constraints
class Between(constraints.Constraint):
    """Constrains weights to be between min_value and max_value."""
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

def bn(name='1'):
    return layers.BatchNormalization(name='bn'+str(name))


class Robust_SSMS_SpectralDropout(tf.keras.layers.Layer):
    def __init__(self, rate=0.4, noise_std=0.05, power=1.6, max_rate=0.7, mag_noise=0.1, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate; self.max_rate = max_rate
        self.noise_std = noise_std
        self.power = power
        self.mag_noise = mag_noise
        self.seed = seed
        self.eps = 1e-6 # Increased epsilon for SGD 0.1 stability

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        if not training:
            return inputs

        # 1. Forward Transform
        input_shape = tf.shape(inputs)
        x_t = tf.transpose(inputs, perm=[0, 3, 1, 2])
        X_freq = tf.signal.rfft2d(x_t)
        
        # 2. Stable Polar Decomposition
        # Use a small floor to prevent gradient singularity at 0
        real = tf.math.real(X_freq)
        imag = tf.math.imag(X_freq)
        mag = tf.sqrt(tf.square(real) + tf.square(imag) + self.eps)

        phase_real = real / mag
        phase_imag = imag / mag
        
        # 3. Create High-Frequency Mask
        freq_shape = tf.shape(X_freq)
        h, w_half = freq_shape[2], freq_shape[3]
        h_range = tf.range(h, dtype=tf.float32)
        w_range = tf.range(w_half, dtype=tf.float32)
        
        h_centers = tf.abs(tf.where(h_range > tf.cast(h//2, tf.float32), 
                                   h_range - tf.cast(h, tf.float32), h_range))
        dist_h, dist_w = tf.meshgrid(h_centers, w_range, indexing='ij')
        distance = tf.sqrt(tf.square(dist_h) + tf.square(dist_w))
        
        max_dist = tf.sqrt(tf.square(tf.cast(h//2, tf.float32)) + tf.square(tf.cast(w_half, tf.float32)))
        normalized_dist = distance / (max_dist + self.eps)
        norm_dist_exp = normalized_dist[tf.newaxis, tf.newaxis, :, :]

        # 4. Variable Rate Dropout (with Cap)
        # Cap variable rate at 0.5 so we never scale by more than 2x
        #variable_rate = tf.stop_gradient(tf.minimum(self.rate * tf.pow(norm_dist_exp, self.power), 0.4))
        growth_range = self.max_rate - self.rate
        variable_rate = self.rate + (growth_range * tf.pow(norm_dist_exp, self.power))
        keep_prob = 1.0 - variable_rate
        
        uniform = tf.random.uniform(freq_shape, seed=self.seed)
        mask = tf.cast(uniform < keep_prob, dtype=tf.float32)
        mag_dropped = mask * (mag / tf.maximum(keep_prob, self.eps))
        
        # Apply mask and scale
        #mag = (mag * mask) / keep_prob

        # 5. Stable Phase Jitter (Avoid tf.math.angle if possible)
        # We perturb the real and imag components directly to avoid atan2 gradients
        # We perturb the real and imag components directly to avoid atan2 gradients
        if self.mag_noise > 0.0:
            # Generate small rotation angle
            gauss_noise = tf.random.normal(
                shape=freq_shape, 
                mean=0.0, 
                stddev=1.0, 
                seed=self.seed + 1 if self.seed is not None else None
            )
            scaled_noise = gauss_noise * self.mag_noise * norm_dist_exp
            
            # Apply as a multiplicative factor: mag = mag * (1 + noise)
            # We use tf.maximum to ensure noise never accidentally flips the magnitude negative
            mag_perturbed = mag_dropped * tf.maximum(1.0 + scaled_noise, 0.1)
        else:
            # If no phase noise, just apply mask scaling to components
            #real = (real / (mag + self.eps)) * mag
            #imag = (imag / (mag + self.eps)) * mag
            mag_perturbed = mag_dropped
            
        real_out = phase_real * mag_perturbed
        imag_out = phase_imag * mag_perturbed

        # 6. Reconstruction
        X_complex = tf.complex(real_out, imag_out)
        x_rec = tf.signal.irfft2d(X_complex, fft_length=[input_shape[1], input_shape[2]])
        
        # 7. Final Safety Net: Local Gradient Clipping
        # This prevents a single bad batch from killing the whole model
        x_rec = tf.transpose(x_rec, perm=[0, 2, 3, 1])
        x_rec = tf.clip_by_value(x_rec, -1e6, 1e6) 
        
        return x_rec

class BatchInstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, block_size=6, epsilon=1e-5,g=0.4, **kwargs):
        super(BatchInstanceNormalization, self).__init__(**kwargs)
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer.")
        self.block_size = block_size
        self.epsilon = epsilon
        
        self.glo = layers.GlobalAveragePooling2D()

    def build(self, input_shape):
        # input_shape will be (batch_size, height, width, channels)
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension must be known for LocalInstanceNormalizationL1.')
        self.gamma = self.add_weight(
            name='gamma',
            shape=(channels,), # One gamma value per channel
            initializer='ones', # Typically initialized to ones
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(channels,), # One beta value per channel
            initializer='zeros', # Typically initialized to zeros
            trainable=True
        )
        self.nweight = self.add_weight(
            name='nweight',
            shape=(channels,), # One beta value per channel
            initializer='zeros', # Typically initialized to zeros
            trainable=True,
            constraint=Between(0.0,1.0)
        )
        
        super(BatchInstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        # Ensure inputs are float32 for consistent calculations
        inputs = tf.cast(inputs, tf.float32)

        # Reshape gamma and beta to (1, 1, 1, channels) for broadcasting across
        # batch, height, and width dimensions during the final scaling and shifting.
        gamma_reshaped = tf.reshape(self.gamma, [1, 1, 1, -1])
        beta_reshaped = tf.reshape(self.beta, [1, 1, 1, -1])
        local_mean = tf.nn.avg_pool(
            inputs,
            ksize=[1, self.block_size, self.block_size, 1],
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )
        
        # Then, compute the mean of these absolute differences within the same local windows.
        
        batch_mean, batch_sigma = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=True)
        #batch_sigma = tf.reduce_mean(tf.abs(inputs-batch_mean),[0,1,2],keepdims=True)
        x_batch = (inputs - batch_mean) / tf.sqrt(batch_sigma + self.epsilon)
        #x_batch = (inputs - batch_mean) / tf.sqrt(batch_sigma + self.epsilon) #local_mean = ins_mean #[:,tf.newaxis,tf.newaxis,:]
        ins_mean, ins_sigma = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        
        #normalized_output = (inputs - local_mean) / (1.25333*local_mad + self.epsilon)
        normalized_output = (inputs - ins_mean) / tf.sqrt(ins_sigma + self.epsilon)
        output = self.nweight*normalized_output + (1-self.nweight)*x_batch
        
        output = output * gamma_reshaped + beta_reshaped; 
        return output