import tensorflow as tf
from src.layers import BatchInstanceNormalization, Robust_SSMS_SpectralDropout
from tensorflow.keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
import tensorflow as tf
from keras.regularizers import L2
from keras import layers

def bn(name='1'):
    return layers.BatchNormalization(name='bn'+str(name))

def VGG16_with_leaky_relu_and_spectral_dropout(input_shape=(32, 32, 3), classes=10,rate=0.35):
    """
    VGG16 architecture with Leaky ReLU activation and Spectral Dropout.

    Args:
        input_shape: Shape of the input images (height, width, channels).
        classes: Number of output classes.

    Returns:
        A tf.keras.Model instance.
    """
    alpha=0.0005;alpha2=0.1
    input_layer = Input(shape=input_shape)
    rate=rate

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block1_conv1')(input_layer)
    x=BatchInstanceNormalization()(x) #block_size=block_size
    x = Robust_SSMS_SpectralDropout(0.1,power=1.0,mag_noise=0.05,max_rate=0.3)(x)
    x = Conv2D(64, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block1_conv2')(x)
    x=BatchInstanceNormalization()(x) #x=BatchNormalization()(x) LocalInstanceNormalizationL1
    x = Robust_SSMS_SpectralDropout(0.1,power=1.2,mag_noise=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
     # Added dropout

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block2_conv1')(x)
    x = Robust_SSMS_SpectralDropout(rate,power=1.3,mag_noise=0.05,max_rate=0.5)(x)#x = Robust_SSMS_SpectralDropout(rate,power=2.5,mag_noise=0.05)(x)
    x=BatchInstanceNormalization()(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block2_conv2')(x)
    x=BatchInstanceNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block3_conv1')(x)
    x = Robust_SSMS_SpectralDropout(rate,power=1.3,mag_noise=0.05,max_rate=0.6)(x)
    x=BatchInstanceNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block3_conv2')(x)
    x = Robust_SSMS_SpectralDropout(rate,power=1.3,mag_noise=0.05,max_rate=0.6)(x)
    x=BatchInstanceNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block3_conv3')(x)
    x=BatchInstanceNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block4_conv1')(x)
    x = Robust_SSMS_SpectralDropout(rate+0.05,power=1.3,mag_noise=0.05,max_rate=0.6)(x)
    x=BatchInstanceNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block4_conv2')(x)
    x = Robust_SSMS_SpectralDropout(rate+0.05,power=1.3,mag_noise=0.05,max_rate=0.6)(x)#x = Robust_SSMS_SpectralDropout(rate+0.05,power=1.8,mag_noise=0.05)(x)
    x=BatchInstanceNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block4_conv3')(x)
    x=BatchInstanceNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block4_conv4')(x)
    x = Robust_SSMS_SpectralDropout(rate+0.1,power=1.3,mag_noise=0.05,max_rate=0.45)(x)
    x=BatchInstanceNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block4_conv5')(x)
    x = Robust_SSMS_SpectralDropout(rate+0.1,power=1.3,mag_noise=0.05,max_rate=0.45)(x)
    x=BatchInstanceNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=alpha2),kernel_regularizer=L2(alpha),name='block4_conv6')(x)
    x=BatchInstanceNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classification part
    x = Flatten()(x)
    x = Dropout(0.5)(x)  # Added dropout
    x = Dense(512, activation=LeakyReLU(alpha=0.1),kernel_regularizer=L2(alpha),name='dense_1')(x)
    x = bn(name='b')(x)#x=BatchNormalization()(x)
    
    output_layer = Dense(classes, activation='softmax',name='dense_2')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model