#import numpy as np
#import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

def conv_block(input_tensor, num_filters, dropout_rate=0):
    """A block with two convolutional layers followed by optional dropout."""
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_tensor)
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def upsample_block(input_tensor, skip_tensor, num_filters):
    """A block for upsampling and concatenation with the skip connection."""
    x = UpSampling2D(size=(2, 2))(input_tensor)
    x = Conv2D(num_filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = concatenate([skip_tensor, x], axis=3)
    x = conv_block(x, num_filters)
    return x

def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    # Contracting path
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512, dropout_rate=0.5)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = conv_block(pool4, 1024, dropout_rate=0.5)

    # Expanding path
    up6 = upsample_block(conv5, conv4, 512)
    up7 = upsample_block(up6, conv3, 256)
    up8 = upsample_block(up7, conv2, 128)
    up9 = upsample_block(up8, conv1, 64)

    # Final layer
    conv10 = Conv2D(1, 1, activation='sigmoid')(up9)

    model = Model(inputs=inputs, outputs=conv10)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    model.summary()
    
    # Load pretrained weights if provided
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
