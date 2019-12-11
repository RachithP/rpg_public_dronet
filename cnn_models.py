import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3

def resnet8(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

def vgg(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))
    
    # Block -1 
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_1')(x1)

    # Block -2
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_1_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_2')(x2)
    
    # Block - 3
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_1_block_3',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2_block_3',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_3')(x3)
    
    # Block - 4
#     x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_1_block_4',
#                kernel_initializer="he_normal", 
#                 kernel_regularizer=regularizers.l2(1e-4))(x3)
#     x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_2_block_4',
#                kernel_initializer="he_normal", 
#                 kernel_regularizer=regularizers.l2(1e-4))(x4)
#     x4 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_4')(x4)

    x = Flatten(name='fc1')(x3)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    
    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

def inceptionv3(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    print("Image Channels : ", img_channels)
    base_model = InceptionV3(include_top=False,
    					weights='imagenet',
                       	input_shape=(img_height, img_width, img_channels))

    # Disbaling trainability of resnet feature extraction layers
    for layer in base_model.layers:
        layer.trainable = False

    # Printing model summary
    # print(base_model.summary())

    '''
    for layer in base_model.layers:
        # check for convolutional layer
        if ('Conv2D' not in layer.__class__.__name__):
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)
    ''' 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=base_model.input, outputs=[steer, coll])
    # print(model.summary())

    return model
