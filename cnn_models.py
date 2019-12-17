import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, AveragePooling2D
from keras.layers.merge import add, Concatenate, concatenate
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

def three_block_model(img_width, img_height, img_channels, output_dim):
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
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_1_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(img_input)
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_2_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_1')(x1)

    # Block -2
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_1_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_2_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_2')(x2)
    
    # Block - 3
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1_block_3',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2_block_3',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_3')(x3)

    x = Flatten(name='fc1')(x3)
    x = Dropout(0.5)(x)

    x1 = Dense(512, activation='relu', name='fc2')(x)
    x1 = Dropout(0.5)(x1)
    
    # Steering channel
    steer = Dense(output_dim)(x1)

    # Collision channel
    x2 = Dense(128, activation='relu', name='fc3')(x)
    x2 = Dropout(0.3)(x2)
    coll = Dense(output_dim)(x2)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

def three_block_model_batchnorm(img_width, img_height, img_channels, output_dim):
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
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_1_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(img_input)
    x1 = keras.layers.normalization.BatchNormalization()(x1)
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_2_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_1')(x1)

    # Block -2
    x1 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_1_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_2_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_2')(x2)
    
    # Block - 3
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1_block_3',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x3 = keras.layers.normalization.BatchNormalization()(x3)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2_block_3',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_3')(x3)

    x = Flatten(name='fc1')(x3)
    x = Dropout(0.5)(x)

    #x1 = Dense(2048, activation='relu', name='fc2')(x)
    x1 = Dense(512, activation='relu', name='fc2')(x)
    x1 = Dropout(0.5)(x1)

    # Steering channel
    steer = Dense(output_dim)(x1)

    # Collision channel
    #x2 = Dense(2048, activation='relu', name='fc3')(x)
    x2 = Dense(128, activation='relu', name='fc3')(x)
    x2 = Dropout(0.3)(x2)
    coll = Dense(output_dim)(x2)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

def three_blocks_avgpooling(img_width, img_height, img_channels, output_dim):
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
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_1_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(img_input)
    x1 = keras.layers.normalization.BatchNormalization()(x1)
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_2_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_1')(x1)

    # Block -2
    x1 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_1_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_2_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_2')(x2)
    
    # Block - 3
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1_block_3',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x3 = keras.layers.normalization.BatchNormalization()(x3)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2_block_3',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_3')(x3)

    x = Flatten(name='fc1')(x3)
    x = Dropout(0.5)(x)

    #x1 = Dense(2048, activation='relu', name='fc2')(x)
    x1 = Dense(256, activation='relu', name='fc2')(x)
    x1 = Dropout(0.3)(x1)

    # Steering channel
    steer = Dense(output_dim)(x1)

    # Collision channel
    x2 = GlobalAveragePooling2D()(x3)
    coll = Dense(output_dim)(x2)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

def deep_network(img_width, img_height, img_channels, output_dim):
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
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_1_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(img_input)
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_2_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_1')(x1)

    # Block -2
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_1_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_2_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    #x1 = Conv2D(64, (1,1), activation='relu', padding='same')(x1)
    #x2 = add([x1, x2])
    #x2 = concatenate([x1, x2])
    x2 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_2')(x2)
    
    # Block - 3
    x3 = Conv2D(64, (3, 3), activation='relu', name='conv_1_block_3', padding='same',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x3 = Conv2D(64, (3, 3), activation='relu', name='conv_2_block_3', padding='same',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_3')(x3)
   
    x4 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_4')(x1)
    x4 = concatenate([x4,x2])
    x4 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_5')(x4)
    x3 = concatenate([x4, x3])

    x3 = Conv2D(16, (1,1), activation='relu', padding='same')(x3)
    x = Flatten(name='fc1')(x3)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    #x = AveragePooling2D(pool_size=(3,3))(x3)
    #x1 = Dense(512, activation='relu', name='fc2')(x)
    #x1 = Dropout(0.5)(x1)
    
    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    #x2 = Dense(128, activation='relu', name='fc3')(x)
    #x2 = Dropout(0.3)(x2)
    #x = GlobalAveragePooling2D()(x3)
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

def deep_network_batch_norm(img_width, img_height, img_channels, output_dim):
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
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_1_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(img_input)
    x1 = keras.layers.normalization.BatchNormalization()(x1)
    x1 = Conv2D(32, (7, 7), activation='relu', padding='same', name='conv_2_block_1',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_1')(x1)

    # Block -2
    x1 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_1_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x1)
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv_2_block_2',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    #x1 = Conv2D(64, (1,1), activation='relu', padding='same')(x1)
    #x2 = add([x1, x2])
    #x2 = concatenate([x1, x2])
    x2 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_2')(x2)
    
    # Block - 3
    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x3 = Conv2D(64, (3, 3), activation='relu', name='conv_1_block_3', padding='same',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x2)
    x3 = keras.layers.normalization.BatchNormalization()(x3)
    x3 = Conv2D(64, (3, 3), activation='relu', name='conv_2_block_3', padding='same',
               kernel_initializer="he_normal", 
                kernel_regularizer=regularizers.l2(1e-4))(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_3')(x3)
   
    x4 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_4')(x1)
    x4 = concatenate([x4,x2])
    x4 = MaxPooling2D(pool_size=(2, 2), strides=[2,2], name='maxpool_block_5')(x4)
    x3 = concatenate([x4, x3])

    x3 = Conv2D(16, (1,1), activation='relu', padding='same')(x3)
    x = Flatten(name='fc1')(x3)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    #x = AveragePooling2D(pool_size=(3,3))(x3)
    #x1 = Dense(512, activation='relu', name='fc2')(x)
    #x1 = Dropout(0.5)(x1)
    
    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    #x2 = Dense(128, activation='relu', name='fc3')(x)
    #x2 = Dropout(0.3)(x2)
    #x = GlobalAveragePooling2D()(x3)
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

def avgpool_network(img_width, img_height, img_channels, output_dim):
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
    x = GlobalAveragePooling2D()(x7)
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

    x = AveragePooling2D(pool_size=(2, 2))(base_model.output)
    x = Flatten()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    #x = GlobalAveragePooling2D()(x)

    # Steering channel
    steer = Dense(output_dim)(x)

    # Collision channel
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=base_model.input, outputs=[steer, coll])
    print(model.summary())

    return model
