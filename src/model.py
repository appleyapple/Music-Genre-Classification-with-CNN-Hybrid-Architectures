from keras import regularizers, optimizers
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Input, Bidirectional, Dense, Activation, Flatten, Dropout, BatchNormalization, Reshape, Permute, concatenate
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.recurrent import GRU, LSTM
from keras import applications

N_CATEGORIES = 8
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 4
INPUT_SHAPE = (128, 128, 3)  # x, y, color channels
IMAGE_SIZE = (128, 128)
frequency_axis = 1
time_axis = 2
channel_axis = 3

def build_crnn_model_regular():

    KERNAL_SIZE = [(3,3), (3,3), (3,3), (3, 3), (3,3), (3,3), (2,2)]
    POOL = [(2, 2), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2)]
    ACTIVATION = 'relu'
    NUM_CONV_LAYERS = 3
    FILTERS = [32, 64, 64, 32, 64, 128, 512]
    

    model = Sequential()
    # model.add(BatchNormalization(axis=frequency_axis,
    #                              input_shape=INPUT_SHAPE))  # IMAGE SIZE

    # First convolution layer specifies shape
    model.add(Conv2D(FILTERS[0], KERNAL_SIZE[0], padding='same',
                    input_shape=INPUT_SHAPE, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.02)))
    model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))

    model.add(MaxPooling2D(pool_size=POOL[0]))
   

    # model.add(Dropout(0.2))

    # Add more convolutional layers
    for layer in range(NUM_CONV_LAYERS - 1):

        model.add(Conv2D(FILTERS[layer + 1], KERNAL_SIZE[layer + 1], 
                    padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation(ACTIVATION))

        model.add(MaxPooling2D(pool_size=POOL[layer + 1]))

    
    # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(256, return_sequences=True, dropout = 0.3))
    model.add(GRU(256, return_sequences=False, dropout = 0.3))
    model.add(Dropout(0.5))

    # Output layer
    # model.add(Flatten())
    # model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001), 
    #                     activity_regularizer=regularizers.l1(0.001)))
    model.add(BatchNormalization())
    model.add(Dense(128, activation=ACTIVATION))
    model.add(Dense(128, activation=ACTIVATION))
    model.add(Dense(64, activation=ACTIVATION))
    model.add(Dense(32, activation=ACTIVATION))
    model.add(Dense(16, activation=ACTIVATION))
    model.add(Dropout(0.2))

    model.add(Dense(N_CATEGORIES, activation='softmax'))
    # optimizers.rmsprop(lr=0.001, decay=1e-6)
    # Adam(lr=0.001, beta_1=0.5, beta_2=0.999)
    model.compile(optimizers.Adadelta(),
                loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    
    return model

def build_cnn_model_regular():
    KERNAL_SIZE = (3, 3)
    POOL = (2, 2)
    ACTIVATION = 'relu'
    NUM_CONV_LAYERS = 3
    FILTERS = [32, 64, 64, 128]
    # Build CNN model
    # Conv2D - feature extraction using filters (passing filter over sections of image)
    # Dropout layer - randomly drop nodes; reduces overfitting
    # Max Pooling - taking maximum value of "window" instead of all values within it; reduce data size
    # Flatten layer - Converts n-dimension array to 1D array for NN

    model = Sequential()

    model.add(Conv2D(FILTERS[0], KERNAL_SIZE, padding='same',
                    input_shape=INPUT_SHAPE))
    model.add(Activation(ACTIVATION))
    model.add(MaxPooling2D(pool_size=POOL))
    model.add(Dropout(0.25))

    for layer in range(NUM_CONV_LAYERS - 1):
        model.add(Conv2D(FILTERS[layer + 1], KERNAL_SIZE, padding='same'))
        model.add(Activation(ACTIVATION))
        model.add(MaxPooling2D(pool_size=POOL))
        model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation(ACTIVATION))
    model.add(Dropout(0.5))
    model.add(Dense(N_CATEGORIES, activation='softmax'))

    model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),
                loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

#https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1298.pdf
def build_cnn_model_duplicated():
    KERNAL_SIZE = [(7, 7), (5, 5), (4, 4), (4, 4), (4, 2),
                   (2, 4), (3, 3), (3, 3), (3, 3)]
    POOL = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2),
            (2, 2), (2, 2), (2, 2), (2, 2)]
    ACTIVATION = 'relu'
    NUM_CONV_LAYERS = 4
    FILTERS = [8, 16, 32, 64, 128, 256, 256]

    # Input layer
    L1 = Input(shape=INPUT_SHAPE, name="input")

    # Visible layers
    L2 = Conv2D(FILTERS[0], KERNAL_SIZE[0],
                padding='same', activation=ACTIVATION, kernel_regularizer=regularizers.l2(0.02))(L1)
    L3 = Dropout(0.2)(L2)
    L4 = Conv2D(FILTERS[0], KERNAL_SIZE[1],
                padding='same', activation=ACTIVATION , kernel_regularizer=regularizers.l2(0.01))(L3)

    # Hidden 1
    H1_1 = MaxPooling2D(pool_size=POOL[0], strides=(2, 2))(L3)

    # Hidden 2
    H2_1 = AveragePooling2D(pool_size=POOL[0], strides=(2, 2))(L4)

    # Hidden 3
    H3_1 = Conv2D(FILTERS[0], KERNAL_SIZE[0],
                  padding='same', activation=ACTIVATION, kernel_regularizer=regularizers.l2(0.01))(L4)
    H3_2 = MaxPooling2D(pool_size=POOL[0], strides=(2, 2))(H3_1)

    # Hidden 4
    H4_1 = Conv2D(FILTERS[2], KERNAL_SIZE[0],
                  padding='same', activation=ACTIVATION, kernel_regularizer=regularizers.l2(0.01))(L4)
    H4_2 = AveragePooling2D(pool_size=POOL[0], strides=(2, 2))(H4_1)

    # Concatenate and flatten layer for dense layer
    concat = concatenate([H1_1, H2_1, H3_2, H4_2])
    flattened = Flatten()(concat)

    # Output layer
    L5 = Dense(128, activation=ACTIVATION)(flattened)
    L6 = Dense(32, activation=ACTIVATION)(L5)
    L7 = Dropout(0.1)(L6)
    L8 = Dense(8, activation='softmax')(L7)

    model = Model(inputs=L1, outputs=L8)

    model.compile(optimizers.Adam(),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model



def build_crnn_model_duplicated():
    KERNAL_SIZE = [(3, 3), (5, 5), (7, 7), (4, 4), (4, 2),
                   (2, 4), (3, 3), (3, 3), (3, 3)]
    POOL = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2),
            (2, 2), (2, 2), (2, 2), (2, 2)]
    ACTIVATION = 'relu'
    FILTERS = [4, 16, 16, 32, 128, 256, 256]

    # Input layer
    L1 = Input(shape=INPUT_SHAPE, name="input")

    # Visible layers
    L2 = Conv2D(FILTERS[0], KERNAL_SIZE[0],
                padding='same', activation=ACTIVATION, kernel_regularizer=regularizers.l2(0.02))(L1)
    L3 = Dropout(0.2)(L2)
    L4 = Conv2D(FILTERS[0], KERNAL_SIZE[1],
                padding='same', activation=ACTIVATION, kernel_regularizer=regularizers.l2(0.01))(L3)

    # Hidden 1
    H1_1 = MaxPooling2D(pool_size=POOL[0])(L3)

    # Hidden 2
    H2_1 = AveragePooling2D(pool_size=POOL[0])(L4)

    # Hidden 3
    H3_1 = Conv2D(FILTERS[1], KERNAL_SIZE[2],
                  padding='same', activation=ACTIVATION, kernel_regularizer=regularizers.l2(0.01))(L4)
    H3_2 = MaxPooling2D(pool_size=POOL[0])(H3_1)

    # Hidden 4
    H4_1 = Conv2D(FILTERS[1], KERNAL_SIZE[2],
                  padding='same', activation=ACTIVATION, kernel_regularizer=regularizers.l2(0.01))(L4)
    H4_2 = AveragePooling2D(pool_size=POOL[0])(H4_1)

    # Concatenate parallel hidden layers
    concat = concatenate([H1_1, H2_1, H3_2, H4_2]) # (None, 64, 64, 160)

    # Reshape for recurrent layer

    # resize_shape = concat.output_shape[2] * concat.output_shape[3]
    shape = concat.get_shape()
    R1 = Permute((time_axis, frequency_axis, channel_axis))(concat)
    # R2 = Reshape((concat.output_shape[1], resize_shape))(R1)
    R2 = Reshape((shape[1], shape[2]*shape[3]))(R1)

    # Recurrent layer
    R3 = LSTM(64, return_sequences=True, dropout=0.3)(R2)
    R4 = LSTM(64, return_sequences=False, dropout=0.3)(R3)
    # R5 = Dropout(0.1)(R4)

    # # Flatten layer for dense layer
    # flattened = Flatten()(R5)

    # Output layer
    # N1 = BatchNormalization()(R4)
    L5 = Dense(64, activation=ACTIVATION)(R4)
    L6 = Dense(32, activation=ACTIVATION)(L5)
    L7 = Dropout(0.1)(L6)
    L8 = Dense(8, activation='softmax')(L7)

    model = Model(inputs=L1, outputs=L8)

    model.compile(optimizers.rmsprop(lr=0.0015),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model