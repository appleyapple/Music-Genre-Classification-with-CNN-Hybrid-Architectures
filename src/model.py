from keras import regularizers, optimizers
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Bidirectional, Dense, Activation, Flatten, Dropout, BatchNormalization, Reshape, Permute
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.recurrent import GRU, LSTM

def build_model():
    N_CATEGORIES = 8
    BATCH_SIZE_TRAIN = 16
    BATCH_SIZE_TEST = 1
    INPUT_SHAPE = (256, 256, 3)  # x, y, color channels
    IMAGE_SIZE = (256, 256)
    KERNAL_SIZE = [(7, 7), (5,5), (4, 4), (4,4), (4,2),(2,4), (3,3), (3,3), (3,3)]
    POOL = [(2, 2), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2)]
    ACTIVATION = 'relu'
    NUM_CONV_LAYERS = 4
    FILTERS = [32, 64, 64, 128, 128, 256, 256]
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    model = Sequential()
    # model.add(BatchNormalization(axis=frequency_axis,
    #                              input_shape=INPUT_SHAPE))  # IMAGE SIZE

    # First convolution layer specifies shape
    model.add(Conv2D(FILTERS[0], KERNAL_SIZE[0], padding='same',
                    input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=POOL[0]))
    model.add(Activation(ACTIVATION))
    model.add(BatchNormalization(axis=channel_axis))

    # model.add(Dropout(0.2))

    # Add more convolutional layers
    for layer in range(NUM_CONV_LAYERS - 1):
        num = 1
        model.add(Conv2D(FILTERS[layer + 1], KERNAL_SIZE[num], padding='same'))
        # model.add(BatchNormalization(axis=channel_axis))
        model.add(MaxPooling2D(pool_size=POOL[num]))
        model.add(Activation(ACTIVATION))

        num = num + 1
        model.add(Dropout(0.3))
    
    model.add(BatchNormalization(axis=channel_axis))

    # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(LSTM(32, return_sequences=True, dropout = 0.3))
    model.add(LSTM(32, return_sequences=False, dropout = 0.3))
    #model.add(Dropout(0.1))

    # Output layer
    # model.add(Flatten())
    # model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001), 
    #                     activity_regularizer=regularizers.l1(0.001)))

    #model.add(Dropout(0.3))
    # model.add(Dense(128, activation=ACTIVATION))
    model.add(Dense(8))
    # model.add(BatchNormalization())
    model.add(Activation("softmax"))
    #model.add(Dense(N_CATEGORIES, activation='softmax'))
    # optimizers.rmsprop(lr=0.001, decay=1e-6)
    # Adam(lr=0.001, beta_1=0.5, beta_2=0.999)
    model.compile(optimizers.Adam(),
                loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    
    return model


