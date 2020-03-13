from keras import regularizers, optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Reshape, Permute
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.recurrent import GRU, LSTM

# from keras.layers.normalization import BatchNormalization

import pandas as pd
import numpy as np


def append_ext(fn):
    return fn+".jpg"


BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
N_CATEGORIES = 8
IMAGE_SIZE = (128, 128)

# Data flow [img_name, genre]
traindf = pd.read_csv('./Data/train.csv',
                      names=["ID", "Class"], dtype=str)
testdf = pd.read_csv('./Data/test.csv',
                     names=["ID", "Class"], dtype=str)

# Appending jpg to end of img_name
traindf["ID"] = traindf["ID"].apply(append_ext)
testdf["ID"] = testdf["ID"].apply(append_ext)

datagen = ImageDataGenerator(rescale=1./255., validation_split=0.25)

# Train generator to feed training data to CNN
train_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./Data/Train/",
    x_col="ID",
    y_col="Class",
    subset="training",
    batch_size=BATCH_SIZE_TRAIN,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=IMAGE_SIZE)

# Validation generator to feed validation data to CNN
valid_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./Data/Train/",
    x_col="ID",
    y_col="Class",
    subset="validation",
    batch_size=BATCH_SIZE_TRAIN,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=IMAGE_SIZE)

# Build CRNN model
# Conv2D - feature extraction using filters (passing filter over sections of image)
# Dropout layer - randomly drop nodes; reduces overfitting
# Max Pooling - taking maximum value of "window" instead of all values within it; reduce data size
# Flatten layer - Converts n-dimension array to 1D array for NN

frequency_axis = 1
time_axis = 2
channel_axis = 3

model = Sequential()
# model.add(BatchNormalization(axis=frequency_axis,
#                              input_shape=(128, 128, 3)))  # IMAGE SIZE

# First convolution layer specifies shape
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(128, 128, 3)))  # IMAGE_SIZE
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Add more convolutional layers
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# Reshaping input for recurrent layer
# (frequency, time, channels) --> (time, frequency, channel)
model.add(Permute((time_axis, frequency_axis, channel_axis)))
resize_shape = model.output_shape[2] * model.output_shape[3]
model.add(Reshape((model.output_shape[1], resize_shape)))

# recurrent layer
model.add(GRU(32, return_sequences=True))
model.add(GRU(32, return_sequences=False))
model.add(Dropout(0.3))

# Output layer
# model.add(Flatten())
model.add(Dense(512))
model.add(Activation("softmax"))
model.add(Dropout(0.5))
model.add(Dense(N_CATEGORIES, activation='softmax'))

model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# ------------------------- TRAINING ------------------------- #

# Train and validate model
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=100
                    )
model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)


# ------------------------- TESTING ------------------------- #

# Test generator to feed test data to CNN
test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory="./Data/Test/",
    x_col="ID",
    y_col="Class",
    batch_size=BATCH_SIZE_TEST,
    class_mode="categorical",
    target_size=IMAGE_SIZE)
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

# Test first n sound clips in test data
test_generator.reset()
pred = model.predict_generator(test_generator,
                               steps=STEP_SIZE_TEST,
                               verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

# Fetch labels from train_gen & set predictions into 1D array
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# Calculate test accuracy
count = 0
for i, genre in enumerate(predictions):
    if genre == testdf["Class"][i]:
        count += 1

# Display results
print(testdf["Class"])
print(predictions[:])
print("Number of correct categorizations: ", count)
print("Test set accuracy: ", count/len(predictions))
