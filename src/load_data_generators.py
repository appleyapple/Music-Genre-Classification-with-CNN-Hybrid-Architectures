import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 1

IMAGE_SIZE = (256, 256)
VALDATION_SPLIT = 0.25

def append_ext(fn):
    return fn+".jpg"

def load_train_gen(traindf):
    datagen = ImageDataGenerator(rescale=1./255., validation_split = VALDATION_SPLIT)
    
    # Train generator to feed training data to CNN
    train_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory="./Data/Train/",
        x_col="ID",
        y_col="Class",
        subset="training",
        batch_size=BATCH_SIZE_TRAIN,
        seed=52,
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
        seed=52,
        shuffle=True,
        class_mode="categorical",
        target_size=IMAGE_SIZE)

    return train_generator, valid_generator

def load_test_gen(testdf):
    # Test generator to feed test data to CNN
    test_datagen = ImageDataGenerator(rescale=1./255.)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=testdf,
        directory="./Data/Test/",
        x_col="ID",
        y_col="Class",
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        class_mode="categorical",
        target_size=IMAGE_SIZE)
    
    return test_generator

def load_data_gen():
    # Data flow [img_name, genre]
    traindf = pd.read_csv('./Data/train.csv',
                        names=["ID", "Class"], dtype=str)
    testdf = pd.read_csv('./Data/test.csv',
                        names=["ID", "Class"], dtype=str)
   
    # Appending jpg to end of img_name
    traindf["ID"] = traindf["ID"].apply(append_ext)
    testdf["ID"] = testdf["ID"].apply(append_ext)

    train_gen, valid_gen = load_train_gen(traindf)
    test_gen = load_test_gen(testdf)

    return train_gen, valid_gen, test_gen




