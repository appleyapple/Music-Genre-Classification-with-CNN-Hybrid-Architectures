from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from src import clr_callback as clr
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gc
import pandas as pd
import numpy as np

from src import model
from src import load_data_generators

EPOCHS = 150

def train():
    train, validation, test = load_data_generators.load_data_gen()

    #Change the function call for other models in the file
    crnn_model = model.build_crnn_model()
    
    csv_logger = CSVLogger('log.csv', append=False, separator=';')

    STEP_SIZE_TRAIN = train.n // train.batch_size
    STEP_SIZE_VALID = validation.n // validation.batch_size
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
    #Cyclic Learning Rate https://github.com/bckenstler/CLR
    clr_method = clr.CyclicLR(mode = 'triangular', base_lr=0.0026, max_lr= 0.007, step_size= (4*STEP_SIZE_TRAIN))
    
    crnn_model.fit_generator(generator=train,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS,
                    callbacks=[csv_logger, mc, es]
                    )
    
    return crnn_model, test


def test(trained_model, test):
    STEP_SIZE_TEST = test.n // test.batch_size
    
    # Test first n sound clips in test data
    scores = trained_model.evaluate_generator(generator=test, steps=STEP_SIZE_TEST, verbose=1, workers=0)
    
    #PREDICT AGAIN AFTER!!!
    print("Evaluated Model")
    print("{}: {}, {}: {}".format(trained_model.metrics_names[0], scores[0], 
                                trained_model.metrics_names[1], scores[1]))

    trained_model.save("model.h5")

if __name__ == "__main__":
    trained_model, test_gen = train()
    test(trained_model, test_gen)