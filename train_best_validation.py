from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from src import clr_callback as clr
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential, Model, load_model

from src import model
from src import load_data_generators
from src import clr_callback as clr

EPOCHS = 1000



import pandas as pd
import numpy as np

BEST_MODEL_FILENAME = "./best_model.h5"

def train_more():
    train, validation, test = load_data_generators.load_data_gen()
    csv_logger = CSVLogger('log_trainedmore.csv', append=False, separator=';')
    crnn_model = load_model(BEST_MODEL_FILENAME)

    # Train and validate model
    STEP_SIZE_TRAIN = train.n // train.batch_size
    STEP_SIZE_VALID = validation.n // validation.batch_size

    #early stopping https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    mc = ModelCheckpoint('more_trained_earlystop.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    #Cyclic Learning Rate https://github.com/bckenstler/CLR
    clr_method = clr.CyclicLR(mode='triangular2', base_lr=0.0001, max_lr= 0.013, step_size= (4*STEP_SIZE_TRAIN))

    crnn_model.fit_generator(generator=train,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=validation,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=1000,
                        callbacks=[csv_logger, clr_method, mc, es]
                        )


    # ------------------------- TESTING ------------------------- #
    STEP_SIZE_TEST = test.n // test.batch_size
    
    # Test first n sound clips in test data
    scores = crnn_model.evaluate_generator(generator=test, steps=STEP_SIZE_TEST, verbose=1, workers=0)
    
    print("Evaluated Model")
    print("{}: {}, {}: {}".format(crnn_model.metrics_names[0], scores[0], 
                                crnn_model.metrics_names[1], scores[1]))

    model.save("more_trained_best_final.h5")

if __name__ == "__main__":
    train_more()