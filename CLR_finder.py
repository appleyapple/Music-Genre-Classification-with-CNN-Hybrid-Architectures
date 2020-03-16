from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from src import model 
from src import load_data_generators
from src import clr_callback as clr

GRAPH_EPOCH = 4
BASE_LR = 0.0001
MAX_LR = 0.1

def find_clr():
    train, validation, test = load_data_generators.load_data_gen()

    #Change the function call for other models in the file
    crnn_model = model.build_model()
    
    STEP_SIZE_TRAIN = train.n // train.batch_size
    
    #Cyclic Learning Rate https://github.com/bckenstler/CLR
    clr_method = clr.CyclicLR(mode='triangular', base_lr=BASE_LR, max_lr= MAX_LR, step_size= (GRAPH_EPOCH*STEP_SIZE_TRAIN))
    
    crnn_model.fit_generator(generator=train,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=GRAPH_EPOCH,
                    callbacks=[clr_method]
                    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    h = clr_method.history

    lr = h['lr']
    acc = h['accuracy']

    ax.plot(lr, acc)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.7f'))

    plt.show()

if __name__ == "__main__":
    find_clr()