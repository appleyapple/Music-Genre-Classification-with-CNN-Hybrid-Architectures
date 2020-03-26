#! /usr/env/python3

import os
import librosa
import librosa.display
import numpy as np
from Data_Processing import utils 
import gc

import matplotlib.pyplot as plt
from matplotlib import figure


def create_spectrogram(filename, id, folder, WIDTH, HEIGHT, DPI, SONG_DURATION, SR):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=SR, duration = SONG_DURATION)
    fig = plt.figure(figsize=[WIDTH, HEIGHT], dpi=DPI)
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    path_list = filename.split(os.sep)
    genre = path_list[2]
    if folder == "general":
        filename  = './Spectrogram/' + genre + "/" + id + '.jpg'
        plt.savefig(filename, bbox_inches='tight',pad_inches=0)
    elif folder == "test":
        filename  = './Data/Test/' + id + '.jpg'
        plt.savefig(filename, bbox_inches='tight',pad_inches=0)
    elif folder =="train":
        filename  = './Data/Train/' + id + '.jpg'
        plt.savefig(filename, bbox_inches='tight',pad_inches=0)
    else:
        filename  = './Data/Validate/' + id + '.jpg'
        plt.savefig(filename, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,clip,sample_rate,fig,ax,S

def Spectrogram_Create(WIDTH, HEIGHT, DPI, SONG_DURATION, SR):
    tracks = utils.load('./fma_metadata/tracks.csv')

    #Spectrograph training/validation/test set
    try:
        os.mkdir("Data")
    except:
        print("Data folder exists, skip creation")

    try:
        os.mkdir("Data/Train")
    except:
        print("Data/Train folder exists, skip creation")

    try:
        os.mkdir("Data/Test")
    except:
        print("Data/Test folder exists, skip creation")
    try:
        os.mkdir("Data/Validate")
    except:
        print("Data/Validate folder exists, skip creation")

    data_set = tracks['set', 'split']
    
    directory = './Samples'

    for subdir, dirs, files in os.walk(directory):
        if(subdir != directory):
            path_list = subdir.split(os.sep)
            
            print("Begin creating spectrograms for {} genre".format(path_list[2]))
            
            for file in files: 
                full_path = os.path.join(subdir, file)
                name = os.path.splitext(os.path.basename(file)) 
                id = name[0]

                if(tracks['set', 'split'][int(id)] == "test"):
                    create_spectrogram(full_path, id, "test", 
                                        WIDTH, HEIGHT, DPI, SONG_DURATION, SR)
                elif(tracks['set', 'split'][int(id)] == "training"):
                    create_spectrogram(full_path, id, "train",
                                        WIDTH, HEIGHT, DPI, SONG_DURATION, SR)
                else:
                    create_spectrogram(full_path, id, "validate",
                                        WIDTH, HEIGHT, DPI, SONG_DURATION, SR)

            print("Finished creating spectrograms for {} genre".format(path_list[2]))
    
    gc.collect()












#Organized storage of spectrograms
# try:
#     os.mkdir('Spectrogram')
# except:
#     print("Spectrogram folder exists, skip creation")


#Not for training or testing just to categorize the imgs into their own folders
# for subdir, dirs, files in os.walk(directory):
#     if(subdir != directory):
#         try:
#             path_list = subdir.split(os.sep)
#             os.mkdir("Spectrogram/" + path_list[2])
#         except:
#             print("{} folder already exists, skip creation".format(path_list[2]) )
        
#         print("Begin creating spectrograms for {} genre".format(path_list[2]))
#         for file in files:
            
#             full_path = os.path.join(subdir, file)
#             name = os.path.splitext(os.path.basename(file)) 
#             id = name[0]
#             create_spectrogram(full_path, id, "general")

#         print("Finished creating spectrograms for {} genre".format(path_list[2]))