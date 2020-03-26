import librosa
import os

from Data_Processing import utils  
import subprocess

import librosa.display
import numpy as np
import gc

import matplotlib.pyplot as plt
from matplotlib import figure
from csv import writer

def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)

        csv_writer.writerow(list_of_elem)

def create_spectrogram(filename, id, folder, WIDTH, HEIGHT, DPI, SR):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=SR)
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
    elif folder == "train":
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

def Pitch_Shift_Convert(pitch):
    
    AUDIO_DIR = './Samples/'
    tracks = utils.load('./fma_metadata/tracks.csv')

    try:
        os.mkdir('./AugmentedData/Pitch')
    except:
        print("Pitch Folder Exists. No Creation Necessary")
    
    
    for subdir, dirs, files in os.walk(AUDIO_DIR):
        path_list = subdir.split(os.sep)
                
        for file in files: 
            full_path = os.path.join(subdir, file)
            name = os.path.splitext(os.path.basename(file)) 
            id = name[0]
            if(tracks['set', 'split'][int(id)] == "training"):
                y, sr = librosa.load(full_path)
                y_one = librosa.effects.pitch_shift(y, sr, n_steps=pitch)
                out_wav_filename = './AugmentedData/Pitch/' + id + '.wav'
                librosa.output.write_wav(out_wav_filename, y_one,sr=sr)


def Tempo_Change_Convert(tempo):
    AUDIO_DIR = './Samples/'

    tracks = utils.load('./fma_metadata/tracks.csv')

    try:
        os.mkdir('./AugmentedData/Tempo')
    except:
        print("Tempo Folder Exists. No Creation Necessary")
    
    
    for subdir, dirs, files in os.walk(AUDIO_DIR):
        path_list = subdir.split(os.sep)
                
        for file in files: 
            full_path = os.path.join(subdir, file)
            name = os.path.splitext(os.path.basename(file)) 
            id = name[0]
            if(tracks['set', 'split'][int(id)] == "training"):
                y, sr = librosa.load(full_path)
                y_speed = librosa.effects.time_stretch(y, tempo)
                out_wav_filename = './AugmentedData/Tempo/' + id + '.wav'
                librosa.output.write_wav(out_wav_filename, y_speed,sr=sr)

def Augmented_Spectrograms(WIDTH, HEIGHT, DPI, SONG_DURATION, SR):
    directory = './AugmentedData'

    tracks = utils.load('./fma_metadata/tracks.csv')

    #STARTING ID BE SURE TO MAKE SURE NO CONFLICTS
    augmented_id = 200000

    for subdir, dirs, files in os.walk(directory):
        path_list = subdir.split(os.sep)
                
        for file in files: 
            full_path = os.path.join(subdir, file)
            name = os.path.splitext(os.path.basename(file)) 
            id = name[0]
            if(tracks['set', 'split'][int(id)] == "training"):

                new_row = [augmented_id, tracks['track', 'genre_top'][int(id)]]

                create_spectrogram(full_path, str(augmented_id), "train",
                                    WIDTH, HEIGHT, DPI, SR)
                
                append_list_as_row('./Data/train.csv', new_row)

                augmented_id += 1


def Augment(WIDTH, HEIGHT, DPI, SONG_DURATION, SR, PITCH, TEMPO):
    try:
        os.mkdir('AugmentedData')
    except:
        print("AugmentedData Exists. No Creation Necessary")

    print("Starting Pitch augment")
    Pitch_Shift_Convert(PITCH)
    print("Done pitch")

    print("Starting tempo change")
    Tempo_Change_Convert(TEMPO)
    print("Done tempo change")

    try:
        os.mkdir('./Data/Train/')

    except:
        print("train exists, no creation")

    print("Spectrogram generation")
    Augmented_Spectrograms(WIDTH, HEIGHT, DPI, SONG_DURATION, SR)
