#! /usr/env/python3
import os
import shutil
from Data_Processing import conversion
from Data_Processing import spect_create
from Data_Processing import data_csv_gen
from Data_Processing import utils
from dotenv import load_dotenv

dotenv_path = "./setup.env" #Setup environment file
load_dotenv(dotenv_path)

#number of files you want to convert to .wav, this
#also affects the number of spectrograms you create
FILES_TO_GENERATE = int(os.environ.get("FILES_TO_GENERATE"))
SONG_DURATION = float(os.environ.get("SONG_DURATION"))
SPECTROGRAM_WIDTH = float(os.environ.get("SPECTROGRAM_WIDTH")) #in inches
SPECTROGRAM_HEIGHT = float(os.environ.get("SPECTROGRAM_HEIGHT")) #in inches
SPECTROGRAM_DPI = int(os.environ.get("SPECTROGRAM_DPI"))
SONG_SAMPLING_RATE = int(os.environ.get("SONG_SAMPLING_RATE"))

print("Beginning Setup")
try:
    shutil.rmtree("./Data/")
except:
    print("Data folder does not exist to delete")
    
conversion.mp3_Convert(FILES_TO_GENERATE)

spect_create.Spectrogram_Create(SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT,
                                SPECTROGRAM_DPI, SONG_DURATION, SONG_SAMPLING_RATE)

data_csv_gen.Generate_CSV()

print("Removing Samples Folder")
shutil.rmtree("./Samples/")

print("Setup Complete")