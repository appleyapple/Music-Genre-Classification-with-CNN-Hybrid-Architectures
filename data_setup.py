#! /usr/env/python3

from Data_Processing import conversion
from Data_Processing import spect_create
from Data_Processing import data_csv_gen
from Data_Processing import utils

#number of files you want to convert to .wav, this
#also affects the number of spectrograms you create
FILES_TO_GENERATE = 50

print("Beginning Setup")
conversion.mp3_Convert(FILES_TO_GENERATE)
spect_create.Spectrogram_Create()
data_csv_gen.Generate_CSV()
print("Setup Complete")