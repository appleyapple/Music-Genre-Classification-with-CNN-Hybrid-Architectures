#! /usr/env/python3

import conversion
import spect_create
import data_csv_gen

#number of files you want to convert to .wav, this
#also affects the number of spectrograms you create
FILES_TO_GENERATE = 50

print("Beginning Setup")
conversion.mp3_Convert(FILES_TO_GENERATE)
spect_create.Spectrogram_Create()
data_csv_gen.Generate_CSV()
print("Setup Complete")