#! /usr/env/python3
#Author ---
#https://gist.github.com/drscotthawley/eb4ffb1ec4de29632403c1db396e419a?fbclid=IwAR0eO5BwyaR5Cs2tuqNsRyr066hxfMC0ITJHPH-hycODfqt_FB7gbj6_wVQ
#Conver the fma_small .mp3 files to .wav
import os
import librosa

import utils  
import subprocess

def mp3_Convert(num_files_to_convert):
    AUDIO_DIR = './fma_small/'

    tracks = utils.load('./fma_metadata/tracks.csv')

    try:
        os.mkdir('Samples')
    except:
        print("Sample Folder Exists. No Creation Necessary")

    small = tracks['set', 'subset'] <= 'small'

    y_small = tracks.loc[small, ('track', 'genre_top')]

    sr = 44100

    file_count = 0

    for track_id, genre in y_small.iteritems():
        if not os.path.exists('Samples/'+genre):
            os.mkdir('Samples/'+genre)

        mp3_filename = utils.get_audio_path(AUDIO_DIR, track_id)
        out_wav_filename = 'Samples/'+genre+'/'+str(track_id)+'.wav'
        in_wav_filename = out_wav_filename
        cmd = 'ffmpeg -hide_banner -loglevel panic -i ' + mp3_filename + ' ' + in_wav_filename
        print("excuting conversion: "+cmd)
        
        try:
            os.system(cmd)
        except:
            print("Could not find file:" + mp3_filename)
            continue
        
        # os.system(cmd) - we use subprocess instead of this

        ## We could just have ffmpeg do the full conversion, but we'll let librosa
        ## apply its own defaults by reading & writing
        print("reading ",in_wav_filename)
        try:
            data, sr = librosa.load(in_wav_filename, sr=sr, mono=True)
        except:
            print("Failed reading the converted in_wav:" + in_wav_filename)
            continue
        
        print("writing ",out_wav_filename)
        try:
            librosa.output.write_wav(out_wav_filename,data,sr=sr)
        except:
            print("failed writing the output_wav:"+out_wav_filename)
            continue
        
        file_count += 1

        if file_count > num_files_to_convert:
            break
    
    print(".mp3 file conversion to .wav complete")