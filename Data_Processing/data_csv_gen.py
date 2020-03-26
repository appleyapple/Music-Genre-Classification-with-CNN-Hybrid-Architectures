from Data_Processing import utils
import os
import pandas as pd


def Generate_CSV():
    print("Begin CSV generation")

    tracks = utils.load('./fma_metadata/tracks.csv')
    #genres = utils.load('./fma_metadata/genres.csv')
    #features = utils.load('./fma_metadata/features.csv')
    #echonest = utils.load('./fma_metadata/echonest.csv')
    #print(tracks['track', 'genre_top'][1482])

    try:
        os.mkdir("Data")
    except:
        print("Data folder exists, skip creation")

    #Creating a list of ids that are in FMA_SMALL, so that we only create a csv file for data we have
    fma_small_track_id = []

    directory = './Data'

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            name = os.path.splitext(os.path.basename(file)) 
            id = name[0]
            fma_small_track_id.append(id)

    data_top = tracks['set', 'split'].head(len(tracks['set', 'split']))


    df_train = pd.DataFrame(columns=['ID', 'Genre'])
    df_validation = pd.DataFrame(columns=['ID', 'Genre'])
    df_test = pd.DataFrame(columns=['ID', 'Genre'])

    for id in data_top.index:
        if str(id) in fma_small_track_id:
            # if(tracks['set', 'split'][id] == "training"):
            #     print("Adding Training Data", id)
            #     df_train = df_train.append({'ID': id, 'Genre': tracks['track', 'genre_top'][id]}, ignore_index = True)
            if(tracks['set', 'split'][id] == "test"):
                print("Adding Testing Data", id)
                df_test = df_test.append({'ID': id, 'Genre': tracks['track', 'genre_top'][id]}, ignore_index = True)
            elif(tracks['set', 'split'][id] == "training"):
                print("Adding Training Data", id)
                df_train = df_train.append({'ID': id, 'Genre': tracks['track', 'genre_top'][id]}, ignore_index = True)
            else:
                print("Adding Validation  Data", id)
                df_validation= df_validation.append({'ID': id, 'Genre': tracks['track', 'genre_top'][id]}, ignore_index = True)

    df_train.to_csv('Data/train.csv', encoding='utf-8', index= False, header=False)
    df_validation.to_csv('Data/validate.csv', encoding='utf-8', index= False, header=False)
    df_test.to_csv('Data/test.csv', encoding='utf-8', index= False, header=False)

    print("Completed CSV generation, check the ./Data folder for your .csv")