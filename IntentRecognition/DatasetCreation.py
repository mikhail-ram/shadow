import json
import numpy as np
import pandas as pd

def make_list(file, intent):
    with open(file, "r", encoding="latin-1") as read_file:
        playlist = json.load(read_file)

    playlist = playlist[intent]
    list = []
    for i in range(len(playlist)):
        text = []
        for j in range(len(playlist[i]['data'])):
            text.append(playlist[i]['data'][j]['text'].strip())
        minilist = [" ".join(text), intent]
        list.append(minilist)
    return list

list = []
test_list = []
files = ['data/IntentRecognition/train/train_AddToPlaylist_full.json', 'data/IntentRecognition/train/train_BookRestaurant_full.json', 'data/IntentRecognition/train/train_GetWeather_full.json', 'data/IntentRecognition/train/train_PlayMusic_full.json', 'data/IntentRecognition/train/train_RateBook_full.json', 'data/IntentRecognition/train/train_SearchCreativeWork_full.json', 'data/IntentRecognition/train/train_SearchScreeningEvent_full.json']
test_files = ['data/IntentRecognition/test/validate_AddToPlaylist.json', 'data/IntentRecognition/test/validate_BookRestaurant.json', 'data/IntentRecognition/test/validate_GetWeather.json', 'data/IntentRecognition/test/validate_PlayMusic.json', 'data/IntentRecognition/test/validate_RateBook.json', 'data/IntentRecognition/test/validate_SearchCreativeWork.json', 'data/IntentRecognition/test/validate_SearchScreeningEvent.json']

for i in range(len(files)):
    list += make_list(files[i], files[i].split(sep = "_")[1])

for i in range(len(test_files)):
    test_list += make_list(test_files[i], test_files[i].split(sep = "_")[1].split(sep=".")[0])

array = pd.DataFrame(list)
test_array = pd.DataFrame(test_list)

array.to_csv('training.tsv', sep='\t', index = False, header = ['text', 'intent'])
test_array.to_csv('test.tsv', sep='\t', index = False, header = ['text', 'intent'])
