import os

import librosa
import soundfile


def trim(path_to_file, path_to_destination):
    y, sr = librosa.load(path_to_file)
    yt, index = librosa.effects.trim(y, top_db=30, frame_length=128, hop_length=32)
    soundfile.write(path_to_destination, yt, sr)
    print(librosa.get_duration(y), librosa.get_duration(yt))


save_folder = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL_PROCESSED/'
directory = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL/'
counter = 1

for dirs in os.listdir(directory):
    # print(dirs)

    for filename in os.listdir(directory + dirs):
        print(str(counter) + ': ' + directory + dirs + '/' + filename)
        counter += 1
        trim(directory + dirs + '/' + filename, directory + dirs + '/' + filename)

## open a wav format music
# f = wave.open("C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL/neutral/neutral-1-CREMA-D.wav")
# y, sr = librosa.load("C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL/neutral/neutral-1-CREMA-D.wav")

## Trim the beginning and ending silence
# yt, index = librosa.effects.trim(y, top_db=30, frame_length=128, hop_length=32)
## Print the durations
# print(librosa.get_duration(y), librosa.get_duration(yt))

# write the file
# soundfile.write("C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL/neutral/neutral-1-CREMA-D.wav", yt, sr)
