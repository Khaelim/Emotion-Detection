import os
import librosa

#  os.remove("ChangedFile.csv")
#  librosa.get_duration(filename='my.wav')
directory = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL_PROCESSED/'
counter = 1

for dirs in os.listdir(directory):
    #print(dirs)

    for filename in os.listdir(directory + dirs):
        print(str(counter) + ': ' + directory + dirs + '/' + filename)
        file_path = "directory + dirs + '/' + filename"
        counter += 1
        #trim(directory + dirs + '/', save_folder + dirs + '/', )
        librosa.get_duration(filename=file_path)