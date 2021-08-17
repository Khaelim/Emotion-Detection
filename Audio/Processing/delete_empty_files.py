import os
import librosa

#  os.remove("ChangedFile.csv")
#  librosa.get_duration(filename='my.wav')
directory = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL_PROCESSED/'
counter = 1

for dir in os.listdir(directory):
    #print(dirs)

    for filename in os.listdir(directory + dir):
        print(str(counter) + ': ' + directory + dir + '/' + filename)
        file_path = directory + dir + '/' + filename
        counter += 1
        #trim(directory + dirs + '/', save_folder + dirs + '/', )
        print(librosa.get_duration(filename=file_path))
        if librosa.get_duration(filename=file_path)<1:
            os.remove(file_path)
            print("Deleted: "+file_path)

    print("Finished: " + dir)

print("done! " )