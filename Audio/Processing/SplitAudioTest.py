import os

from pydub import AudioSegment
import math


class SplitWavAudioMubin():
    def __init__(self, folder, dest_folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + filename
        self.dest_folder = dest_folder

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.dest_folder + split_filename, format="wav")

    def multiple_split(self, min_per_split):
        total_secs = math.ceil(self.get_duration() / 60 * 60)
        for i in range(0, total_secs, min_per_split):

            index = self.filename.find('.wav')
            split_fn = filename[:index] + '-' + str(i) + filename[index:]
            # split_fn = str(i) + '_' + self.filename
            self.single_split(i, i + min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_secs - min_per_split:
                print('All splited successfully')


# change to fit current filesystem
folder = ''
save_folder = 'C:/temp1/'

directory = 'C:/temp/'
print(directory)
counter = 1
for dirs in os.listdir(directory):
    #print(dirs)

    for filename in os.listdir(directory + dirs):
        print(str(counter) + ': ' + directory + dirs + '/' + filename)
        counter += 1
        # file = filename
        # split_wav = SplitWavAudioMubin(directory + dirs + '/', save_folder + dirs + '/', file)
        # split_wav.multiple_split(min_per_split=1)


"""Print all filenames"""
# for dirs in os.listdir(directory):
#     counter = 1
#     for filename in os.listdir(directory + dirs):
#         print(str(counter)+ ': ')
#         print(os.path.join(directory, dirs, filename))
#         counter += 1


"""TEST"""

# folder = '../Basic'
# file = 'microphone.wav'
# split_wav = SplitWavAudioMubin(folder, file)
# split_wav.multiple_split(min_per_split=1)
