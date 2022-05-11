import os

import matplotlib.pyplot as plt
import librosa.display
import numpy as np
# from pathlib import Path
import scipy
import skimage
import tensorflow
from scipy.io import wavfile


def audio_to_image(path_to_audio, destination_directory):
    # sample_rate, samples = wavfile.read(AUDIO_FILE)
    samples, sample_rate = librosa.load(path_to_audio, sr=16000)  # same but with librosa instead of spicy

    # Display Mel-Spectrogram using decibel scale
    sgram = librosa.stft(samples)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_sgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate), ref=np.min)
    librosa.display.specshow(mel_sgram, sr=sample_rate)  # , x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    #plt.savefig(destination_directory)
    skimage.io.imsave(destination_directory, mel_sgram)


tgt_directory = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL_PROCESSED/'
dst_directory = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_AS_IMAGE/'
for dir in os.listdir(tgt_directory):
    # print(dirs)
    counter = 1
    for filename in os.listdir(tgt_directory + dir):
        print(str(counter) + ': ' + tgt_directory + dir + '/' + filename)
        file_path = tgt_directory + dir + '/' + filename
        # dst_file_path = Path(dst_directory + dir + '/' + filename)
        # dst_file_path.rename(dst_file_path.with_suffix('.png'))
        dst_file_path = dst_directory + dir + '/' + filename[:len(filename) - 4] + '.png'

        audio_to_image(file_path, dst_file_path)
        counter += 1
