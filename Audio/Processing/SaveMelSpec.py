# Display a visual representation of audio file
import os

import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import scipy
import tensorflow
from matplotlib.backends.backend_template import FigureCanvas
from scipy.io import wavfile


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def audio_to_image(path_to_audio):
    # sample_rate, samples = wavfile.read(AUDIO_FILE)
    samples, sample_rate = librosa.load(path_to_audio, sr=None)  # same but with librosa instead of spicy

    # # get some data  *****most of this is unnecessary*****
    # sample_rate, wav_data = wavfile.read(path_to_audio, 'rb')
    # sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    # # Show some basic information about the audio.
    # duration = len(wav_data) / sample_rate
    # print(f'Sample rate: {sample_rate} Hz')
    # print(f'Total duration: {duration:.2f}s')
    # print(f'Size of the input: {len(wav_data)}')
    # # normalise waveform
    # waveform = wav_data / tensorflow.int16.max
    # # plot the waveform
    # plt.plot(waveform)
    # plt.xlim([0, len(waveform)])
    # plt.show()

    # Display Mel-Spectrogram using decibel scale
    # sgram = librosa.stft(samples)
    # sgram_mag, _ = librosa.magphase(sgram)
    # mel_sgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate), ref=np.min)
    # librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')

    #plt.show()
    window_size = 1024
    window = np.hanning(window_size)
    stft = librosa.core.spectrum.stft(samples, n_fft=window_size, hop_length=512, window=window)
    out = 2 * np.abs(stft) / np.sum(window)

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax)#, y_axis='log', x_axis='time')
    fig.savefig('spec.png')


    # # Disply amplitude waveform
    # plt.figure(figsize=(14, 5))
    # librosa.display.waveplot(samples, sr=sample_rate)
    # plt.show()
    #
    # # Display Mel-Spectrogram using amplitude scale //Fails and displays magnitude
    # sgram = librosa.stft(samples)
    # librosa.display.specshow(sgram)
    #
    # plt.show()


directory = 'C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL_PROCESSED/'

for dir in os.listdir(directory):
    # print(dirs)
    counter = 1
    for filename in os.listdir(directory + dir):
        print(str(counter) + ': ' + directory + dir + '/' + filename)
        file_path = directory + dir + '/' + filename

        audio_to_image(file_path)
        counter += 1
        break
    break
