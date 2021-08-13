# Display a visual representation of audio file

import matplotlib.pyplot as plt
import librosa.display
import numpy as np

AUDIO_FILE = "./microphone.wav"

#sample_rate, samples = wavfile.read(AUDIO_FILE)
samples, sample_rate = librosa.load(AUDIO_FILE, sr=None)  #same but with spicy instead of librosa

# Display Mel-Spectrogram using decibel scale
sgram = librosa.stft(samples)
sgram_mag, _ = librosa.magphase(sgram)
mel_sgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate), ref=np.min)
librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')

plt.show()

# # Disply amplitude waveform
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(samples, sr=sample_rate)
# plt.show()
#
# # Display Mel-Spectrogram using amplitude scale //Fails and displays magnitude
# sgram = librosa.stft(samples)
# librosa.display.specshow(sgram)
# plt.show()
