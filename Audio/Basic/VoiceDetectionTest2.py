#Record only speech and display a visual representation //Mel Spectrogram with decibel normalisation

import speech_recognition as sr
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa

import numpy as np

AUDIO_FILE = "microphone.wav"

r = sr.Recognizer()
#print(sr.Microphone.list_microphone_names())

with sr.Microphone() as source:
    r.dynamic_energy_threshold = True
    r.adjust_for_ambient_noise(source)
    r.dynamic_energy_adjustment_ratio = 2
    print("speak now-ish")
    audio = r.listen(source)

print("hopefully it was speach")

with open("microphone.wav", "wb") as f:
    f.write(audio.get_wav_data())

#sample_rate, samples = wavfile.read(AUDIO_FILE)
samples, sample_rate = librosa.load(AUDIO_FILE, sr=16000)  #same but with librosa instead of spicy


# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(samples, sr=sample_rate)

sgram = librosa.stft(samples)
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)

mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')

plt.show()