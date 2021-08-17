import librosa
import pyaudio
import wave
#define stream chunk
import soundfile

chunk = 1024

#open a wav format music
f = wave.open("C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL/neutral/neutral-1-CREMA-D.wav")

y, sr = librosa.load("C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL/neutral/neutral-1-CREMA-D.wav")


# Trim the beginning and ending silence
yt, index = librosa.effects.trim(y, top_db=30, frame_length=128, hop_length=32)
# Print the durations
print(librosa.get_duration(y), librosa.get_duration(yt))


# write the file
soundfile.write("C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL/neutral/neutral-1-CREMA-D.wav", yt, sr)
# librosa.output.write_wav("C:/Users/Khaelim/Python Projects/Datasets/AUDIO_FINAL/anger/angry-1-CREMA-D.wav", yt, sr)