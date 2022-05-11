import pyaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

sample_len = 16000
frame_len = int(16000 * 1) # 1sec

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_len,
                input=True,
                frames_per_buffer=frame_len)

cnt = 0
plt.ion()
while True:
    # data read
    data = stream.read(frame_len, exception_on_overflow=False)

    # byte --> float
    frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

    sgram = librosa.stft(frame_data)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_len)

    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    librosa.display.specshow(mel_sgram, sr=sample_len, x_axis='time', y_axis='mel')

    # visualize input audio
    #plt.imshow(mel_sgram, cmap='jet', aspect='auto', origin='lower')
    plt.pause(0.001)
    plt.show()

    # print dix
    print(cnt)
    cnt += 1

stream.stop_stream()
stream.close()
p.terminate()