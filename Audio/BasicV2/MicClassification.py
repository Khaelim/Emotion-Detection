import pyaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage

# Load model
mymodel = tf.keras.models.load_model('C:/Users/Khaelim/Python Projects/Emotion-Detection/DataSets/Audio/my_audio_model', compile=True)
class_names=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


sample_len = 16000
frame_len = int(16000 * 1) # 1sec

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_len,
                input=True,
                frames_per_buffer=frame_len)

plt.ion()
while True:
    # data read
    data = stream.read(frame_len, exception_on_overflow=False)

    # byte --> float
    frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

    # Create the mel spectrogram
    sgram = librosa.stft(frame_data)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_len)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

    # Reshape image to 4 dimensional array
    new_image = tf.reshape(mel_sgram, [1, 128, 32, 1])

    # clasify mel spec
    predictions = mymodel.predict(new_image)
    
    # print classifications
    print(str("Prediction Array: "))
    print(predictions[0])
    print(str("Prediction No: "))
    print(str(class_names[np.argmax(predictions[0])]))

stream.stop_stream()
stream.close()
p.terminate()