import speech_recognition as sr
import sounddevice as sd

r = sr.Recognizer()
#print(sr.Microphone.list_microphone_names())

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    print("Begin speaking")
    audio = r.listen(source)

print("End of speech")

# # write audio to a RAW file
# with open("microphone.raw", "wb") as f:
#     f.write(audio.get_raw_data())
#
# write audio to a WAV file
with open("microphone.wav", "wb") as f:
    f.write(audio.get_wav_data())

# # write audio to an AIFF file
# with open("microphone.aiff", "wb") as f:
#     f.write(audio.get_aiff_data())
#
# # write audio to a FLAC file
# with open("microphone.flac", "wb") as f:
#     f.write(audio.get_flac_data())

