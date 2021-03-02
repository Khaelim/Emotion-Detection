import tkinter as tk


def open_video():
    exec(open("ReFresh.py").read())
def open_audio():
    #exec(open("ReFresh.py").read())
    print("ToDo")
def open_audio_video():
    #exec(open("ReFresh.py").read())
    print("ToDo")
root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

video = tk.Button(frame,
                   text="Video",
                   command=open_video)
video.pack(side=tk.LEFT)

audio = tk.Button(frame,
                    text="Audio",
                    command=open_audio)
audio.pack(side=tk.LEFT)

audioVideo = tk.Button(frame,
                    text="Audio + Video",
                    command=open_audio_video)
audioVideo.pack(side=tk.LEFT)

quit = tk.Button(frame,
                   text="Quit",
                   fg="red",
                   command=quit)
quit.pack(side=tk.LEFT)

root.mainloop()