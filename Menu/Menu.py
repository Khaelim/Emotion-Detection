import tkinter as tk


def open_video():
    exec(open("ReFresh.py").read())


root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

video = tk.Button(frame,
                   text="Video")#,
                   #fg="red"),
                   #command=quit)
video.pack(side=tk.LEFT)

audio = tk.Button(frame,
                    text="Audio",
                    command=open_video)
audio.pack(side=tk.LEFT)

audioVideo = tk.Button(frame,
                   text="Audio + Video")#,
                   #fg="red")#,
                   #command=write_slogan)
audioVideo.pack(side=tk.LEFT)

quit = tk.Button(frame,
                   text="Quit",
                   fg="red",
                   command=quit)
quit.pack(side=tk.LEFT)

root.mainloop()