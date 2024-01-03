import tkinter as tk
from Visualization.UI.video_app import VideoApp

def run_video_app():
    root = tk.Tk()
    VideoApp(root, 0)
    root.mainloop()

if __name__ == "__main__":
    run_video_app()
