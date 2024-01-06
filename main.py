def run_video_app():
    import tkinter as tk
    from Visualization.UI.video_app import VideoApp
    root = tk.Tk()
    VideoApp(root, 0)
    root.mainloop()

def run_train_model():
    import sys
    sys.path.append("D:/Casper/SuperVision/RunModels")

    from cprint import pprint
    from train_config import get_config
    from train_or_test import train_model
    pprint('',show_time=True)

    model_things = get_config()
    train_model(model_things)

if __name__ == "__main__":
    # run_video_app()
    run_train_model()
