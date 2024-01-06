

def run_video_app():
    import tkinter as tk
    from Visualization.UI.video_app import VideoApp
    root = tk.Tk()
    VideoApp(root, 0)
    root.mainloop()

def run_train_model():
    from RunModels.pprint import pprint
    from RunModels.train_config import get_config
    from RunModels.train_or_test import train_model
    pprint('he',show_time=True)
    model_things = get_config()
    print(model_things)
    train_model(model_things)

if __name__ == "__main__":
    # run_video_app()
    run_train_model()
