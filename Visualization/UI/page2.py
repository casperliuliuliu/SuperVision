from tkinter import ttk

class Page2:
    def __init__(self, func_frame, main_app):
        self.func_frame = func_frame
        self.main_app = main_app
        self.create_tab2_widgets()

    def create_tab2_widgets(self):
        self.ori_btn = ttk.Button(self.func_frame, text="Haha", command=self.haha)
        self.ori_btn.grid(row=0, column=0, padx=10)

    def haha(self):
        print("haha")
