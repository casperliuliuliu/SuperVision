from tkinter import ttk

class Page1:
    def __init__(self, func_frame, main_app):
        self.func_frame = func_frame
        self.main_app = main_app
        self.create_tab1_widgets() 

    def create_tab1_widgets(self):
        self.ori_btn = ttk.Button(self.func_frame, text="Hehe", command=self.hehe)
        self.ori_btn.grid(row=0, column=0, padx=10)

    def hehe(self):
        print("hehe")
