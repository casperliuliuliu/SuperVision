from tkinter import ttk
import tkinter as tk
import tkmacosx
from Visualization.funcs import change_color

class Page1:
    def __init__(self, func_frame, main_app):
        self.func_frame = func_frame
        self.main_app = main_app
        self.create_tab1_widgets() 

    def create_tab1_widgets(self):
        self.test_btn = tkmacosx.Button(self.func_frame, text="TEST", command=lambda: change_color(self.test_btn, self.main_app.test_var), bg="white")
        self.test_btn.grid(row=0, column=0, padx=10)

        self.main_app.var_a=tk.Entry(self.func_frame)
        self.main_app.var_a.grid(row=1, column=0, padx=5)

        self.main_app.var_b=tk.Entry(self.func_frame)
        self.main_app.var_b.grid(row=2, column=0, padx=5)
