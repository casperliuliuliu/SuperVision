import tkmacosx
from Visualization.funcs import change_color

class Page2:
    def __init__(self, func_frame, main_app):
        self.func_frame = func_frame
        self.main_app = main_app
        self.create_tab1_widgets() 

    def create_tab1_widgets(self):
        self.ori_btn = tkmacosx.Button(self.func_frame, text="app2", command=lambda: change_color(self.ori_btn, self.main_app.test_var), bg="white")
        self.ori_btn.grid(row=0, column=0, padx=10)
        
