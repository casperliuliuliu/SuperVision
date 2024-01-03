import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from .page1 import Page1
from .page2 import Page2
from ...Filters.image_processing import *
from ...Filters.edge_detection import *

class VideoApp:

    def __init__(self, root, video_source=0, width=640, height=480):
        self.root = root
        self.root.title("Video Filter App")

        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(3, width)
        self.cap.set(4, height)

        self.video_frame = tk.Frame(root)
        self.video_frame.pack(pady=10)

        self.canvas_left = tk.Canvas(self.video_frame)
        self.canvas_left.pack(side=tk.LEFT, padx=10)

        self.canvas_right = tk.Canvas(self.video_frame)
        self.canvas_right.pack(side=tk.LEFT, padx=10)

        self.func_frame = ttk.Notebook(root)
        self.func_frame.pack(pady=10)

        self.tab1_frame = ttk.Frame(self.func_frame)
        self.tab1_functions = Page1(self.tab1_frame, self)

        self.tab2_frame = ttk.Frame(self.func_frame)
        self.tab2_functions = Page2(self.tab2_frame, self)

        self.func_frame.add(self.tab1_frame, text="Page1")
        self.func_frame.add(self.tab2_frame, text="Page2")

        self.declare_variable()
        self.video_update()



    def video_update(self):
        ret, frame = self.cap.read()
        if ret:
            print(self.test_var.get())
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_left = Image.fromarray(img)
            img_left = ImageTk.PhotoImage(img_left)

            img_right = self.cool_filter(img)
            img_right = ImageTk.PhotoImage(img_right)

            # Update left canvas with the resized left image
            self.canvas_left.config(width=img_left.width(), height=img_left.height())
            self.canvas_left.create_image(0, 0, anchor=tk.NW, image=img_left)
            self.canvas_left.image = img_left  # Save reference to prevent garbage collection

            # Update right canvas with the resized right image
            self.canvas_right.config(width=img_right.width(), height=img_right.height())
            self.canvas_right.create_image(0, 0, anchor=tk.NW, image=img_right)
            self.canvas_right.image = img_right  # Save reference to prevent garbage collection
    
    def declare_variable(self):
        self.test_var = tk.BooleanVar(value=0)

    def cool_filter(self, img):
        temp_img = img.copy()
        if self.test_var.get():
            temp_img = canny_filter(temp_img, self.var_a.get(), self.var_b.get())


        self.root.after(10, self.video_update)
        img_right = Image.fromarray(temp_img)
        return img_right
    