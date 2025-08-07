import os
from tkinter import *
from tkinter import messagebox
from tkinter import *

from tkinter import messagebox, filedialog
import cv2


import matplotlib.pyplot as plt

import tk as tk
from PIL import Image, ImageTk
import tkinter as tk
from cvlib.object_detection import draw_bbox
from dataupload import ViewData
global act1
def act():
    x = Admin.get()
    y = password.get()
    if x.__eq__("Admin") and y.__eq__("Admin"):
        s1 = "login successfully"
        messagebox.showinfo(" Success", s1)
        winadmin.destroy()
        land = ViewData()
        # exec(open("dataupload.py").read())


    else:
        messagebox.showinfo("login failed")


winadmin = Tk()
winadmin.title("Dental Caries Detection")
winadmin.maxsize(width=1100, height=1000)
winadmin.minsize(width=1100, height=1000)
winadmin.configure(bg='#34bfbb')

image1 = Image.open('1.jpg')
img = image1.resize((1100, 1000))

test = ImageTk.PhotoImage(img)

label1 = tk.Label(winadmin, image=test)
label1.image = test

# Position image
label1.place(x=1, y=1)

# image1 = Image.open("3.png")
test = ImageTk.PhotoImage(img)

label1 = tk.Label(winadmin, image=test)
label1.image = test

# Create Canvas
# canvas1 = Canvas(win, width=400, height=400)

# canvas1.pack(fill="both", expand=True)

# Display image
# canvas1.create_image(0, 0, image=bg, anchor="nw")

Label(winadmin, text='Dental Caries Detection', bg="#ffb366", font='verdana 15 bold') \
    .place(x=200, y=120)

Admin = Label(winadmin, text="Admin", bg="#34bfbb", width=10, font='Verdana 10 bold')
Admin.place(x=200, y=320)

password = Label(winadmin, text="password", bg="#34bfbb", width=10, font='Verdana 10 bold')
password.place(x=200, y=370)

# Entry Box ------------------------------------------------------------------

Admin = StringVar()
password = StringVar()

Admin = Entry(winadmin, width=30, bg="silver", show='*', textvariable=Admin)
Admin.place(x=400, y=370)

password = Entry(winadmin, width=30, bg="silver", textvariable=password)
password.place(x=400, y=320)

Button(winadmin, text="login", font='Verdana 10 bold', bg="#34bfbb", command=act).place(x=300, y=520)
winadmin.mainloop()


