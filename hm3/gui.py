import tkFileDialog as filedialog
from Tkinter import *
from PIL import ImageTk, Image
from shutil import copyfile
import os
from pathlib import Path
import time
import hmw3


window = Tk()
window.title('Modena Skyline Segmentation')
window.geometry('{0}x{1}+0+0'.format(window.winfo_screenwidth(), window.winfo_screenheight()))

panel = Label(window)
panel2 = Label(window)
panel.grid(row=1, column=0)
panel2.grid(row=1, column=1)


def search_function():
    filename = filedialog.askopenfilename()
    img = ImageTk.PhotoImage(Image.open(filename))
    panel.config(image=img)
    panel.image = img

    itp = './tmp/orig/tmp.png'
    itpp = Path(itp)

    if itpp.exists():
        os.remove(itp)
        print('exists')
    copyfile(filename, itp)


def segmentation():

    ipl = './tmp/labeled/tmp.png'
    iplp = Path(ipl)
    if iplp.exists():
        os.remove(ipl)
        print('exists')

    hmw3.main_app('./tmp/orig/tmp.png', './tmp/labeled/tmp.png', False)

    while not iplp.exists():
        time.sleep(5)
    img = ImageTk.PhotoImage(Image.open('./tmp/labeled/tmp.png'))

    panel2.config(image=img)
    panel2.image = img


def on_closing():
    window.destroy()


browsebutton = Button(window, text='Search...', command=search_function)
browsebutton.grid(row=0, column=0)
browsebutton1 = Button(window, text='Do Segmentation', command=segmentation)
browsebutton1.grid(row=0, column=1)
window.protocol('WM_DELETE_WINDOW', on_closing)
mainloop()
