from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from shutil import copyfile
import os
from os import getcwd,chdir
from pathlib import Path
import time
import subprocess as sp

GP = "/home/fede/Desktop/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation-master/"
VE = "/home/fede/.virtualenvs/vp3/bin/python"

window = Tk()
window.title("Modena Skyline Segmentation")
window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
#window.configure(background='gray')
#panel1 =
panel = Label(window)
panel2 = Label(window)
fname =''
# The Pack geometry manager packs widgets in rows or columns.
#panel.pack(side=LEFT, fill="both", expand="yes")
#panel2.pack(side=LEFT, fill="both", expand="yes")
panel.grid(row=1, column=0)
panel2.grid(row=1, column=1)

def browsefunc():
    filename = filedialog.askopenfilename()
    #pathlabel.config(text=filename)
    img = ImageTk.PhotoImage(Image.open(filename))
    fname = filename
    panel.config(image=img)
    panel.image = img

    #itp = image to predict
    itp = os.path.join(GP, "pred_img/tmp.png")
    itpp = Path(itp)
    print('ciao')
    if itpp.exists():
        os.remove(itp)
        print('esiste')
    copyfile(filename, itp)



def pro():
    prev = getcwd()
    chdir(GP)

    ipl = os.path.join(GP, "prediciton/Label/tmp.png")
    iplp = Path(ipl)
    if iplp.exists():
        os.remove(ipl)
        print('esiste')

    ipo = os.path.join(GP, "prediciton/OverLay/tmp.png")
    ipop = Path(ipo)
    if ipop.exists():
        os.remove(ipo)
        print('esiste')



    sp.Popen([VE, 'Inference.py'])



    my_file = Path("prediciton/OverLay/tmp.png")
    while not my_file.exists():
        time.sleep(5)
    img = ImageTk.PhotoImage(Image.open("prediciton/OverLay/tmp.png"))

    panel2.config(image=img)
    panel2.image = img


browsebutton = Button(window, text="Search...", command=browsefunc)
#browsebutton.pack(side=BOTTOM)
browsebutton.grid(row=0, column=0)
browsebutton1 = Button(window, text="Do Segmentation", command=pro)
browsebutton1.grid(row=0, column=1)


#pathlabel = Label(window)
#pathlabel.pack()


def on_closing():
        window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)



mainloop()