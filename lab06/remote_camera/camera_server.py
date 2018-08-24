'''
    Simple socket server using threads
'''

import socket
import sys

import threading
list1Lock = threading.Lock()

import cv2
import urllib
import numpy as np
HOST = ''   # Symbolic name meaning all available interfaces
PORT = 8084 # Arbitrary non-privileged port

img_list = []

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

#Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()

print('Socket bind complete')

#Start listening on socket
s.listen(10)
print('Socket now listening')

def camerathread():
    stream = cv2.VideoCapture(0)
    img = None
    frame = 0
    while True:
            retval, img = stream.read()
            try:
                list1Lock.acquire()
                img_list.append(img)
                if len(img_list) > 1:
                    img_list.pop(0)
            finally:
                list1Lock.release()
            #print("\rCamera. nimg="+str(len(img_list)) + "  ")
            frame += 1
            if cv2.waitKey(1) == 27:
                break

#Function for handling connections. This will be used to create threads
def clientthread(conn, nclient):
    #Sending message to connected client
    conn.send(b'HTTP/1.0 200 OK\r\nContent-Type: multipart/x-mixedreplace;boundary=myboundary\r\n\r\n')

    #infinite loop so that function do not terminate and thread do not end.
    while True:
        if len(img_list) > 0:
            try:
                list1Lock.acquire()
                img = img_list[0]
            finally:
                list1Lock.release()
            retval, im_buf = cv2.imencode('.jpg', img)
            im_buf = np.array(im_buf).tostring()
            reply = b'--myboundary\r\nContent-Type: image/jpeg\r\nContent-Length: '+bytes(len(im_buf))+b'\r\n\r\n'+im_buf+b'\r\n'
            conn.sendall(reply)
        else:
            print("Client " + str(nclient) + ". Images finished")
            break

    #came out of loop
    conn.close()

threading.Thread(target=camerathread).start()
nclient = 0
while 1:
    #wait to accept a connection - blocking call
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))

    #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
    #start_new_thread(clientthread ,(conn,nclient, ))
    threading.Thread(target=clientthread, args=(conn, nclient)).start()
    nclient += 1
s.close()