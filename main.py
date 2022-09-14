import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk

window = tk.Tk()
window.title("STUDENT ATTENDANCE USING FACE RECOGNITION SYSTEM")
window.geometry('800x500')



dialog_title = 'QUIT'
dialog_text = "are you sure?"
window.configure(background='green')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

label1 = tk.Label(window, background="green", fg="black", text="Name :", width=10, height=1,
                  font=('Helvetica', 16))
label1.place(x=83, y=40)
std_name = tk.Entry(window, background="yellow", fg="black", width=25, font=('Helvetica', 14))
std_name.place(x=280, y=41)
label2 = tk.Label(window, background="green", fg="black", text="Matric Number :", width=14, height=1,
                  font=('Helvetica', 16))
label2.place(x=100, y=90)
std_number = tk.Entry(window, background="yellow", fg="black", width=25, font=('Helvetica', 14))
std_number.place(x=280, y=91)

clearBtn1 = tk.Button(window, background="red",  fg="white", text="CLEAR", width=8, height=1,
                      activebackground="red", font=('Helvetica', 10))
clearBtn1.place(x=580, y=42)
clearBtn2 = tk.Button(window, background="red", fg="white", text="CLEAR", width=8,
                      activebackground="red", height=1, font=('Helvetica', 10))
clearBtn2.place(x=580, y=92)


takeImageBtn = tk.Button(window, background="yellow", fg="black", text="CAPTURE IMAGE",
                         activebackground="red",
                         width=15, height=3, font=('Helvetica', 12))
takeImageBtn.place(x=130, y=360)


window.mainloop()

path ='images'
images = []
personNames = []
myList = os.listdir(path)
print(myList)

for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')

encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()