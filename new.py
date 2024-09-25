
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import dlib

video_capture = cv2.VideoCapture(0)

hittler_image = face_recognition.load_image_file("photos/hittler.jpeg")
hittler_encoding = face_recognition.face_encodings(hittler_image)[0]

shravan_image = face_recognition.load_image_file("photos/shravan.jpeg")
shravan_encoding = face_recognition.face_encodings(shravan_image)[0]

modi_image = face_recognition.load_image_file("photos/modi.jpeg")
modi_encoding = face_recognition.face_encodings(modi_image)[0]

Rtata_image = face_recognition.load_image_file("photos/Rtata.jpeg")
Rtata_encoding = face_recognition.face_encodings(Rtata_image)[0]

rohit_image = face_recognition.load_image_file("photos/rohit-yadav.jpeg")
rohit_encoding = face_recognition.face_encodings(rohit_image)[0]

melodi_image = face_recognition.load_image_file("photos/melodi.jpeg")
melodi_encoding = face_recognition.face_encodings(melodi_image)[0]

known_face_encoding = [
    hittler_encoding,
    shravan_encoding,
    modi_encoding,
    Rtata_encoding,
    rohit_encoding,
    melodi_encoding,
]

known_faces_names = [
    "hittler",
    "shravan bhosale",
    "narendra modi",
    "ratan tata",
    "rohit yadav",
    "melodi"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    # Update the current time inside the loop
                    now = datetime.now()
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("resume-project1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()