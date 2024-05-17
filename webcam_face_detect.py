
import sys
import cv2
import face_recognition as fr
import os
import face_recognition
import numpy as np


def get_encoded_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./face_repository"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("face_repository/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):

    face = fr.load_image_file("face_repository/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def webcam_face_detect(video_mode, faces, nogui = False, cascasdepath = "haarcascade_frontalface_default.xml"):

    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    face_cascade = cv2.CascadeClassifier(cascasdepath)

    video_capture = cv2.VideoCapture(video_mode)
    num_faces = 0


    while True:
        ret, image = video_capture.read()

        if not ret:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30,30)
            )

        face_locations = face_recognition.face_locations(image)
        unknown_face_encodings = face_recognition.face_encodings(image, face_locations)

        print("The number of faces found = ", len(faces))
        num_faces = len(faces)
        font = cv2.FONT_HERSHEY_DUPLEX

        face_names = []
        for face_encoding in unknown_face_encodings:
       
            matches = face_recognition.compare_faces(faces_encoded, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(image, (left-20, top-10), (right+20, bottom+15), (300, 0, 0), 2)
                cv2.rectangle(image, (left-20, bottom -10), (right+20, bottom+15), (500, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left -10, bottom + 10), font, 0.5, (300, 300, 300), 1)

        if not nogui:
            
            cv2.imshow("Faces found", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()
    return num_faces


if __name__ == "__main__":
    if len(sys.argv) < 2:
        video_mode= 0
    else:
        video_mode = sys.argv[1]
    faces = get_encoded_faces()

    webcam_face_detect(video_mode, faces)

