from flask import Flask, render_template, Response
import cv2
import face_recognition
import os
import csv
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load student images
path = 'students'

images = []
student_names = []

for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    student_names.append(os.path.splitext(file)[0])

# Encode faces
def encode_faces(images):

    encoded_list = []

    for img in images:

        rgb_img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        encode = face_recognition.face_encodings(
            rgb_img
        )[0]

        encoded_list.append(encode)

    return encoded_list

known_encodings = encode_faces(images)

# Attendance function
def mark_attendance(name):

    with open(
        'attendance.csv',
        'a+',
        newline=''
    ) as f:

        f.seek(0)

        data = f.readlines()

        name_list = []

        for line in data:

            entry = line.split(',')

            name_list.append(entry[0])

        if name not in name_list:

            now = datetime.now()

            time = now.strftime('%H:%M:%S')

            writer = csv.writer(f)

            writer.writerow([name, time])

# Webcam
camera = cv2.VideoCapture(0)

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        small_frame = cv2.resize(
            frame,
            (0, 0),
            None,
            0.25,
            0.25
        )

        rgb_small = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        face_locations = face_recognition.face_locations(
            rgb_small
        )

        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations
        )

        for encode_face, face_location in zip(
            face_encodings,
            face_locations
        ):

            matches = face_recognition.compare_faces(
                known_encodings,
                encode_face
            )

            face_distance = face_recognition.face_distance(
                known_encodings,
                encode_face
            )

            match_index = np.argmin(face_distance)

            if matches[match_index]:

                name = student_names[
                    match_index
                ].upper()

                y1, x2, y2, x1 = face_location

                y1, x2, y2, x1 = (
                    y1 * 4,
                    x2 * 4,
                    y2 * 4,
                    x1 * 4
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.rectangle(
                    frame,
                    (x1, y2 - 35),
                    (x2, y2),
                    (0, 255, 0),
                    cv2.FILLED
                )

                cv2.putText(
                    frame,
                    name,
                    (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                mark_attendance(name)

        ret, buffer = cv2.imencode(
            '.jpg',
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

# Home page
@app.route('/')
def index():

    return render_template('index.html')

# Video feed
@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# Attendance page
@app.route('/attendance')
def attendance():

    records = []

    with open('attendance.csv', 'r') as f:

        reader = csv.reader(f)

        for row in reader:
            records.append(row)

    return render_template(
        'attendance.html',
        records=records
    )

if __name__ == "__main__":
    app.run(debug=True)