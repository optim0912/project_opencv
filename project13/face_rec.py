import sys
import numpy as np
import cv2
from pathlib import Path


face_label_txt = Path('./train_images/label.txt')
face_labels = []
try:
    face_labels = face_label_txt.read_text(encoding = 'utf-8').splitlines()
except:
    print("label.txt could not be read. ")
    sys.exit()

colors = np.random.uniform(0, 255, size=(len(face_labels), 3))

def drawBox(img, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), colors[classId], 2)

    label = f'{face_labels[classId]}: {conf:.2f}'

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left - 1, top - labelSize[1] - baseLine),
                  (left + labelSize[0], top), colors[classId], -1)
    cv2.putText(img, label, (left, top - baseLine), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)

def face_recognition(recognition_net, crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blob = cv2.dnn.blobFromImage(gray, 1 / 255., (150, 200))
    recognition_net.setInput(blob)
    prob = recognition_net.forward()  # prob.shape=(1, 3)

    _, confidence, _, maxLoc = cv2.minMaxLoc(prob)
    face_idx = maxLoc[0]

    return face_idx, confidence


detection_net = cv2.dnn.readNet('opencv_face_detector/opencv_face_detector_uint8.pb',
                                'opencv_face_detector/opencv_face_detector.pbtxt')

if detection_net.empty():
    print('Detection Net open failed!')
    sys.exit()

recognition_net = cv2.dnn.readNet('frozen_face_rec_cnn_new.pb')

if detection_net.empty():
    print('Recognition Net open failed!')
    sys.exit()

cap = cv2.VideoCapture('test.mp4')
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 30, (w, h))

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    count += 1
    if count % (60*30) == 0:
        print(f'The {count // 1800}-minute video has been made.')

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    detection_net.setInput(blob)
    detect = detection_net.forward()

    detect = detect[0, 0, :, :]

    boxesToDraw = []
    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        crop = frame[y1:y2, x1:x2]
        face_idx, confidence = face_recognition(recognition_net, crop)

        #if confidence < 0.6:
        #    break
        
        boxesToDraw.append([frame, face_idx, confidence, x1, y1, x2, y2])

    # 객체별 바운딩 박스 그리기 & 클래스 이름 표시
    for box in boxesToDraw:
        drawBox(*box)

    cv2.imshow('frame', frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
