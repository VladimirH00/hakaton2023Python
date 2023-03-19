import datetime
import os
import time
from multiprocessing import Queue

import numpy as np
from PyQt6 import QtWidgets
import PyQt6
from PyQt6.QtWidgets import QWidget, QPushButton, QTextEdit, QRadioButton
from PyQt6 import QtCore
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication
from PyQt6 import QtGui
from PyQt6.QtWidgets import QVBoxLayout
import cv2
import face_recognition

import sys

running = False
q = Queue()

if not os.path.exists('save_unmask/' + str(datetime.datetime.now().date())):
    os.mkdir('save_unmask/' + str(datetime.datetime.now().date()))


path = 'save_unmask/' + str(datetime.datetime.now().date())
data_set = 'dataset'
images = []
classNames = []

############### инициализация распознования объектов
classes = []
whT = 320
confThr = 0.7
nmsThr = 0.4

myList = os.listdir(path)
with open('object_detections/coco.names', 'r') as f:
    classes = f.read().rstrip('\n').split("\n")

net = cv2.dnn.readNet('object_detections/yolov3.cfg', 'object_detections/yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
###############

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList


encodeListKnown = findEncodings(images)


class OwnImageWidget(QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()


class Window(QtWidgets.QMainWindow):
    def __init__(self, elements):
        super().__init__()
        self.setWindowTitle('Искусственный интеллект')
        self.cap = None
        self.grpbox = PyQt6.QtWidgets.QGroupBox("Режим работы")
        self.grpbox.setFont(QtGui.QFont("Times New Roman", 15))
        self.currentClass = 'car'
        self.combobox = PyQt6.QtWidgets.QComboBox()
        self.combobox.addItems(['all'] + elements)
        hbox = QtWidgets.QHBoxLayout()
        self.rad1 = QRadioButton("Распознание людей по лицу")
        self.rad1.setChecked(True)
        hbox.addWidget(self.rad1)
        self.rad2 = QRadioButton("Нарушение масочного режима")
        hbox.addWidget(self.rad2)
        self.rad3 = QRadioButton("Обнаружение объектов")
        hbox.addWidget(self.rad3)
        hbox.addWidget(self.combobox)
        self.grpbox.setLayout(hbox)
        self.setGeometry(0, 0, 600, 600)
        self.btnStart = QPushButton('Начать просмотр')
        self.name = QTextEdit()
        self.imgCam = OwnImageWidget()
        self.btnStart.setCheckable(True)
        layout = QVBoxLayout()
        layout.addWidget(self.btnStart)
        layout.addWidget(self.grpbox)
        layout.addWidget(self.name)
        layout.addWidget(self.imgCam)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)
        self.threadManager = QThreadPool()
        self.btnStart.clicked.connect(self.activateThread)

    def activateThread(self):
        global running
        running = not running
        if running:
            self.btnStart.setText('Выключить камеру')
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.threadManager.start(self.updateFrame)
            if self.rad2.isChecked():
                self.threadManager.start(self.saveImage)
        else:
            self.btnStart.setText('Начать просмотр')
            self.cap.release()

    def findObjects(self, outputs, img):
        global classes
        wT, hT, _ = img.shape
        bbox = []
        classIds = []
        confs = []
        for out in outputs:
            for det in out:
                scores = det[5:]
                classId = np.argmax(scores)
                confindence = scores[classId]
                if confindence > confThr:
                    # mean-The Object is detected
                    # process
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confindence))

        indexes = cv2.dnn.NMSBoxes(bbox, confs, confThr, nmsThr)
        for i in range(len(bbox)):
            if i in indexes:
                x, y, w, h = bbox[i]
                x = x*2
                y = round(y *0.5)
                print(y)
                print(y+h)
                label = str(classes[classIds[i]])
                if self.combobox.currentText() != 'all':
                    if label == self.combobox.currentText():
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    def encodingImg(self, img, facesLocations):
        encodeCurFrame = face_recognition.face_encodings(img, facesLocations)
        for encodeFace, faceLoc in zip(encodeCurFrame, facesLocations):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            return matches[matchIndex]

    def saveImage(self):
        global running
        global q
        global encodeListKnown
        while running:
            if not q.empty():
                frame = q.get()
                img = frame['img']
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                facesLocations = face_recognition.face_locations(imgS)
                encodeCurFrame = face_recognition.face_encodings(imgS, facesLocations)
                for encodeFace, faceLoc in zip(encodeCurFrame, facesLocations):
                    if len(encodeListKnown) == 0:
                        encodeListKnown.append(encodeFace)
                        if not os.path.exists('save_unmask/' + str(datetime.datetime.now().date())):
                            os.mkdir('save_unmask/' + str(datetime.datetime.now().date()))
                        cv2.imwrite("save_unmask/" + str(datetime.datetime.now().date()) + '/User' + str(
                            int(round(time.time() * 1000))) + ".jpg", img)
                    else:
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        if faceDis.size > 0:
                            matchIndex = np.argmin(faceDis)
                            if matches[matchIndex] < 0.7:
                                encodeListKnown.append(encodeFace)
                                if not os.path.exists('save_unmask/' + str(datetime.datetime.now().date())):
                                    os.mkdir('save_unmask/' + str(datetime.datetime.now().date()))
                                cv2.imwrite("save_unmask/" + str(datetime.datetime.now().date()) + '/User' + str(
                                    int(round(time.time() * 1000))) + ".jpg", img)

    def updateFrame(self):
        global running
        global q
        global myList
        global encodeListKnownSecond
        global classNames
        global net
        while running:
            success, img = self.cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            img_height, img_width, img_colors = img.shape
            scale_w = float(self.imgCam.frameSize().width()) / float(img_width)
            scale_h = float(self.imgCam.frameSize().height()) / float(img_height)
            scale = min([scale_w, scale_h])
            facesCurFrame = face_recognition.face_locations(imgS)

            if self.rad1.isChecked():
                encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnownSecond, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnownSecond, encodeFace)

                    matchIndex = np.argmin(faceDis)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    if 1-matches[matchIndex-1] > 0.6:
                        name = classNames[matchIndex-1]
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    else:
                        cv2.putText(img, 'No name', (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            elif self.rad2.isChecked():
                facesLocations = facesCurFrame
                for (top, right, bottom, left) in facesLocations:
                    frame = {}
                    top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
                    img_rez = np.copy(img[top:bottom, left:right, :])
                    frame["img"] = img_rez
                    q.put(frame)
                    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

                if scale == 0:
                    scale = 1
            else:
                blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                layers = net.getLayerNames()
                outputN = []
                for i in net.getUnconnectedOutLayers():
                    outputN.append(layers[i[0] - 1])
                outputs = net.forward(outputN)
                self.findObjects(outputs, img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format.Format_RGB888)
            self.imgCam.setImage(image)

    def closeEvent(self, event):
        global running
        running = False
        if self.cap:
            self.cap.release()


def application():
    app = QApplication(sys.argv)
    window = Window(classes)

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    myList = os.listdir(data_set)
    for cls in myList:
        curImg = cv2.imread(f'{data_set}/{cls}')
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])

    encodeListKnownSecond = findEncodings(images)
    application()
