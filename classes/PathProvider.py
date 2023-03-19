import os
import urllib.request
import numpy as np
import cv2
import face_recognition


class PathProvider:

    def __init__(self, listUsers):
        images = []
        class_names = []
        for id in listUsers:
            url = f'http://hakaton2023/storage/dataset/{id}.jpg'
            req = urllib.request.urlopen(url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            curImg = cv2.imdecode(arr, -1)
            images.append(curImg)
            class_names.append(id)
        self.__encodings = self.findEncodings(images)
        self.__classNames = class_names

    def get_encodings(self):
        return self.__encodings

    def get_class_names(self):
        return self.__classNames

    def findEncodings(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)
            if encode:
                encodeList.append(encode[0])
        return encodeList