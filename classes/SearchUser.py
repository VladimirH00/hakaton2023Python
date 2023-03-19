import urllib.request
import numpy as np
import cv2
import face_recognition


class SearchUser:

    def __init__(self, data_set_provider):
        self.__output = None
        self.__data_set_provider = data_set_provider
        self.__images = []
        self.__class_names = []
        self.__encodeList = data_set_provider.get_encodings()

    def recognition(self, file_name, path=''):
        req = urllib.request.urlopen(f'http://hakaton2023/storage/find-data/{file_name}')
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        imgS = cv2.resize(img, (0, 0), None, 0.75, 0.75)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)   
        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        self.__user_id = None
        self.__other_users = list()
        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            # matches = face_recognition.compare_faces(self.__encodeList, encodeFace)
            class_names = self.__data_set_provider.get_class_names()
            faceDis = face_recognition.face_distance(self.__encodeList, encodeFace)
            matchIndex = np.argmin(faceDis)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = int(y1 * 1.33), int(x2 * 1.33), int(y2 * 1.33), int(x1 * 1.33)
            # if 1 - faceDis[matchIndex] > 0.6:
            imgNew = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            imgNew = cv2.rectangle(imgNew, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            self.__user_id = class_names[matchIndex]
            self.percent = 100 - int(faceDis[matchIndex]*100)
            faceDis[np.argmin(faceDis)] = 9999
            self.__other_users.append([class_names[np.argmin(faceDis)], faceDis[np.argmin(faceDis)]])
            faceDis[np.argmin(faceDis)] = 9999
            self.__other_users.append([class_names[np.argmin(faceDis)], faceDis[np.argmin(faceDis)]])
                # imgNew = cv2.putText(imgNew, class_names[matchIndex], (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # else:
            #     imgNew = cv2.putText(img, 'No name', (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        width = int(imgNew.shape[1] * 50 / 100)
        height = int(imgNew.shape[0] * 50 / 100)
        dsize = (width, height)
        self.__output = cv2.resize(imgNew, dsize)
        return self

    def get_id_user(self):
        return self.__user_id

    def get_other_users(self):
        return self.__other_users

    def get_result_image(self):
        return self.__output
