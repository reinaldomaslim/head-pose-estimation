import cv2
import dlib
import numpy as np
import face_recognition
import time
import face_recognition_knn
import operator

from imutils import face_utils
from scipy.spatial import distance as dist

import vlc


smilePath = "./haarcascade_smile.xml"

class Face(object):

    EYE_AR_THRESH = 0.23
    EYE_AR_CONSEC_FRAMES = 3
    eye_counter = 0
    
    yawn_count = 0
    start_yawn = False

    life_counter = 3
    alive = True

    
    name = 'unknown'

    pitch_limit = [-15, 15]
    yaw_limit = [-30, 30]

    smileCascade = cv2.CascadeClassifier(smilePath)

    def __init__(self, face_rect):
        self.face_rect = face_rect
        self.last_time_eyes_open = time.time()
        self.last_time_see_road = time.time()
        self.names = {}
        self.play_alarm = False
        self.music = vlc.MediaPlayer('rooster.mp3')

        
    def match(self, face_rects):
        min_dist = 1000
        threshold = 100
        cur_center = self.face_rect.center()

        for i in range(len(face_rects)):
            face_rect = face_rects[i]
            face_center = face_rect.center()

            #find which one is closest to current face_rect we know
            distance = abs(cur_center.x-face_center.x)+abs(cur_center.y-face_center.y)
            # distance = dist.euclidean(face_center, cur_center)

            if distance < threshold and distance < min_dist:
                min_dist = distance
                ind = i

        if min_dist == 1000:
            self.life_counter -= 1
            if self.life_counter == 0:
                self.alive = False
        else:
            self.life_counter = 3
            self.face_rect = face_rects[ind]
            face_rects.pop(ind)

        return face_rects

    def update(self, face_img, shape, euler_angle):
        self.img = face_img

        #check who is this
        name = self.classify_face(face_img) 
        if name != 'unknown':
            self.name = name

        cv2.putText(self.img, str(self.name), (10, 10), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 255, 0), thickness=1)
        
        euler_angle = np.round(euler_angle, 1)

        rpy = str(euler_angle[2, 0]) + ', ' + str(euler_angle[0, 0]) + ', ' + str(euler_angle[1, 0])

        cv2.putText(self.img, rpy, (10, self.img.shape[0]-10), cv2.FONT_HERSHEY_PLAIN,
            0.7, (0, 255, 0), thickness=1)

        #check head pose if seeing road
        if self.see_road(euler_angle):
            cv2.putText(self.img, "seeing road", (10, 25), cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 255, 0), thickness=1)
            self.last_time_see_road = time.time()
            not_seeing_road_elapsed_time = 0.0
        else:
            not_seeing_road_elapsed_time = round((time.time() - self.last_time_see_road), 1)            
            cv2.putText(self.img, "not seeing road: "+ str(not_seeing_road_elapsed_time), (10, 25), cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 0, 255), thickness=1)




        #check status of eyes
        if self.eyes_open(shape):
            cv2.putText(self.img, "eyes_open", (10, 40), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0), thickness=1)
            self.last_time_eyes_open = time.time()
            closed_eyes_elapsed_time = 0.0
        else:
            closed_eyes_elapsed_time = round((time.time() - self.last_time_eyes_open), 1)
            cv2.putText(self.img, "eyes closed: "+str(closed_eyes_elapsed_time), (10, 40), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 255), thickness=1)

        if not_seeing_road_elapsed_time > 3.0 or closed_eyes_elapsed_time > 3.0:
            self.music.play()
        else:
            self.music.stop()

        #check status of eyes
        mouth_status = self.mouth_status(shape)
        if mouth_status == "yawn":
            if not self.start_yawn:
                self.yawn_count += 1

            self.start_yawn = True
            cv2.putText(self.img, mouth_status + ': ' + str(self.yawn_count), (10, 55), cv2.FONT_HERSHEY_PLAIN,
                                1, (0, 0, 255), thickness=1)
        else:
            if mouth_status == "mouth closed":
                self.start_yawn = False

            cv2.putText(self.img, mouth_status+ ', yawn: ' + str(self.yawn_count), (10, 55), cv2.FONT_HERSHEY_PLAIN,
                                1, (0, 255, 0), thickness=1)

        smile = self.detect_smile()
        # Set region of interest for smiles
        for (x, y, w, h) in smile:
            cv2.putText(self.img, 'smile ', (10, 70), cv2.FONT_HERSHEY_PLAIN,
                                1, (0, 255, 255), thickness=1)
            # cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 0, 0), 1)
            

                
    def detect_smile(self):
        roi_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        smile = self.smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
            )

        return smile


    def see_road(self, euler_angle):
        #check the head yaw and pitch only

        pitch = euler_angle[0, 0]
        yaw = euler_angle[1, 0]

        if pitch > self.pitch_limit[0] and pitch < self.pitch_limit[1]:
            if yaw > self.yaw_limit[0] and yaw < self.yaw_limit[1]:
                return True

        return False

    def eyes_open(self, shape):

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < self.EYE_AR_THRESH:
            self.eye_counter += 1
            if self.eye_counter > self.EYE_AR_CONSEC_FRAMES:
                return False
            else:
                return True
        else:
            # if the eyes were closed for a sufficient number of
            self.eye_counter = 0
            return True

    def mouth_status(self, shape):
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        mouth = shape[mStart:mEnd]
        mouthEAR = self.mouth_aspect_ratio(mouth)
        
        if mouthEAR > 0.7:
            return "yawn"
        elif mouthEAR < 0.7 and mouthEAR > 0.4:
            return "mouth open"
        else:
            return "mouth closed"

    def classify_face(self, face_img):
    
        face_img = cv2.resize(face_img, (48, 48))
    
        try:
            encoded_face = face_recognition.face_encodings(face_img)[0]
        except:
            return "unknown"
        
        name = face_recognition_knn.predict([encoded_face], model_path="trained_knn_model.clf")[0]

        if name != "unknown":
            if name in self.names:
                self.names[name] += 1
            else:
                self.names[name] = 1

            name = max(self.names.items(), key=operator.itemgetter(1))[0]


        return name
    
    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[14], mouth[18])
        B = dist.euclidean(mouth[3], mouth[9])

        C = dist.euclidean(mouth[6], mouth[0])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
     
        # return the eye aspect ratio
        return ear

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
     
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
     
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
     
        # return the eye aspect ratio
        return ear
