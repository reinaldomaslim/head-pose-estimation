import cv2
import dlib
import numpy as np
import face_recognition
import time
import face_recognition_knn

from imutils import face_utils
from scipy.spatial import distance as dist


class Face(object):

    EYE_AR_THRESH = 0.22
    EYE_AR_CONSEC_FRAMES = 3
    eye_counter = 0
    alive = True
    last_time_eyes_open = time.time()
    last_time_see_road = time.time()
    name = 'unknown'

    def __init__(self, face_rect):
        self.face_rect = face_rect

    def match(self, face_rects):
        min_dist = 1000
        threshold = self.img.shape[0]/10
        cur_center = self.face_rect.center()

        for i in range(len(face_rects)):
            face_rect = face_rects[i]
            face_center = face_rect.center()
            #find which one is closest to current face_rect we know
            dist = abs(cur_center.x-face_center.x)+abs(cur_center.y-face_center.y)

            if dist < threshold and dist < min_dist:
                min_dist = dist
                ind = i

        if min_dist == 1000:
            self.alive = False
        else:
            self.face_rect = face_rects[ind]
            face_rects.pop(ind)

        return face_rects

    def update(self, face_img, shape, euler_angle):
        self.img = face_img

        #check who is this
        name = self.classify_face(face_img) 
        if name != 'unknown':
            self.name = name

        cv2.putText(self.img, str(self.name), (int(10*self.img.shape[0]/240), int(10*self.img.shape[0]/240)), cv2.FONT_HERSHEY_PLAIN,
            1.0*self.img.shape[0]/240, (0, 255, 0), thickness=1)
        
        #check status of eyes
        if self.eyes_open(shape):
            cv2.putText(self.img, "eyes_open", (10, 25), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0), thickness=1)
            self.last_time_eyes_open = time.time()
        else:
            closed_eyes_elapsed_time = str(round((time.time() - self.last_time_eyes_open), 1)) + ' s'
            cv2.putText(self.img, "eyes closed: "+closed_eyes_elapsed_time, (10, 25), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 255), thickness=1)


        #check head pose if seeing road
        if self.see_road(euler_angle):
            cv2.putText(self.img, "seeing road", (10, 40), cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 255, 0), thickness=1)
            self.last_time_see_road = time.time()
        else:
            not_seeing_road_elapsed_time = str(round((time.time() - self.last_time_see_road), 1)) + ' s'            
            cv2.putText(self.img, "not seeing road: "+not_seeing_road_elapsed_time, (10, 40), cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 0, 255), thickness=1)

    def see_road(self, euler_angle):
        #check the head yaw and pitch only
        pitch_limit = [-15, 15]
        yaw_limit = [-30, 30]

        pitch = euler_angle[0, 0]
        yaw = euler_angle[1, 0]

        if pitch > pitch_limit[0] and pitch < pitch_limit[1]:
            if yaw > yaw_limit[0] and yaw < yaw_limit[1]:
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

    def classify_face(self, face_img):
        face_img = cv2.resize(face_img, (48, 48))
        try:
            encoded_face = face_recognition.face_encodings(face_img)[0]
        except:
            return "unknown"
        
        name = face_recognition_knn.predict([encoded_face], model_path="trained_knn_model.clf")[0]
                
        return name
    
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

