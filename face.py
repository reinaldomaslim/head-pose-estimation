import cv2
import dlib
import numpy as np
import face_recognition
import time
import face_recognition_knn
import operator
from scipy import stats

from imutils import face_utils
from scipy.spatial import distance as dist

import vlc

smilePath = "./haarcascade_smile.xml"

class Face(object):

    EYE_AR_THRESH = 0.20
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
            
        brow = self.brow_status(shape)

        if brow == 5:
            #sad
            brow_text = 'sad'
        elif brow == 3:
            #suprise
            brow_text = 'brow raised'
        elif brow == 7:
            #fear
            brow_text = 'fear'
        elif brow == 4:
            #anger
            brow_text = 'tense'
        else:
            #neutral
            brow_text = ''

        cv2.putText(self.img, brow_text, (10, 85), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 255), thickness=1)


        if self.in_distress(shape):
            cv2.putText(self.img, 'distress ', (10, 100), cv2.FONT_HERSHEY_PLAIN,
                                1, (0, 255, 255), thickness=1)

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

        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        if ear < self.EYE_AR_THRESH:
            self.eye_counter = 0
            return False            
        else:
            # if the eyes were closed for a sufficient number of
            self.eye_counter += 1
            if self.eye_counter > self.EYE_AR_CONSEC_FRAMES:
                return True
            else:
                return False


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


    def brow_status(self, shape):
        #classify as inner_brow_raise, outer_brow_raise, and brow lowerer. wrt to each other and wrt to upper eye
        res = 0 

        normalizer = dist.euclidean(shape[24], shape[33]) + dist.euclidean(shape[19], shape[33])

        #inner brow raise
        left_inner = dist.euclidean(shape[20], shape[38]) + dist.euclidean(shape[21], shape[39])
        right_inner = dist.euclidean(shape[22], shape[42]) + dist.euclidean(shape[23], shape[43]) 
        inner_brow_ratio = (left_inner+right_inner)/normalizer

        #outer brow raise
        left_outer = dist.euclidean(shape[18], shape[36]) + dist.euclidean(shape[19], shape[37])
        right_outer = dist.euclidean(shape[24], shape[44]) + dist.euclidean(shape[25], shape[45])
        outer_brow_ratio = (left_outer+right_outer)/normalizer

        #brow lowerer
        left_brow = shape[17:22]
        _, _, _, _,  std_err_left= stats.linregress(left_brow[:, 0], left_brow[:, 1])
        right_brow = shape[22:27]
        _, _, _, _,  std_err_right = stats.linregress(right_brow[:, 0], right_brow[:, 1])

        brow_lowerer_ratio = std_err_left+std_err_right
        
        #thresholds  = 0.50, 0.60, 0.18
        res = 0
        if inner_brow_ratio > 0.50:
            res += 1
        if outer_brow_ratio > 0.60:
            res += 2
        if brow_lowerer_ratio < 0.18:
            res += 4

        return res

    def in_distress(self, shape):
        small_triangle = dist.euclidean(shape[21], shape[27]) + dist.euclidean(shape[22], shape[27]) + dist.euclidean(shape[21], shape[22])
        large_triangle = dist.euclidean(shape[33], shape[26]) + dist.euclidean(shape[26], shape[17]) + dist.euclidean(shape[17], shape[33])

        procerus_aspect_ratio = small_triangle/large_triangle
        if procerus_aspect_ratio < 0.185:
            #in distress
            return True
        else:
            return False



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
    