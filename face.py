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
from filterpy.gh import GHFilter

import vlc

smilePath = "./haarcascade_smile.xml"
audio = False

class Face(object):

    EYE_AR_THRESH = 0.20
    EYE_AR_CONSEC_FRAMES = 3
    name = 'unknown'
    pitch_limit = [-20, 20]
    yaw_limit = [-30, 30]
    init_life_count = 5
    smileCascade = cv2.CascadeClassifier(smilePath)

    def __init__(self, face_rect, img):
        self.face_rect = face_rect
        self.last_time_eyes_open = time.time()
        self.last_time_see_road = time.time()
        self.names = {}
        self.play_alarm = False
        self.full_img = img
        self.gaze_filter = GHFilter(x=0, dx=np.array([0]), dt=1, g=.6, h=.02)
        self.skip_frame = 0
        self.is_driver = False
        self.start_yawn = False
        self.life_counter = self.init_life_count
        self.eye_counter = 0
        self.yawn_count = 0
        self.alive = True
        self.born_time = time.time()
   
        if audio:
            self.music = vlc.MediaPlayer('rooster.mp3')

    def life_reset(self):
        self.life_counter = self.init_life_count

    def match(self, face_rects, img):
        self.full_img = img
        min_dist = 1000
        threshold = 100
        cur_center = self.face_rect.center()
        ind = -1

        for i in range(len(face_rects)):
            face_rect = face_rects[i]
            face_center = face_rect.center()

            #find which one is closest to current face_rect we know
            distance = abs(cur_center.x-face_center.x)+abs(cur_center.y-face_center.y)
            
            if distance < min(threshold, min_dist):
                min_dist = distance
                ind = i

        if ind == -1:
            self.life_counter -= 1
            if self.life_counter == 0:
                self.alive = False
        else:
            self.life_reset()
            self.face_rect = face_rects[ind]
            face_rects.pop(ind)

        return face_rects

    def update(self, face_img, shape, euler_angle):
        self.img = face_img
        row, col = face_img.shape[:2]

        #check who is this, do this in skip to save computation
        if self.skip_frame % 10 == 0:
            name = self.classify_face(face_img) 
            if name != 'unknown':
                self.name = name
            self.skip_frame = 0

        self.skip_frame += 1

        cv2.putText(self.img, str(self.name), (10, 10), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 255, 0), thickness=1)
        
        euler_angle = np.round(euler_angle, 1)

        rpy = str(euler_angle[2, 0]) + ', ' + str(euler_angle[0, 0]) + ', ' + str(euler_angle[1, 0])

        cv2.putText(self.img, rpy, (10, self.img.shape[0]-10), cv2.FONT_HERSHEY_PLAIN,
            0.7, (0, 255, 0), thickness=1)

        elapsed_time = str(round(time.time() - self.born_time, 1))
        cv2.putText(self.img, elapsed_time, (self.img.shape[1]-40, self.img.shape[0]-10), cv2.FONT_HERSHEY_PLAIN,
            0.7, (0, 255, 0), thickness=1)

        gaze_direction, leftEye_patch, rightEye_patch = self.check_gaze(shape)
        cv2.putText(self.img, "gaze: " + str(gaze_direction), (10, self.img.shape[0]-25), cv2.FONT_HERSHEY_PLAIN,
                0.7, (0, 255, 0), thickness=1)

        self.img[:leftEye_patch.shape[0], int(col/2)-leftEye_patch.shape[1]:int(col/2)] = cv2.cvtColor(leftEye_patch, cv2.COLOR_GRAY2BGR)
        self.img[:rightEye_patch.shape[0], int(col/2):int(col/2)+rightEye_patch.shape[1]] = cv2.cvtColor(rightEye_patch, cv2.COLOR_GRAY2BGR)

        #check head pose if seeing road
        if self.see_road(euler_angle, gaze_direction):
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

        if audio:
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

    def see_road(self, euler_angle, gaze_direction):
        #check the head yaw and pitch only

        pitch = euler_angle[0, 0]
        yaw = euler_angle[1, 0] + gaze_direction

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

    def check_gaze(self, shape):
        #check gaze of eyes

        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        xmin = np.clip(np.amin(leftEye[:, 0]), 0, self.full_img.shape[1]-1)
        ymin = np.clip(np.amin(leftEye[:, 1]), 0, self.full_img.shape[0]-1)
        xmax = np.clip(np.amax(leftEye[:, 0]), 0, self.full_img.shape[1]-1)
        ymax = np.clip(np.amax(leftEye[:, 1]), 0, self.full_img.shape[0]-1)


        leftEye_patch = cv2.resize(self.full_img[ymin:ymax, xmin:xmax], (30, 10))
        leftEye_patch = cv2.GaussianBlur(leftEye_patch, (9, 9), 0)

        xmin = np.clip(np.amin(rightEye[:, 0]), 0, self.full_img.shape[1]-1)
        ymin = np.clip(np.amin(rightEye[:, 1]), 0, self.full_img.shape[0]-1)
        xmax = np.clip(np.amax(rightEye[:, 0]), 0, self.full_img.shape[1]-1)
        ymax = np.clip(np.amax(rightEye[:, 1]), 0, self.full_img.shape[0]-1)

        rightEye_patch = cv2.resize(self.full_img[ymin:ymax, xmin:xmax], (30, 10))
        rightEye_patch = cv2.GaussianBlur(rightEye_patch, (9, 9), 0)
        
        sum_left = np.sum(leftEye_patch, axis = 0)
        sum_right = np.sum(rightEye_patch, axis = 0)

        eyeball_pos_left = np.argmin(sum_left)
        eyeball_pos_right = np.argmin(sum_right)

        cv2.line(leftEye_patch, (eyeball_pos_left, 0), (eyeball_pos_left, rightEye_patch.shape[0]), (255), 1, cv2.LINE_AA)
        cv2.line(rightEye_patch, (eyeball_pos_right, 0), (eyeball_pos_right, rightEye_patch.shape[0]), (255), 1, cv2.LINE_AA)

        gaze_angle_left = 5.0*(leftEye_patch.shape[1]/2 - eyeball_pos_left) 
        gaze_angle_right = 5.0*(rightEye_patch.shape[1]/2 - eyeball_pos_right) 

        avg_gaze = (gaze_angle_right+gaze_angle_left)/2
        avg_gaze, _ = self.gaze_filter.update(avg_gaze)

        return round(avg_gaze[0], 2), leftEye_patch, rightEye_patch

    def mouth_status(self, shape):
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        mouth = shape[mStart:mEnd]
        mouthEAR = self.mouth_aspect_ratio(mouth)
        
        if mouthEAR > 0.8:
            return "yawn"
        elif mouthEAR < 0.8 and mouthEAR > 0.4:
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
        if brow_lowerer_ratio < 0.17:
            res += 4

        return res

    def in_distress(self, shape):
        small_triangle = dist.euclidean(shape[21], shape[27]) + dist.euclidean(shape[22], shape[27]) + dist.euclidean(shape[21], shape[22])
        large_triangle = dist.euclidean(shape[33], shape[26]) + dist.euclidean(shape[26], shape[17]) + dist.euclidean(shape[17], shape[33])

        procerus_aspect_ratio = small_triangle/large_triangle
        if procerus_aspect_ratio < 0.190:
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
    