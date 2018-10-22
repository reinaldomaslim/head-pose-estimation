import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import face_recognition
import time
import face_recognition_knn

face_landmark_path = './shape_predictor_68_face_landmarks.dat'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]

D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

def see_road(euler_angle):
    #check the head yaw and pitch only
    pitch_limit = [-15, 15]
    yaw_limit = [-30, 30]

    pitch = euler_angle[0, 0]
    yaw = euler_angle[1, 0]

    if pitch > pitch_limit[0] and pitch < pitch_limit[1]:
        if yaw > yaw_limit[0] and yaw < yaw_limit[1]:
            return True

    return False

def eye_aspect_ratio(eye):
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

def classify_face(face_img):

    # try:
    #     unknown_face_encoding = face_recognition.face_encodings(face_img)[0]

    #     results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)
    # except:
    #     return False

    # return results[0]
    # face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    face_img = cv2.resize(face_img, (48, 48))
    try:
        encoded_face = face_recognition.face_encodings(face_img)[0]
    except:
        return "unknown"
    
    name = face_recognition_knn.predict([encoded_face], model_path="trained_knn_model.clf")

    return name


def main():
    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
        
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.23
    EYE_AR_CONSEC_FRAMES = 5

    # initialize the frame counters and the total number of blinks
    COUNTER = 0


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while cap.isOpened():

        ret, img = cap.read()
        #flip img
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = img.copy()

        if ret:
            #detect faces from image
            face_rects = detector(img, 0)
            
            for i in range(len(face_rects)):
                mask = np.zeros(img.shape[:2], dtype = np.uint8)

                #get face landmarks
                face_rect = face_rects[i]
                shape = predictor(img, face_rect)

                shape = face_utils.shape_to_np(shape)
                reprojectdst, euler_angle = get_head_pose(shape)


                #detect hull to crop face
                hull = cv2.convexHull(shape)
                cv2.fillPoly(mask, pts =[hull], color=1)


                #visualize
                cv2.polylines(res,[hull],True,(255, 0, 0))
                
                for (x, y) in shape:
                    cv2.circle(res, (x, y), 1, (0, 255, 0), -1)

                for start, end in line_pairs:
                    cv2.line(res, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                (x, y, w, h) = face_utils.rect_to_bb(face_rect)
                img_masked = cv2.bitwise_and(img, img, mask = mask)

                faceOrig = cv2.resize(img_masked[y:y + h, x:x + w], (256, 256))
                faceAligned = cv2.resize(fa.align(img_masked, gray, face_rect), (img.shape[1], img.shape[0]))


                # name = classify_face(faceAligned)
                # cv2.putText(faceAligned, str(name), (20, 150), cv2.FONT_HERSHEY_PLAIN,
                #     2, (0, 255, 0), thickness=2)


                cv2.putText(faceAligned, "P, Y, R: " + str(np.round(euler_angle[:, 0], 2)), (20, 20), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0), thickness=1)

                #check head pose if seeing road
                if see_road(euler_angle):
                    cv2.putText(faceAligned, "SEEING ROAD", (20, 60), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), thickness=2)
                else:
                    cv2.putText(faceAligned, "NOT SEEING ROAD", (20, 60), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 0, 255), thickness=2)


                #find left and right eye aspect ratio to detect blink
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(faceAligned, "EYES CLOSED", (20, 90), cv2.FONT_HERSHEY_PLAIN,
                                        2, (0, 0, 255), thickness=2)


                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    # if the eyes were closed for a sufficient number of
    
                    # reset the eye frame counter
                    COUNTER = 0

                    cv2.putText(faceAligned, "EYES OPEN", (20, 90), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), thickness=2)
         
         
                cv2.putText(faceAligned, "mean EAR: {:.2f}".format(ear), (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
         

                #compare face

                res = np.hstack((res, faceAligned))


            cv2.imshow("demo", res)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
