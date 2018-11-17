import cv2
import dlib
import numpy as np
import face_recognition
import time
import face_recognition_knn
import math
import copy
from imutils import face_utils
from scipy.spatial import distance as dist
from face import Face


prototxt = 'deploy.prototxt'
weight = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, weight)

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

def detect(img):

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    detections = np.squeeze(detections)


    dets = dlib.rectangles()
    for i in range(detections.shape[0]):

        if detections[i, 2] > 0.3:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[i, 3:7] * np.array([w, h, w, h])
            # dets.append(box.astype(np.uint16))
            box = box.astype(np.uint16)
            rect = dlib.rectangle(box[0], box[1], box[2], box[3])

            dets.append(rect)

    return dets


def main():
    # return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
        
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    
    init = True

    faces_list = []
    no_face_detected_count = 0

    frame_count = 0
    hz = []

    while cap.isOpened():
        # try:
        print("-----")
        start_time = time.time()
        ret, img = cap.read()
        #flip img
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = img.copy()

        if init:
            fa = face_utils.FaceAligner(predictor, desiredFaceWidth=int(img.shape[0]/2))
            init = False

        if ret:
            #detect faces from image
            #skip frames for this cos it's slower
            if frame_count%2 == 0:
                #cnn way
                # face_rects = detect(img)
                #dlib way
                face_rects = detector(img, 0)
                
                prev_face_rects = copy.copy(face_rects)
                frame_count = 0
            else:
                face_rects = copy.copy(prev_face_rects)

            if len(face_rects) == 0:
                no_face_detected_count += 1
                if no_face_detected_count > 10:
                    #reset faces
                    faces_list = []
            else:
                no_face_detected_count = 0

            #PROBLEM WITH MULTIPLE FACES, FALSE POSITIVES

            # print(len(face_rects), faces_list)
            
            # #check if faces_list has duplicates:
            # for face_0, face_1 in zip(*[iter(faces_list)]*2):
            #     distance = abs(face_0.face_rect.center().x-face_1.face_rect.center().x)+abs(face_0.face_rect.center().y-face_1.face_rect.center().y)
            #     print(distance)
            #     if distance < 10:
            #         print("pair remove")
            #         #delete one of these
            #         if face_0.born_time - face_1.born_time > 0:
            #             faces_list.remove(face_0)
            #             face_1.life_reset()
            #         else:
            #             faces_list.remove(face_1)
            #             face_0.life_reset()

            init_face_rects_len = len(face_rects)
            #match old faces with new detections
            for face in faces_list:
                face_rects = face.match(face_rects, gray)
                if not face.alive:
                    faces_list.remove(face)

            # print(len(face_rects))

            if init_face_rects_len > len(faces_list):
                #add new faces
                for face_rect in face_rects:
                    face = Face(face_rect, gray)
                    faces_list.append(face)

            #finalized faces_list for this timestep
            expansion = np.zeros((img.shape[0], int(math.ceil(len(faces_list)/2)*img.shape[0]/2), 3), dtype = np.uint8)

            #choose the main driver
            if len(faces_list) == 1:
                faces_list[0].is_driver = True
            else:
                max_area = 0
                #choose the largest face
                for i in range(len(faces_list)):
                    area = faces_list[i].face_rect.area()
                    if area > max_area:
                        max_area = area
                        ind = i
                for i in range(len(faces_list)):
                    if i != ind:
                        faces_list[i].is_driver = False
                    else:
                        faces_list[i].is_driver = True
                
                #reorder list
                if max_area != 0:
                    faces_list[0], faces_list[ind] = faces_list[ind], faces_list[0]

            #process each face
            for i in range(len(faces_list)):
                face = faces_list[i]
                mask = np.zeros(img.shape[:2], dtype = np.uint8)

                #get face landmarks
                face_rect = face.face_rect
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
                x = np.clip(x, 0, img.shape[1])
                y = np.clip(y, 0, img.shape[0])
                w = np.clip(w, 1, img.shape[1]-x)
                h = np.clip(h, 1, img.shape[0]-y)

                img_masked = cv2.bitwise_and(img, img, mask = mask)
                faceOrig = cv2.resize(img_masked[y:y + h, x:x + w], (int(img.shape[0]/2), int(img.shape[0]/2)))
                face_img = fa.align(img_masked, gray, face_rect)

                face.update(face_img, shape, euler_angle)
                start_y = int(i%2*img.shape[0]/2)
                start_x = int(int(i/2)*img.shape[0]/2)
                expansion[start_y:start_y+int(img.shape[0]/2), start_x:start_x+int(img.shape[0]/2)] = face.img

                if face.is_driver:
                    cv2.rectangle(expansion,(start_x,start_y),(start_x+int(img.shape[0]/2),start_y+int(img.shape[0]/2)), (0, 255, 0), 2) 

            res = np.hstack((res, expansion))
            


            frame_count+=1
            cv2.imshow("demo", res)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        hz.append(1.0/(time.time()- start_time))
        if len(hz) > 20:
            hz.pop(0)

        print("hz: " + str(sum(hz)/len(hz)))
    
        # except:
        #     print("error")
        #     continue

if __name__ == '__main__':
    main()
