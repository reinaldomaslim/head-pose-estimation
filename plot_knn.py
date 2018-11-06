import numpy as np
import matplotlib.pyplot as plt
import math
import os
import os.path
import pickle
import face_recognition
import cv2

from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.decomposition import PCA


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def plot_pca(train_dir, model_save_path=None):
    
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        print(class_dir)

        count = 0
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            
            image = cv2.imread(img_path)
            image = cv2.resize(image, (48, 48))

            # Add face encoding for current image to the training set
            encoded_face = face_recognition.face_encodings(image)
            
            if len(encoded_face) == 0:
                continue

            X.append(encoded_face[0])
            y.append(class_dir)

            if count > 20:
                break

            count+=1

    X = np.asarray(X)

    with open(model_save_path, 'rb') as f:
        clf = pickle.load(f)
    
    # Z = clf.predict(X)
    pca = PCA(n_components=2)

    X_r = pca.fit(X).transform(X)
    plt.figure()

    class_names = ['reinaldo', 'kokhui', 'emily', 'eewei', 'yanpai']
    colors = ['navy', 'turquoise', 'darkorange', 'blue', 'green', 'magenta', 'yellow', 'cyan', 'red']

    lw = 2    
    for color, i in zip(colors, class_names):
        ind = []
        for j in range(len(y)):
            if y[j] == i:
                ind.append(j)  
                  
        plt.scatter(X_r[ind, 0], X_r[ind, 1], color=color, alpha=.8, lw=lw, label=i)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of encoded FACE dataset')

    plt.show()




plot_pca("face_database", model_save_path="trained_knn_model.clf")


