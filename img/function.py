# -----------------------------------
# GLOBAL FEATURE EXTRACTION
# -----------------------------------
from sklearn.preprocessing import LabelEncoder
import numpy as np
import mahotas
import cv2
import os
from const import *
import shutil


# detail about train set
def info(train_path):
    # print(train_path)
    dict = {}
    listLabel = os.listdir(train_path)
    for each in listLabel:
        numberFiles = len(os.listdir(os.path.join(train_path, each)))
        dict[each] = numberFiles
    return dict


# make a folder to empty folder
def clearDir():
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# feature = Color Histogram + Haralick Texture + Hu Moments
def get_feature(image):
    image = cv2.resize(image, fixed_size)
    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    return global_feature


def get_train_label():
    # get the training labels
    train_labels = os.listdir(train_path)
    # sort the training labels
    train_labels.sort()
    # print(train_labels)
    return train_labels


def get_data_label():
    train_labels = get_train_label()

    # empty lists to hold feature vectors and labels
    global_features = []
    labels = []

    # loop over the training data sub-folders
    for training_name in train_labels:
        # join the training data path and each species training folder
        dir = os.path.join(train_path, training_name)
        # get the current training label
        current_label = training_name
        # loop over the images in each sub-folder
        for f in os.listdir(dir):
            # get the image file name
            file = os.path.join(dir, f)
            # read the image and return feature
            image = cv2.imread(file)
            global_feature = get_feature(image)

            # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)

        print("[STATUS] processed folder: {}".format(current_label))

    print("[STATUS] completed Global Feature Extraction...")

    # get the overall feature vector size
    print("[STATUS] feature vector size {}".format(
        np.array(global_features).shape))

    # get the overall training label size
    print("[STATUS] training Labels {}".format(np.array(labels).shape))

    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    print("[STATUS] training labels encoded...")

    # scale features in the range (0-1)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # rescaled_features = scaler.fit_transform(global_features)
    # print("[STATUS] feature vector normalized...")

    print("[STATUS] target labels: {}".format(target))
    print("[STATUS] target labels shape: {}".format(target.shape))

    global_features = np.array(global_features)
    global_labels = np.array(target)

    return global_features, global_labels
