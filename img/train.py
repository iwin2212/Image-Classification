# -----------------------------------
# TRAINING OUR MODEL
# -----------------------------------
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import glob
import cv2
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from sklearn.preprocessing import MinMaxScaler
import joblib
from function import fv_hu_moments, fd_hu_moments, fd_haralick, fd_histogram
from const import *
import time
warnings.filterwarnings('ignore')


def train():
    # get the training labels
    train_labels = os.listdir(train_path)

    # sort the training labels
    train_labels.sort()

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # create all the machine learning models
    models = []
    # models.append(('LR', LogisticRegression(random_state=seed)))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    # models.append(('RF', RandomForestClassifier(
    #     n_estimators=num_trees, random_state=seed)))
    # models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))

    # variables to hold the results and names
    results = []
    names = []

    # import the feature vector and trained labels
    h5f_data = h5py.File(h5_data, 'r')
    h5f_label = h5py.File(h5_labels, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    # verify the shape of the feature vector and labels
    print("[STATUS] features shape: {}".format(global_features.shape))
    print("[STATUS] labels shape: {}".format(global_labels.shape))

    print("[STATUS] training started...")

    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(
        np.array(global_features), np.array(global_labels), test_size=test_size, random_state=seed)

    print("[STATUS] splitted train and test data...")
    print("Train data  : {}".format(trainDataGlobal.shape))
    print("Test data   : {}".format(testDataGlobal.shape))
    print("Train labels: {}".format(trainLabelsGlobal.shape))
    print("Test labels : {}".format(testLabelsGlobal.shape))

    # 10-fold cross validation
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(
            model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Độ chính xác của thuật toán')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.legend(loc=0)
    img_name = str(int(time.time()))
    plt.savefig(os.path.join(output_path, img_name))
    return models, trainDataGlobal, trainLabelsGlobal, train_labels, results, img_name