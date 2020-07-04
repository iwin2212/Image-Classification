# -----------------------------------
# TRAINING OUR MODEL
# -----------------------------------
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.preprocessing import MinMaxScaler
from function import get_data_label, get_feature, get_train_label
from const import *
import time
warnings.filterwarnings('ignore')


def train_test():
    train_labels = get_train_label()
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    # create all the machine learning models
    model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

    global_features, global_labels = get_data_label()

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
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(
        model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    std = float(format((cv_results.std() * 100), ".2f"))
    mean = float(format((cv_results.mean() * 100), ".2f"))

    fig = plt.figure()
    fig.suptitle('Độ chính xác của thuật toán')
    ax = fig.add_subplot(111)
    plt.legend(loc=0)
    img_name = str(int(time.time()))
    plt.savefig(os.path.join(model_path, img_name))


# -----------------------------------
# TESTING OUR MODEL
# -----------------------------------
    clf = model
    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)

    # read the image
    for file in glob.glob(test_path + "/*.jpg"):
        image = cv2.imread(file)
        # resize the image
        image = cv2.resize(image, fixed_size)  # (500, 500, 3)
        global_feature = get_feature(image)
        # scale features in the range (0-1)
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # rescaled_features = scaler.fit_transform(
        #     global_feature.reshape(1, -1))[0]

        # predict label of test image
        prediction = clf.predict(global_feature.reshape(1, -1))[0]

        label = train_labels[prediction]
        # show predicted label on image
        cv2.putText(image, train_labels[prediction], (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # display the output image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.savefig(os.path.join(output_path, label))
        return label, mean, std, img_name
