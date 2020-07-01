import cv2
import glob
from function import fv_hu_moments, fd_hu_moments, fd_haralick, fd_histogram
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# -----------------------------------
# TESTING OUR MODEL
# -----------------------------------

# to visualize results


def test(models, trainDataGlobal, trainLabelsGlobal, train_labels):
    # create the model - Random Forests
    for model in models:
        clf = model

    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)

    # loop through the test images
    for file in glob.glob(test_path + "/*.jpg"):
        # read the image
        image = cv2.imread(file)
        # resize the image
        image = cv2.resize(image, fixed_size)  # (500, 500, 3)

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

        # scale features in the range (0-1)
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # rescaled_feature = scaler.fit_transform(global_feature.reshape(-1,1))

        # predict label of test image
        prediction = clf.predict(global_feature.reshape(1, -1))[0]

        # show predicted label on image
        cv2.putText(image, train_labels[prediction], (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # display the output image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.savefig()
