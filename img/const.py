import os
# --------------------
# tunable-parameters
# --------------------
images_per_class = 80
bins = 8
fixed_size = tuple((500, 500))
num_trees = 100
test_size = 0.10
seed = 9
scoring = "accuracy"

path = 'D:/learning/XLA/Image-Classification/img'

train_path = os.path.join(path, 'dataset/train/')
test_path = os.path.join(path, 'dataset/test/')

h5_data = os.path.join(path, 'output/data.h5')
h5_labels = os.path.join(path, 'output/labels.h5')

output_path = os.path.join(path, 'static/results')
model_path = os.path.join(path, 'static/model')

