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

# link
path = 'D:/learning/XLA/Image-Classification/img'

train_path = os.path.join(path, 'dataset/train/')
test_path = os.path.join(path, 'dataset/test/')

output_path = os.path.join(path, 'static/results')
model_path = os.path.join(path, 'static/model')
