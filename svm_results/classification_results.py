import numpy as np
import feature_extraction_utils as feu


toy = "green"
toy_color = "green"

# extract features
my_dir = "../samples/" + toy + "/"
X, y, img_num = feu.get_train_features(my_dir, toy_color)
y /= 50
y[y != 2] = 0
y[y == 2] = 1

