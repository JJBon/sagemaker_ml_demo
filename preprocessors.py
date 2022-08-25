import numpy as np
import cv2
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, encoder = LabelEncoder()):
        self.encoder = encoder

    def fit(self, X, y=None):
        # note that x is the target in this case
        self.encoder.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = np_utils.to_categorical(self.encoder.transform(X))
        return X
    
    
def _im_resize(df, n, image_size):
    im = cv2.imread(df[n])
    im = cv2.resize(im, (image_size, image_size))
    return im