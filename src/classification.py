# ! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


class GMMClassifier():
    def __init__(self):

        self.models = []

    def fit(self, X_train, y_train):
        from sklearn.mixture import GMM
        unlabels = range(0, np.max(y_train)+1)
        for lab in unlabels:
            model = GMM()
            model.fit(X_train[y_train == lab])
            self.models.insert(lab, model)

    def predict(self, X_test):
        scores = np.zeros([X_test.shape[0], len(self.models)])
        for lab in range(0, len(self.models)):

            sc = self.models[lab].score(X_test)
            scores[:, lab] = sc
        pred = np.argmax(scores, 1)
        return pred
