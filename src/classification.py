# ! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


class GMMClassifier():

    def __init__(self, each_class_params=None, **same_params):
        """
        same_params: classifier params for each class are same
        each_class_params: is list of dictionary of params for each
            class classifier. For example:
            [{'covariance_type': 'full'}, {'n_components': 2}])
        """
        self.same_params = same_params
        self.each_class_params = each_class_params

        self.models = []

    def fit(self, X_train, y_train):
        from sklearn.mixture import GMM

        unlabels = range(0, np.max(y_train) + 1)

        for lab in unlabels:
            if self.each_class_params is not None:
                # print 'eacl'
                # print self.each_class_params[lab]
                model = GMM(**self.each_class_params[lab])
                # print 'po gmm ', model
            elif len(self.same_params) > 0:
                model = GMM(**self.same_params)
                # print 'ewe ', model
            else:
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
