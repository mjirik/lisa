# ! /usr/bin/python
# -*- coding: utf-8 -*-
# import funkcí z jiného adresáře
# import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
# sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
# sys.path.append(os.path.join(path_to_script, "../src/"))

import unittest
import numpy as np
import lisa.classification


class OrganSegmentationTest(unittest.TestCase):
    def test_gmmclassifier(self):

        X_tr = np.array([1, 2, 0, 1, 1, 0, 7, 8, 9, 8, 6, 7]).reshape(-1, 1)
        y_tr = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(-1)

        X_te = np.array([1, 7, 8]).reshape(-1, 1)
        y_te = np.array([0, 1, 1]).reshape(-1)

        # cl = GMMClassifier(each_class_params=[{},{}])
        cl = lisa.classification.GMMClassifier(each_class_params=[
            {'covariance_type': 'full'},
            {'n_components': 2}])
        cl.fit(X_tr, y_tr)

        y_te_pr = cl.predict(X_te)
        self.assertTrue((y_te_pr == y_te).all())


if __name__ == "__main__":
    unittest.main()
