import sklearn
from sklearn import svm
import tiled_liver_statistics

from tiled_liver_statistics import feat_hist, super_feat_hist

def f1(im):
    return super_feat_hist(im, 4)

fc = [
    #[feat_hist, svm.SVC],
    #[feat_hist, sklearn.naive_bayes.GaussianNB],
    #[f1, sklearn.naive_bayes.GaussianNB],
    [lambda im: super_feat_hist(im, 4), sklearn.naive_bayes.GaussianNB]
]

