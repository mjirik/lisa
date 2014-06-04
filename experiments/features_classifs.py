import sklearn
from sklearn import svm
import tiled_liver_statistics

from tiled_liver_statistics import feat_hist, super_feat_hist, lbp3d, f_lbp3d

def f1(im):
    return super_feat_hist(im, 4)
    

fc = [
    #[feat_hist, svm.SVC],
    #[feat_hist, sklearn.naive_bayes.GaussianNB],
    #[f1, sklearn.naive_bayes.GaussianNB],
    #[lambda im: super_feat_hist(im, 4), svm.SVC],
    #[lambda im: super_feat_hist(im, 4), sklearn.naive_bayes.GaussianNB]
    #[lambda im: f_lbp3d(im), svm.SVC],
    [lambda im: lbp3d(im, '/home/petr/Dokumenty/git/lbpLibrary/masks/mask3D_8_4.json', True), svm.SVC]
]

