#!/usr/bin/env python
# encoding:utf-8

'''sklearn config file
'''

import numpy as np
from sklearn.preprocessing import *
# LabelEncoder, LabelBinarizer, Binarizer, \
#                         MinMaxScaler, MaxAbsScaler, \
#                         StandardScaler, RobustScaler, Normalizer, \
#                         QuantileTransformer, PowerTransformer, FunctionTransformer


from utils import NonScaler

from sklearn.svm import LinearSVC 
# SVC The multiclass support is handled according to a one-vs-one scheme
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,\
                    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
# from lightgbm import plot_importance

RANDOM_STATE = 0
SCALERS = [
    ('Non', NonScaler()),
    ('Binarizer', Binarizer(threshold=0)),
    ('MinMaxScaler', MinMaxScaler()), #relative abundance matrix not necessary to scale
    ('MaxAbsScaler', MaxAbsScaler()), # specifically designed for scaling sparse data, not necessary
    ('StandardScaler', StandardScaler()),
    ('RobustScaler', RobustScaler(quantile_range=(25, 75),with_centering=False)), # (false: not remove median),中位数避免outliers effect
    ('PowerTransformer_YeoJohnson', PowerTransformer(method='yeo-johnson',standardize=False)), #generate -nums
    ('QuantileTransformer_Normal', QuantileTransformer(output_distribution='normal')), # -nums
    ('QuantileTransformer_Uniform', QuantileTransformer(output_distribution='uniform')),
    ('Normalizer', Normalizer()), #0 remains 
    ('Log1p', FunctionTransformer(np.log1p,validate=False)), # ori abundance data
]

"""
('PowerTransformer_BoxCox', PowerTransformer(method='box-cox')), #The Box-Cox transformation can only be applied to strictly positive data >0
https://github.com/shaygeller/Normalization_vs_Standardization/blob/master/src/models/expirament_utils.py
Distance-based classifiers like SVM, KNN, and MLP(neural network)

SVM, MLP, KNN, and NB got a significant boost from different scaling methods
Distance-based classifiers like SVM, KNN, and MLP(neural network) dramatically benefit from scaling

(BernoulliNB(),dict()),  #dict(clf__alpha=[0.001, 0.01, 0.1, 0.2, 0.5, 1])), #先验为伯努利分布的朴素贝叶斯 binary/boolean features
(MultinomialNB(),dict()), #alpha = 1 is called Laplace smoothing[default] works with occurrence counts suitable for classification with discrete features (e.g., word counts for text classification)
NB is not affected because the model's priors determined by the count in each class and not by the actual value.
https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e
"""


#feature_importances_
#A notable exception are decision tree-based estimators that are robust to arbitrary scaling of the data
Tree_based_CLASSIFIERS = [     
    (DecisionTreeClassifier(random_state=RANDOM_STATE), 
     dict(clf__max_depth=list(map(int, np.logspace(2, 6, 5, base=2))))), #[4, 5, 8, 11, 16]

    (BaggingClassifier(random_state=RANDOM_STATE),  #bagged trees
     dict(clf__n_estimators=list(map(int, np.linspace(5, 50, 10))))),

    (GradientBoostingClassifier(),  #boosted trees
     dict(clf__learning_rate=[0.001, 0.01, 0.1, 0.2, 0.5])), #[0.05..0.3] [0.1 default]

    (AdaBoostClassifier(random_state=RANDOM_STATE), 
     dict(clf__learning_rate=[0.001, 0.01, 0.1, 0.2, 0.5])),

    (RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE),  #random forest
     dict(clf__max_depth=list(map(int, np.logspace(2, 6, 5, base=2))))),

    (ExtraTreesClassifier(random_state=RANDOM_STATE), 
     dict(clf__max_depth=list(map(int, np.logspace(2, 6, 5, base=2))))), #[3-10]

    (XGBClassifier(random_state=RANDOM_STATE), 
     dict(clf__max_depth=list(map(int, np.logspace(2, 6, 5, base=2))), 
          clf__min_child_weight=range(1,6,2))),

    (LGBMClassifier(random_state=RANDOM_STATE), 
     dict(clf__max_depth=list(map(int, np.logspace(5, 6, 5, base=2))))) # min_child_weight=0.001: 分支结点的最小权重 num_leaves : int, optional (default=31)
]


Other_CLASSIFIERS = [
    (KNeighborsClassifier(),  #predict_proba
     dict(clf__n_neighbors=list(map(int, np.linspace(5, 20, 4))))),

    (GaussianNB(),dict()),   #predict_proba predict_log_proba

    (LogisticRegression(penalty='elasticnet',  l1_ratio=0.15, solver='saga', multi_class='auto',random_state=RANDOM_STATE),  #coef_
     dict(clf__C=list(np.logspace(-4, 4, 3)))), #l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1

    (LinearSVC(random_state=RANDOM_STATE),  #coef_
     dict(clf__C=list(np.logspace(-4, 4, 3)))),

#    (MLPClassifier(max_iter=10000, random_state=RANDOM_STATE),  # neural networks can easily overfit the data
#     dict(clf__alpha = [0.1, 0.001, 0.0001], 
#          clf__solver = ["lbfgs", "sgd", "adam"])), #multilayer_perceptron

    (SGDClassifier(penalty='elasticnet',  l1_ratio=0.15, random_state=RANDOM_STATE),  #svm the data should have zero mean and unit variance. coef_
     dict(clf__loss=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'])) #Defaults to 'hinge', which gives a linear SVM.
]
