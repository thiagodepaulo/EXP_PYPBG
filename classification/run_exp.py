from estimator_selector import EstimatorSelectionHelper
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from util import load_dataset, text_preproc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer

## load dataset
(M, y, y_names) = load_dataset('agnews')

pre_steps = [('preproc',None),('vect',None)]
vect_param = [{ 
    'preproc':[None,FunctionTransformer(text_preproc)],
    'vect':[TfidfVectorizer()],
    'vect__max_df':[0.4, 0.6, 0.8, 1.0],
    },{
    'preproc':[None,FunctionTransformer(text_preproc)],
    'vect':[CountVectorizer()]
    }]

models_param_grid = {
            'mnb':{ 'mnb__alpha':[1,0.1,0.01,0.001] },
            'knn':{'knn__n_neighbors':[1,5,10,50,100]},
            'lsvc':{'lsvc__penalty':['l1','l2']},
            'sgd':{'sgd__penalty':['l1','l2']},
            'cnb':{ 'cnb__alpha':[1,0.1,0.01,0.001] },
            'ridge':{}, 'perceptron':{}, 'pa':{}, 'rf':{}, 'nc':{}, 'cnb':{}
}
models = {
            'mnb':MultinomialNB(),
            'ridge':RidgeClassifier(tol=1e-2, solver="sag"),
            'perceptron':Perceptron(max_iter=50),
            'pa':PassiveAggressiveClassifier(max_iter=50),
            'knn':KNeighborsClassifier(n_neighbors=10),
            'rf':RandomForestClassifier(),
            'lsvc':LinearSVC(dual=False, tol=1e-3),
            'sgd':SGDClassifier(alpha=.0001, max_iter=50),
            'nc':NearestCentroid(),
            'cnb':ComplementNB()
        } 

params = {}
steps = {}
for key in models:
    steps[key] = Pipeline(pre_steps + [(key,models[key])])
    params[key] = [{**map, **models_param_grid[key]} for map in vect_param]


estimator = EstimatorSelectionHelper(steps,params)
estimator.fit(M,y, n_jobs=-1)
df = estimator.score_summary()
df.to_csv('teste3.csv')