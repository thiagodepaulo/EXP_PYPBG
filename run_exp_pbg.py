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
from upbg import UPBG


## load dataset
(M, y, y_names) = load_dataset('20ng')

pre_steps = [('preproc',None),('vect',None)]
vect_param = [{ 
    'preproc':[None,FunctionTransformer(text_preproc)],
    'vect':[TfidfVectorizer()],
    'vect__max_df':[0.4, 0.6, 0.8, 1.0],
    },{
    'preproc':[None,FunctionTransformer(text_preproc)],
    'vect':[CountVectorizer()]
    }]

for alpha,beta in [(x,y) for x in [0.5,0.05,0.005,0.0005,0] for y in [0.1, 0.01, 0.001, 0.0001, 0]]:
    l1,l2=[],[]
    
    models_param_grid = {
            'pbg':{ 'pbg__alpha': l1.append(alpha),
            'pbg__beta':l2.append(beta),
            'pbg__local_max_itr':[1,2,5,10,50],
            'pbg__global_max_itr':[1,2,5,10,50],
            'pbg__local_threshold':[1e-6],
            'pbg__global_threshold':[1e-6],
            'pbg__n_components':[50,100,200,500]}
    }   
    models = {
            'pbg':UPBG(100)            
        } 

    params = {}
    steps = {}
    for key in models:
        steps[key] = Pipeline(pre_steps + [(key,models[key])])
        params[key] = [{**map, **models_param_grid[key]} for map in vect_param]


    estimator = EstimatorSelectionHelper(steps,params)
    estimator.fit(M,y, n_jobs=-1)
    df = estimator.score_summary()
    df.to_csv(f'teste_pbg_{alpha}_{beta}.csv')