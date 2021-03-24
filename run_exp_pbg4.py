from estimator_selector import EstimatorSelectionHelper
import logging
import numpy as np
from optparse import OptionParser
import sys
import os
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
from sklearn.metrics import classification_report
from upbg import UPBG
import pickle as pk

def evaluate(model, M_test, y_test, target_names):
    y_pred = model.predict(M_test)    
    s = classification_report(y_test, y_pred, target_names=target_names)
    return s


if __name__ == "__main__":

    alpha = float(sys.argv[1])
    beta = float(sys.argv[2])    
    dts = sys.argv[3]
    dir_out = sys.argv[4]
    k2 = int(sys.argv[5])
    ## load dataset

    (M_train , M_test, y_train, y_test, target_names) = load_dataset(dts, subset=True)
    
    M_train, M_test = text_preproc(M_train), text_preproc(M_test)
    vect = CountVectorizer()
    M_train = vect.fit_transform(M_train)
    M_test = vect.transform(M_test)    

    #for alpha,beta in [(x,y) for x in [0.05, 0.005, 0.0005, 0] for y in [0.01, 0.001, 0.0001, 0]]:
    lmitr = 50
    gmitr = 50
    for k in [k2]:
        print(k,alpha,beta,lmitr, gmitr)
        pbg = UPBG(k, alpha=alpha, beta=beta, local_max_itr=lmitr, global_max_itr=gmitr,
                            local_threshold=1e-6, global_threshold=1e-6,
                            feature_names=vect.get_feature_names())
        
        ## parâmetros de configuração
        pbg.evaluate = evaluate
        pbg.M_test = M_test
        pbg.y_test = y_test
        pbg.target_name = target_names
        pbg.ld = []
        pbg.save_interval = -2

        print('fitting.....')
        pbg.fit(M_train, y_train)
        print('DONE!')
        print('salvando resultados...')
        arq_result = os.path.join(dir_out , f'out_pbg4_e20ng_{alpha}_{beta}_{lmitr}_{gmitr}_{k}')
        with open(arq_result, 'wb') as f:
            pk.dump(pbg.ld, f)     
        print('DONE!')
        print('Salvando matrizes')
        arq_A = os.path.join(dir_out , f'A_pbg3_e20ng_{alpha}_{beta}_{lmitr}_{gmitr}_{k}')
        np.save(arq_A, pbg.transform(M_train))
        print('DONE!!')
        