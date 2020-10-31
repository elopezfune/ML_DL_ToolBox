import numpy as np   #Importing Numpy
import pandas as pd  #Importing Pandas
import scipy
import os, logging, datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


class ML_DL_Toolbox(BaseEstimator, RegressorMixin):
    '''Sci-kit learn wrapper for creating pseudo-lebeled estimators'''

    def __init__(self, seed=42, prob_threshold=0.9, num_folds=5, num_iterations=10):
        '''
        @seed: random number generator
        @prob_threshold: probability threshold to select pseudo labeled events
        @num_folds: number of folds for the cross validation analysis
        @num_iterations: number of iterations for labelling
        '''
        self.seed = seed
        self.prob_threshold = prob_threshold
        self.num_folds = num_folds
        self.num_iterations = num_iterations
        
        
    def get_params(self, deep=True):
        return {"seed": self.seed, "prob_threshold": self.prob_threshold, "num_folds": self.num_folds,
                "num_iterations": self.num_iterations}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __gets_best_model(self, X,target):
        best_classifiers=[]
        outer_cv = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=1)
        model_factory = [AdaBoostClassifier(), BaggingClassifier(), BayesianGaussianMixture(),
                         BernoulliNB(), CalibratedClassifierCV(), CategoricalNB(),
                         ComplementNB(), DecisionTreeClassifier(), ExtraTreesClassifier(),
                         GaussianMixture(), GaussianNB(), GaussianProcessClassifier(),
                         GradientBoostingClassifier(), KNeighborsClassifier(), LinearDiscriminantAnalysis(),
                         LogisticRegression(), LogisticRegressionCV(), MLPClassifier(), MultinomialNB(),
                         QuadraticDiscriminantAnalysis(), RandomForestClassifier(), SGDClassifier()
                        ]
        logging.basicConfig(filename="ml_dl_toolbox_logfilename.log", level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        try:
            for el in model_factory:
                #el.seed = self.seed
                scores = cross_val_score(el, X.drop(target,axis=1), X[target], cv=outer_cv, n_jobs=-1, scoring='neg_mean_squared_error')
                score_description = [el,'{el}'.format(el=el.__class__.__name__),"%0.5f" % abs(np.sqrt(scores.mean()*-1)), "%0.5f" % np.sqrt(scores.std() * 2)]
                best_classifiers.append(score_description)
                best_model=pd.DataFrame(best_classifiers,columns=["algorithm","model","RSME","\sigma"]).sort_values("RSME",axis=0, ascending=True)
                best_model=best_model.reset_index() 
        except OSError:
            logging.error('Check data structure')
        else:
            logging.info('Best fitting algorithm: '+ best_model["model"][0])
            return best_model["algorithm"][0]



    def pseudo_labeler(self, X, target):
        labeledX=X.loc[X[target]!=-1].copy()
        labeledX["temp_label"]=labeledX[target]
        unlabeledX=X.loc[X[target]==-1].copy()

        i=0
        while len(unlabeledX)>0 and i<=self.num_iterations:
            model = self.__gets_best_model(labeledX.drop(target,axis=1), "temp_label")
            classifier =  model.fit(labeledX.drop([target,"temp_label"],axis=1), labeledX["temp_label"]) 
            unlabeledX["probability"]=[np.linalg.norm(el) for el in classifier.predict_proba(unlabeledX.drop(target,axis=1))]
            unlabeledX["temp_label"] = classifier.predict(unlabeledX.drop([target,"probability"],axis=1))
            del model, classifier
            labeledX=pd.concat([labeledX,unlabeledX[unlabeledX["probability"]>=self.prob_threshold][labeledX.columns]],axis=0)
            unlabeledX = unlabeledX[unlabeledX["probability"]<self.prob_threshold][labeledX.columns].drop("temp_label",axis=1)
            i+=1

        if len(unlabeledX)==0:
            return labeledX[labeledX[target]==-1]
        else:
            parameters = {'num_leaves': 2**8,
                          'learning_rate': 0.01,
                          'is_unbalance': True,
                          'min_split_gain': 0.1,
                          'min_child_weight': 1,
                          'reg_lambda': 1,
                          'subsample': 1,
                          'objective':'binary',
                          #'device': 'gpu', # comment this line if you are not using GPU
                          'task': 'train'
                          }
            num_rounds = 300
            lgb_train = lgb.Dataset(labeledX.drop([target,"temp_label"],axis=1), labeledX["temp_label"], free_raw_data=False)
            Light_Gradient_Boosting = lgb.train(parameters, lgb_train, num_boost_round=num_rounds)
            unlabeledX["temp_label"]=np.where(Light_Gradient_Boosting.predict(unlabeledX.drop(target,axis=1)) > 0.5, 1, 0)
            labeledX=pd.concat([labeledX,unlabeledX[labeledX.columns]],axis=0)
            return labeledX[labeledX[target]==-1]
