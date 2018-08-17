## Copyright 2018 Nick Dand
##
## This file is part of psopredict
##
## psopredict is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## psopredict is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with psopredict.  If not, see <https://www.gnu.org/licenses/>.


"""Module in which appropriate model and scoring scheme are selected"""

import sys
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV


class PPModeller:

    """Class for specifying model and scorer"""

    def __init__(self, model_string, score_string, n_folds, mode,
                 param_grid, random_state):
        sys.exit('IMPLEMENT SEPARATE MODELLERS BY MODEL TYPE')
        self.random_state = random_state
        self._set_model(model_string)
        self._set_scorer(score_string)
        self._set_CVs(n_folds, mode)
        self.gridsearch = GridSearchCV(self.model, param_grid,
                                       scoring=self.scorer, cv=self.cvs,
                                       refit=False, return_train_score=False)
        self.fitted = False

    def _set_scorer(self, score_string):
        if score_string == 'roc_auc':
            self.scorer = make_scorer(roc_auc_score, needs_threshold=True)
            self.needs_threshold = True
        else:
            sys.exit('Exiting. Unsupported scoring type')

    def get_scorer(self):
        return self.scorer

    def _set_model(self, model_string):
        self.pred_class = None
        self.pred_numer = None
        if model_string == 'logistic':
            self.model = LogisticRegression(random_state=self.random_state)
            self.pred_class = self.model.predict
            self.pred_numer = self.model.predict_proba
        elif model_string == 'SVC':
            self.model = SVC(random_state=self.random_state)
            self.pred_class = self.model.predict
            self.pred_numer = self.model.decision_function
        else:
            sys.exit('Exiting: Unsupported model type')

    def _set_CVs(self, n_folds, mode):
        if mode == 'binary':
            self.cvs = StratifiedKFold(n_folds, shuffle=True,
                                       random_state=self.random_state)
        else:
            sys.exit('Exiting. Stratified CV not implemented for cts outcome')

    def fit(self, X, y):
        self.gridsearch.fit(X, y)
        self.fitted = True

    def refit(self, X, y):
        sys.exit('This pred_numer thing isn\'t working for SVC!!')
        if self.fitted:
            self.model.set_params(**self.gridsearch.best_params_)
            self.model.fit(X, y)
            if self.needs_threshold:
                return self.pred_numer(X)
            else:
                return self.pred_class(X)
        else:
            sys.exit('Exiting: Cannot refit model before grid search!')
