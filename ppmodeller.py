# Copyright 2018 Nick Dand
#
# This file is part of psopredict
#
# psopredict is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# psopredict is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with psopredict.  If not, see <https://www.gnu.org/licenses/>.


"""Module in which appropriate model and scoring scheme are selected"""

import sys
#from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import logregmodeller
import svcmodeller


class PPModeller:

    """Class for specifying model and scorer"""

    def __init__(self, model_string, score_string, n_folds, mode,
                 param_grid, random_state):

        if score_string not in ['roc_auc']:
            sys.exit('Exiting. Unsupported scoring type')

        if model_string == 'logistic':
            self.x = logregmodeller.LogRegModeller(mode, random_state)
        elif model_string == 'SVC':
            self.x = svcmodeller.SVCModeller(mode, random_state)
        else:
            sys.exit('Exiting. Unsupported model type')

        self._set_CVs(n_folds, mode, random_state)
        self.gridsearch = GridSearchCV(self.x.model, param_grid,
                                       scoring=score_string, cv=self.cvs,
                                       refit=False, return_train_score=False)
        self.params_fixed = False

    def _set_CVs(self, n_folds, mode, random_state):
        if mode == 'binary':
            self.cvs = StratifiedKFold(n_folds, shuffle=True,
                                       random_state=random_state)
        else:
            sys.exit('Exiting. Stratified CV not implemented for mode ' + mode)

    def fit_gridsearch(self, X, y):
        self.gridsearch.fit(X, y)

    def fix_params(self, params):
        self.x.fix_params(params)
        self.params_fixed = True

    def new_fit(self, X, y):
        if not self.params_fixed:
            sys.exit('Cannot perform new fit until parameters fixed.')
        return self.x.new_fit(X, y)

    def get_model(self):
        return self.x.model
