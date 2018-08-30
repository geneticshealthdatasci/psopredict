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


"""Module in which Logistic Regression model/scoring scheme are implemented"""

import sys
#from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import StratifiedKFold, GridSearchCV


class LogRegModeller:

    """Class for implementing Logistic Regression model and scorer"""

    def __init__(self, mode, random_state):
        if mode not in ['binary']:
            sys.exit('Exiting. Logistic regression not supported for mode ' +
                     mode)
        self.model = LogisticRegression(random_state=random_state)
    
    def fix_params(self, params):
        
        """Fix parameters for subsequent model fits"""
        
        self.model.set_params(**params)

    def new_fit(self, X, y):

        """
        Fits model with best parameters and returns a dict containing:
        predicted probabilities
        """
        
        self.model.fit(X, y)
        output = {}
        output['y_prob'] = self.model.predict_proba(X)
        return output
