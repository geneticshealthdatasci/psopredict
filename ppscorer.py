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


"""Generate custom scoring object to collect TP/TN/FP/FN"""

import sys
from sklearn.metrics import make_scorer


class PPScorer:
    
    """Class for custom scorer"""
    
    def __init__(self, real_scorer):
        
        """
        The argument real_scorer should be the scorer on which hyperparameter
        optimisation is based, e.g. roc_auc object from sklearn
        """
        
        self.ys = []
        self.real_scorer = real_scorer
        suffix = self.real_scorer.__name__.split('_')[-1]
        if suffix == 'score':
            self.greater_is_better = True
        elif suffix == 'error' or suffix == 'loss':
            self.greater_is_better = False
        else:
            sys.exit('Exiting. Cannot infer which way to optimise ' +
                     'objective function.')
    
    
    def collect_and_score(self, y_true, y_pred):
   
        """Collect the predicted values before calculating score"""
        
        self.ys.append(y_pred)
        return self.real_scorer(y_true=y_true, y_pred=y_pred)
    
    
    def custom_scorer(self):
        
        """The scorer that can be passed into GridSearchCV etc"""
        
        return make_scorer(self.collect_and_score,
                           greater_is_better=self.greater_is_better)


    def get_ys(self):
        return self.ys