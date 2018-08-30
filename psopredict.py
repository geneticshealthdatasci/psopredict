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


""" Master script to read parameters file and run machine learning procedure"""


from __future__ import division
import sys
import paramreader
import ppmodeller
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


PARAMS_REQD = ['out', 'data', 'target', 'mode', 'model', 'scoring']

if not len(sys.argv) == 2:
    sys.exit('Exiting. Please include parameter file as argument')

# Handle parameters
print('Reading parameters')
pr = paramreader.ParamReader(sys.argv[1])
pr.check_params(PARAMS_REQD)
pr.log_params()
pr.interpret_params()
params = pr.get_params()

# Set random seed for reproducible analysis
random_seed = params['random_seed']
random_state = np.random.RandomState(random_seed)

# Load data
print('Loading data')
df = pd.read_hdf(params['data'])
print(df.shape)

X = df.drop(params['target'], axis=1)
pr.write_to_log('Loaded data comprising ' + str(X.shape[0]) +
                ' records with ' + str(X.shape[1]) + ' features', gap=True)
pr.write_to_log('Preview of predictor data:')
pr.write_to_log(X.iloc[0:5, 0:5], pandas=True, gap=True)

y = df[params['target']]
pr.write_to_log('Preview of outcome data:')
pr.write_to_log(y.iloc[0:5], pandas=True, gap=True)

pr.write_to_log('Removing ' + str(sum(pd.isnull(y))) +
                ' records due to null outcome values', gap=True)
X = X.loc[~pd.isnull(y), :]
y = y.loc[~pd.isnull(y)]

print(X.shape, y.shape)

# Split off test set and ignore
if params['mode'] == 'binary':
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params['test_pc'], random_state=random_state,
            stratify=y)
elif params['mode'] == 'continuous':
    sys.exit('Exiting. Stratified train/test split for continous outcomes ' +
             'not yet implemented')
else:
    sys.exit('Exiting. "mode" parameter must be one of [binary/continuous]')


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# Using training set only, want to perform repeated CV

def run_gridsearch(fm_random_state):
    """
    Method to run gridsearch from global parameters using supplied random state
    """
    modeller = ppmodeller.PPModeller(model_string=params['model'],
                                     score_string=params['scoring'],
                                     n_folds=params['cv_folds'][0],
                                     mode=params['mode'],
                                     param_grid=params['param_grid'],
                                     random_state=fm_random_state)
    modeller.fit_gridsearch(X_train, y_train)
    return modeller

print('Fitting models')

if params['simple_mode']:
    
    pr.write_to_log('WARNING: In simple mode, cross-validation is not ' +
                    'repeated; parameter for number of repeats is ignored',
                    gap=True)
    
    random_states = [np.random.RandomState(random_seed + 1)]
    post_GS_modeller = map(run_gridsearch, random_states)[0]
    
    fitted_params = post_GS_modeller.gridsearch.best_params_
    post_GS_modeller.fix_params(fitted_params)
    
    train_results = post_GS_modeller.new_fit(X_train, y_train)
    
    print(train_results)
    
    print('HOW DO WE KNOW THAT MODEL IS FITTED CORRECTLY AFTER GRIDSEARCH?')
    print('WANT TO CHECK BY OUTPUTTING CV RESULTS AND CHECKING AUROC')
    print('ALSO WANT TO CHANGE OUTPUT SO NO LONGER GET Y_PRED BUT SUMMARY' +
          'RESULTS AND ROC VALUES')
    
#    print('')
#    print(fitted_GS.gridsearch.cv_results_)
#    print(fitted_GS.gridsearch.best_params_)
#    print('')
    
    
    
#    y_train_pred = fitted_GS.refit(X_train, y_train)
#    print(y_train_pred)
#    print(y_train_pred['y_prob'].__class__)
#    print(y_train.__class__)
#    print(fitted_GS.get_model().get_params(deep=False))
    
#    scorer = fitted_GS.get_scorer()
#    print(scorer(y_train, y_train_pred))

else:
    sys.exit('Exiting: non- simple mode not fully implemented')
    
    random_seeds = map(lambda x: random_seed + x + 1,
                       range(params['cv_folds'][1]))
    random_states = map(lambda x: np.random.RandomState(x), random_seeds)
    
    print(random_seeds)
    print(random_states)
    print(params['param_grid'])

    fitted_GSs = map(run_gridsearch, random_states)
    
    print(len(fitted_GSs))
    for fitted_GS in fitted_GSs:
        print('')
        print('')
        #print(fitted_GS.gridsearch.best_params_)
        print(fitted_GS.gridsearch.cv_results_)
    
    for fitted_GS in fitted_GSs:
        print(fitted_GS.gridsearch.best_params_)
