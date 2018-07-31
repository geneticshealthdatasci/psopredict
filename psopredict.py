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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


PARAMS_REQD = ['out', 'data', 'target', 'mode']

if not len(sys.argv) == 2:
    sys.exit('Exiting. Please include parameter file as argument')

## Handle parameters
pr = paramreader.ParamReader(sys.argv[1])
pr.check_params(PARAMS_REQD)
pr.log_params()
pr.interpret_params()
params = pr.get_params()

## Set random seed for reproducible analysis
random_state = np.random.RandomState(params['random_seed'])

## Load data
df = pd.read_hdf(params['data'])

## Split off test set and ignore
X_train, X_test, y_train, y_test = train_test_split(
        df.drop(params['target'], axis=1),
        df[params['target']],
        test_size = params['test_pc'], random_state = random_state,
        stratify = params['target'])

print(X_train.size)

pr.write_to_log(df.head(), pandas=True)