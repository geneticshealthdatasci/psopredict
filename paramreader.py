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


"""Class of object that reads parameters file and handles logging"""


import sys
import pandas as pd

class ParamReader:
    
    """Reads and checks parameters file, handles logging"""
    
    def __init__(self, params_filename):
        self.params = self.__read_params(params_filename)
        self.check_params(['out'])
        self.log = self.params['out'] + '.log'
        with open(self.log, 'w') as f:
            f.writelines('Parameters file: ' + params_filename + '\n')

        
    def __read_params(self, pfile):
        
        """Reads paramaters file"""
        
        ps = {}
        f = open(pfile)
        for line in f.readlines():
            if (not line[0] == '#') and (len(line.strip()) > 0):
                tokens = line.strip().split('\t')
                if not len(tokens) == 2:
                    sys.exit('Exiting. Ensure parameters file comprises two ' +
                             'columns (tab separated)')
                ps[tokens[0]] = tokens[1]
        f.close()
        return ps

        
    def check_params(self, params_reqd):
        
        """Checks for provided list of required parameters"""
        
        for param_reqd in params_reqd:
            if not param_reqd in self.params:
                sys.exit('Exiting. Ensure paramater "' + param_reqd +
                         '" specified.')

    
    def interpret_params(self):
        
        """Convert numeric parameters; populate defaults where needed"""
        
        defaulted = False
        
        if 'simple_mode' in self.params:
            if self.params['simple_mode'] in ['True', 'T', 'TRUE']:
                self.params['simple_mode'] = True
            elif self.params['simple_mode'] in ['False', 'F', 'FALSE']:
                self.params['simple_mode'] = False
            else:
                sys.exit('Exiting. Cannot interpret simple_mode parameter')
        else:
            self.params['simple_mode'] = True
            self.write_to_log('Simple mode defaults to True')
            defaulted = True

        if 'random_seed' in self.params:
            self.params['random_seed'] = int(self.params['random_seed'])
        else:
            self.params['random_seed'] = 0
            self.write_to_log('Random seed defaults to 0')
            defaulted = True
        
        if 'test_pc' in self.params:
            self.params['test_pc'] = float(self.params['test_pc'])
        else:
            self.params['test_pc'] = 0.3
            self.write_to_log('Test set percentage defaults to 0.3')
            defaulted = True
        
        if 'cv_folds' in self.params:
            tokens = self.params['cv_folds'].split(' ')
            if len(tokens) == 2:
                self.params['cv_folds'] = (int(tokens[0]), int(tokens[1]))
            else:
                self.params['cv_folds'] = (int(tokens[0]), 1)
        else:
            self.params['cv_folds'] = (5, 3)
            self.write_to_log('CV defaults to 5-fold repeated 3 times')
            defaulted = True

        if 'param_grid' in self.params:
            param_grid = {}
            tokens = self.params['param_grid'].split(' ')
            for token in tokens:
                param = token.split(':')
                if len(param) == 2:
                    param_vals = param[1].split(',')
                    param_vals = map(float, param_vals)
                    param_grid[param[0]] = param_vals
                else:
                    sys.exit('Exiting. param_grid is not correctly specified')
            self.params['param_grid'] = param_grid
        else:
            if self.params['model'] == 'logistic':
                grid = {'C': map(lambda x: 10 ** x, range(-5, 6))}
            elif self.params['model'] == 'SVC':
                grid = {'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
                        'gamma': [0.001, 0.0001]}
            else:
                sys.exit('Exiting. No default parameter grid set for model')
            self.params['param_grid'] = grid
            self.write_to_log('For ' + self.params['model'] + ' model type, ' +
                              'param_grid defaults to ' +
                              str(self.params['param_grid']))
            defaulted = True

        if defaulted:
            self.write_to_log('')


    def get_params(self):
        
        """Makes parameters externally accessible"""
        
        return self.params
    
    
    def log_params(self):
        
        """Writes out a list of parameters"""
        
        self.write_to_log('Parameters:')
        for param in self.params:
            self.write_to_log('  ' + param + ': ' + self.params[param])
        self.write_to_log('')
        

    def write_to_log(self, content='', pandas=False, gap=False):
        
        """Writes to log file in a "safe" manner"""
        
        with open(self.log, 'a') as f:
            if pandas:
                f.writelines(content.to_string() + '\n')
            else:
                f.writelines(str(content) + '\n')
            
            if gap:
                f.writelines('\n')
