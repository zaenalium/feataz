from feat_interaction import *
from feat_creation import *
import warnings 
import os

class AutoFeat:
    def __init__(self, train, test, numeric_variables, categorical_variables,target_variable, scope = ['encoding','discretization', 'woe', 'interaction', 'combination' ], directory = '.feataz') -> None:
        self.train = train
        self.test = test
        self.num_var = numeric_variables
        self.cat_var = categorical_variables
        self.scope = scope
        self.directory = directory
        self.target = target_variable

    def get_features(self, return_original = False):

        
        if not return_original:
            tr_all, ts_all = pd.DataFrame([]), pd.DataFrame([])
            if 'discretization' in self.scope:
                ts, tr = feat_discretiser(self.train, self.test, self.num_var, save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1)

            if 'encoding' in self.scope:
                ts, tr = feat_encoding(self.train, self.test, self.cat_var, y = self.target, save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1)

            if 'interaction' in self.scope:
                ts, tr = create_fi(self.train, self.test,num_var=self.num_var, cat_var=self.cat_var, save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1)

            if 'combination' in self.scope:
                ts, tr = feature_combination_calc(self.train, self.test, num_var=self.num_var, save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1)

            if 'woe' in self.woe:
                ts, tr = feat_sc_woe(self.train, self.test,y = self.target,  save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1)

        else:
            tr_all, ts_all =self.train, self.test
            if 'discretization' in self.scope:
                ts, tr = feat_discretiser(self.train, self.test, self.num_var, save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1)

            if 'encoding' in self.scope:
                ts, tr = feat_encoding(self.train, self.test, self.cat_var, y = self.target, save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1)

            if 'interaction' in self.scope:
                ts, tr = create_fi(self.train, self.test,num_var=self.num_var, cat_var=self.cat_var, save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1)

            if 'combination' in self.scope:
                ts, tr = feature_combination_calc(self.train, self.test, num_var=self.num_var, save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1)

            if 'woe' in self.woe:
                ts, tr = feat_sc_woe(self.train, self.test,y = self.target,  save_to_file= False, directory = self.directory)
                tr_all, ts_all = pd.concat([tr_all, tr], axis = 1),  pd.concat([ts_all, ts], axis = 1) 
        
        return tr_all, ts_all
    
    def save_to_file(self, sapararate_by_columns = True):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        if sapararate_by_columns:
            if 'discretization' in self.scope:
                feat_discretiser(self.train, self.test, self.num_var, save_to_file= True, directory = self.directory)

            if 'encoding' in self.scope:
                feat_encoding(self.train, self.test, self.cat_var, y = self.target, save_to_file= True, directory = self.directory)

            if 'interaction' in self.scope:
                create_fi(self.train, self.test,num_var=self.num_var, cat_var=self.cat_var, save_to_file= True, directory = self.directory)
            
            if 'combination' in self.scope:
                feature_combination_calc(self.train, self.test, num_var=self.num_var, save_to_file= True, directory = self.directory)

            if 'woe' in self.woe:
                feat_sc_woe(self.train, self.test,y = self.target,  save_to_file= True, directory = self.directory)
        else:
            warnings.warn('If you have limited RAM please consider sapararate_by_columns = True. Otherwise, it will store all data at once to the memory, it can cause out of memory error if you have large datasets')


    

