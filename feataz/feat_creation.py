import scorecardpy as sc
import pandas as pd
import numpy as np
import glob
import os
from feature_engine.discretisation import *
from feature_engine.encoding import *
from feature_engine.selection import DropConstantFeatures
from itertools import combinations
from feat_interaction import FeatInteraction
import warnings
from feature_engine.creation import RelativeFeatures, MathFeatures, CyclicalFeatures
from feature_engine.selection import DropConstantFeatures



def feat_discretiser(train, test, num_var, y = None, method = ['efd', 'ewd', 'dtd', 'gwd'], bins = 10, save_to_file = False, directory = '.feataz'):
    

    if 'dtd' in method and y == None:
        raise Exception("If you are using a dtd (DecisionTreeDiscretiser) please provide the valid target!!")
    
    efd = EqualFrequencyDiscretiser(q = bins, variables=num_var)
    ewd = EqualWidthDiscretiser(bins = bins, variables=num_var)
    dtd = DecisionTreeDiscretiser(cv = 3, 
                                    scoring='neg_mean_squared_error',
                                    variables = num_var,
                                    regression = False)
    gwd = GeometricWidthDiscretiser(bins = bins,
                                      variables = num_var)
    

    method_all = {'efd': efd,
                 'ewd': ewd,
                 'dtd' : dtd,
                  'gwd': gwd
                 }
    tr_all, ts_all = pd.DataFrame([]), pd.DataFrame([])
    for i in method:
        dsc = method_all.get(i)
        if i == 'dtd':
            dsc.fit(train[num_var], train[y])
        else:
            dsc.fit(train[num_var])
            
        train_dsc = dsc.transform(train[num_var])
        test_dsc = dsc.transform(test[num_var])
        
        if i == 'dtd':
            train_dsc = train_dsc.rank(method = 'dense')
            test_dsc = test_dsc.rank(method = 'dense')
            
        train_dsc.columns=  [f'{i}_{x}' for x in num_var]
        test_dsc.columns =  [f'{i}_{x}' for x in num_var]


        if save_to_file:    
            for j in train_dsc.columns:
                train_dsc[[j]].to_parquet(f'{directory}/train_{j}.parquet')
                test_dsc[[j]].to_parquet(f'{directory}/test_{j}.parquet')
            
            print(f'Done: features saved in {directory} directory!')

        else:
            tr_all, ts_all = pd.concat([tr_all, train_dsc], axis = 1), pd.concat([ts_all, test_dsc], axis = 1)

    return tr_all, ts_all

def feat_encoding(train, test, cat_var, y = None, method = ['ohe', 'cfe', 'me', 'woe', 'dte'], save_to_file = False, directory = '.feataz'):
    
    if ('dte' in method or 'me' in method or 'woe' in method) and y == None:
        raise Exception("If you are using a dtd (DecisionTreeDiscretiser) please provide the valid target!!")
    
    ohe = OneHotEncoder(
        top_categories=3,
        variables= cat_var,
        drop_last_binary=True
    )
    
    cfe = CountFrequencyEncoder(
        encoding_method = 'frequency',
        variables = cat_var,
        ignore_format = True
    )
    
    
    me = MeanEncoder(
        variables = cat_var,
        ignore_format = True
    )
    
    rle = RareLabelEncoder(
        tol=0.1,
        n_categories=2,
        variables=cat_var,
        ignore_format=True,
    )
    
    woe = WoEEncoder(
        variables= cat_var,
        ignore_format=True,
    )
    
    dte = DecisionTreeEncoder(
        variables=cat_var,
        regression=False,
        scoring='roc_auc',
        cv=3,
        random_state=0,
        ignore_format=True)

    method_all = {'ohe': ohe,
                 'cfe': cfe,
                 'me' : me,
                  'woe': woe,
                  'dte' : dte
                 }

    for cl in cat_var:
        train[cl] = train[cl].astype(str)
        test[cl] = test[cl].astype(str)

    tr_all, ts_all = pd.DataFrame([]), pd.DataFrame([])

    for i in method:
        print(i)
        dsc = method_all.get(i)
        if i in ['dte', 'woe']:
            train_t = rle.fit_transform(train)
            test_t = rle.transform(test)
            dsc.fit(train_t[cat_var], train_t[y])
            train_dsc = dsc.transform(train_t[cat_var])
            test_dsc = dsc.transform(test_t[cat_var])
            
        elif i == 'me':
            dsc.fit(train[cat_var], train[y])
            train_dsc = dsc.transform(train[cat_var])
            test_dsc = dsc.transform(test[cat_var])
        else:
            dsc.fit(train[cat_var])
            train_dsc = dsc.transform(train[cat_var])
            test_dsc = dsc.transform(test[cat_var])
        
        # if i == 'dtd':
        #     train_dsc = train_dsc.rank(method = 'dense')
        #     test_dsc = test_dsc.rank(method = 'dense')
            
        train_dsc.columns=  [f'{i}_{x}' for x in train_dsc.columns]
        test_dsc.columns =  [f'{i}_{x}' for x in test_dsc.columns]


            
        if save_to_file:    
            for j in train_dsc.columns:
                train_dsc[[j]].to_parquet(f'{directory}/train_{j}.parquet')
                test_dsc[[j]].to_parquet(f'{directory}/test_{j}.parquet')
            
            print(f'Done: features saved in {directory} directory!')

        else:
            tr_all, ts_all = pd.concat([tr_all, train_dsc], axis = 1), pd.concat([ts_all, test_dsc], axis = 1)

    return tr_all, ts_all

def create_fi(train, test, num_var, cat_var, metric = ['sum','min', 'max', 'mean', 'median', 'std'], save_to_file = False, directory = '.feataz'):
    
    tr_all, ts_all = pd.DataFrame([]), pd.DataFrame([])
    for i in range(1, len(cat_var)+1, 1) : # 
        for j in num_var:
           for k in (list(combinations(cat_var, i)) ):
                fi = FeatInteraction(list(k), j, metric = metric)
                vars = list(k) +[j]
                fi.fit(train[vars])
                tr1 = fi.transform(train[vars]).drop(vars, axis = 1)
                ts1 = fi.transform(test[vars]).drop(vars, axis = 1)
                if save_to_file:
                    for l in tr1.columns:
                        tr1[[l]].to_parquet(f'{directory}/train_fi_{l}.parquet')
                        ts1[[l]].to_parquet(f'{directory}/test_fi_{l}.parquet')
                        print(f'Done: feature {l} saved in {directory} directory!')
                else:
                    tr_all, ts_all = pd.concat([tr_all, tr1], axis = 1), pd.concat([ts_all, ts1], axis = 1)

    return tr_all, ts_all


def feature_combination_calc(train, test, num_var, method = ['mf', 'rf', 'cf'], save_to_file = False, directory = '.feataz'):
    
    mf = MathFeatures(
        variables=num_var,
        func = ["sum", "prod", "min", "max", "std"],
    )
    

    rf = RelativeFeatures(
        variables=num_var,
        reference=num_var,
        func = ["sub", "div", "mod", "add", "truediv", "floordiv", "mul"],
    )
    
    cf = CyclicalFeatures(variables=num_var, drop_original=False)
    
    method_all = {'mf': mf,
                 'rf': rf,
                 'cf' : cf
                 }
    
    tr_all, ts_all = pd.DataFrame([]), pd.DataFrame([])

    for i in method:
        transformer = method_all.get(i)
        
        if i == 'rf':
            train_inp = train[num_var] + 1
            test_inp = test[num_var] + 1

            k = 1
            while ((train_inp[num_var] == 0).sum() + (test_inp[num_var] == 0).sum()).sum() > 0 :
                print(k)
                train_inp = train[num_var] + k
                test_inp = test[num_var] + k                
            
            train_t = transformer.fit_transform(train_inp[num_var])
            test_t = transformer.fit_transform(test_inp[num_var])
        else:
            train_t = transformer.fit_transform(train[num_var])
            test_t = transformer.fit_transform(test[num_var])

        drp = DropConstantFeatures(tol=0.85)

        drp.fit(train_t)

        feat_drp = list(dict.fromkeys(drp.features_to_drop_ + num_var) )

        train_t.drop(feat_drp, axis = 1, inplace = True)
        test_t.drop(feat_drp, axis = 1, inplace = True)        
        
        if save_to_file:
            for j in train_t.columns:
                train_t[[j]].to_parquet(f'{directory}/train_fc_{j}.parquet')
                test_t[[j]].to_parquet(f'{directory}/test_fc_{j}.parquet')
                
                print(f'Done: features {j} saved in {directory} directory!')
        else:
            tr_all, ts_all = pd.concat([tr_all, train_t], axis = 1), pd.concat([ts_all, test_t], axis = 1)

    return tr_all, ts_all


def feat_sc_woe(train, test, y, variables, save_to_file = False, directory = '.feataz'):
    tr_all, ts_all = pd.DataFrame([]), pd.DataFrame([])
    for l in variables:
        try:
            bins = sc.woebin(train[[l, y]], y=y)
            train_woe = sc.woebin_ply(train[[l, y]], bins)
            test_woe = sc.woebin_ply(test[[l, y]], bins)
            if save_to_file:
                train_woe[[l+'_woe']].to_parquet(f'{directory}/train_woesc_{l}.parquet')
                test_woe[[l+'_woe']].to_parquet(f'{directory}/test_woesc_{l}.parquet')
                print(f'Done: feature {l} saved in {directory} directory!')
            else:
                tr_all, ts_all = pd.concat([tr_all, train_woe], axis = 1), pd.concat([ts_all, test_woe], axis = 1)

        except:
             warnings.warn(f'error occured while creating WOE for {l}')