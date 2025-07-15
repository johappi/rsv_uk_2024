import arviz as az
from cmdstanpy import CmdStanModel
from cmdstanpy import set_cmdstan_path
import numpy as np
import pandas as pd
#import argparse
import sys
import os

import pickle

from pathlib import Path

default_root = Path(__file__).resolve().parents[2]
work_dir = Path(os.getenv("WORK_DIR", default_root))
os.chdir(work_dir)

sys.path.insert(0, './src/') 


def fit_stan(model_variant, imputed = False, imp_i = None, fit_path = None, num_chains = 4, iter_warmup = 1000, iter_sampling = 500, seed = 200, adapt_delta = None, include_enp = False, thin = 5):
    # data
    vars_needed = ['rsv_intent_hyp_pre', 'rsv_intent_hyp_post', 'RCT']
    besd_ord = [f'besd{i}' for i in [1,2,3,14,15,17]]
    besd_ber = [f'besd{i}' for i in [4,5,8,9,10,11,12,13]]
    besd_cat = [f'besd{i}' for i in [6,7]] # [f'besd{i}' for i in [6,7,16,18]] # [f'besd{i}' for i in [6,7]]
    besd_vars = besd_ord + besd_ber + besd_cat
    if model_variant in ['besd_soc']:
        vars_needed = vars_needed + besd_vars
        socio_dems_needed = True
        micro_needed = False
    elif model_variant in  ['rsv_causal_w_soc', 'rsv_child_causal_w_soc']:
        socio_dems_needed = True
        micro_needed = False
    elif model_variant in ['mrp_ord', 'mrp_ber']:
        socio_dems_needed = True
        micro_needed = True 
    elif model_variant in ['ord_wo_poststrat', 'mrp_ber']:
        socio_dems_needed = True
        micro_needed = False 
    elif model_variant in ['rsv_causal_wo_soc']:
        socio_dems_needed = False
        micro_needed = False
    elif model_variant in ['rsv_child_causal_wo_soc']:
        socio_dems_needed = False
        micro_needed = False
    else:
        raise ValueError('non valid model variant')

    if micro_needed or socio_dems_needed:
        itl3_needed = True

    conda_path = ""
    for path in sys.path:
        if "envs/cmdstanpy" in path:
            print(path)
            conda_path = path.split("/lib")[0]
            print(conda_path)
            break

    set_cmdstan_path(conda_path + "/bin/cmdstan")

    if not imputed:
        survey = pd.read_csv('./dat/trs_questionnaire/england_wales/survey_ew.tsv', sep ='\t', index_col = 0)
    else:
        survey = pd.read_csv(f'./dat/trs_questionnaire/england_wales/survey_ew_imp{imp_i}.tsv', sep ='\t', index_col = 0)

    recode_mapping = {}
    recode_mapping_reverse = {}
    recode_mapping['rsv_intent_hyp_pre'] = {'No, definitely not':1, 'Unsure, but leaning towards no':2,'Unsure, but leaning towards yes':3, 'Yes, definitely':4}
    recode_mapping['rsv_intent_hyp_post'] = {'No, definitely not':1, 'Unsure, but leaning towards no':2,'Unsure, but leaning towards yes':3, 'Yes, definitely':4}
    recode_mapping['RCT'] = {'RCT0':1, 'RCT1':2, 'RCT2':3, 'RCT3':4, 'RCT4':5, 'RCT5':6, 'RCT6':7}
    
    
    # if socio_dems_needed:
    
    micro_itl3 = pd.read_csv('./dat/trs_microdata/england_wales/itl3_renamed.tsv', sep ='\t', index_col = 0)

    socio_dems_original = [k for k in micro_itl3.columns if k not in ['n','country']]

    # merge categories with low counts

    reduce_dims_dict = {}
    for c in socio_dems_original:
        _unique = survey[c].unique()
        reduce_dims_dict[c] = dict(zip(_unique, _unique))
        
    reduce_dims_dict['AGE']['55-64'] = '55+'
    reduce_dims_dict['AGE']['65+'] = '55+'

    reduce_dims_dict['SEX']['Other'] = np.nan

    reduce_dims_dict['ETH']['Other ethnic group'] = 'Other'
    reduce_dims_dict['ETH']['White: Roma'] = 'White: Other White'
    reduce_dims_dict['ETH']['White: Gypsy or Irish Traveller'] = 'White: Other White'
    reduce_dims_dict['ETH']['Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups'] = 'Other'

    reduce_dims_dict['YUK']['Arrived 1981 to 1990'] = 'Before 1990'
    reduce_dims_dict['YUK']['Arrived 1971 to 1980'] = 'Before 1990'
    reduce_dims_dict['YUK']['Arrived 1961 to 1970'] = 'Before 1990'
    reduce_dims_dict['YUK']['Arrived 1951 to 1960'] = 'Before 1990'
    reduce_dims_dict['YUK']['Before 1951'] = 'Before 1990'

    reduce_dims_dict['LAN']['Portuguese'] = 'Other'
    reduce_dims_dict['LAN']['Arabic'] = 'Other'
    reduce_dims_dict['LAN']['Spanish'] = 'Other'

    reduce_dims_dict['ENP_v2']['Not at all'] = 'Not well or not at all'
    reduce_dims_dict['ENP_v2']['Not well'] = 'Not well or not at all'

    # combine several ITL3 regions in Scotland with very low counts in survey
    reduce_dims_dict['itl3']['TLM61'] = 'TLM61_64_65_66'
    reduce_dims_dict['itl3']['TLM64'] = 'TLM61_64_65_66'
    reduce_dims_dict['itl3']['TLM65'] = 'TLM61_64_65_66'
    reduce_dims_dict['itl3']['TLM66'] = 'TLM61_64_65_66'

    # reduce_dims_dict['HEA']['Very bad'] = 'Bad or very bad'
    # reduce_dims_dict['HEA']['Bad'] = 'Bad or very bad'



    for c in socio_dems_original:
        survey[c] = survey[c].map(reduce_dims_dict[c])
        micro_itl3[c] = micro_itl3[c].map(reduce_dims_dict[c])

    if socio_dems_needed:
    #     vars_needed = vars_needed + socio_dems_original
        survey = survey[ vars_needed + socio_dems_original ].copy()
    else:
        survey = survey[ vars_needed ].copy()


    if socio_dems_needed:

        micro_itl3 = micro_itl3.groupby( [ k for k in micro_itl3.columns if k not in ['n'] ], as_index = False )['n'].sum()

        # rename columns
        socio_dems_recode = { k : k.strip('_v2') for k in socio_dems_original }

        survey = survey.rename(socio_dems_recode, axis = 1)
        micro_itl3 = micro_itl3.rename(socio_dems_recode, axis = 1)

        socio_dems = list( socio_dems_recode.values() )

        vars_needed = vars_needed + socio_dems

        for var in socio_dems:
            # if var != 'itl3':
            _a = survey[var].unique()
            _b = micro_itl3[var].unique()
            _c = [k for k in _a if k not in _b]
            _d = [k for k in _b if k not in _a]
            if len(_c) > 0:
                print(f'Warning: variable {var}: category {_c} is in survey but not in micro data')    
                if micro_needed:
                    for bad_cat in _c:
                        survey.loc[survey[var] == bad_cat, var] = np.nan
                        print(f'recoded category {bad_cat} of variable {var} to nan')
            if len(_d) > 0:
                print(f'Warning: variable {var}: category {_d} is in micro data but not in survey')

        _df = survey.isna().sum(axis=0)#.min()
        pd.set_option('display.max_rows', None)
        print(f'Number of missing values per variable is now')
        print(_df)
        pd.reset_option('display.max_rows')

    if itl3_needed:
        survey = survey[ ~(survey[ vars_needed ].isna().sum(axis = 1) > 0) ]
    else:
        survey = survey[ ~(survey[ [c for c in vars_needed if c not in ['itl3'] ] ].isna().sum(axis = 1) > 0) ]

    if socio_dems_needed and micro_needed:
        # recode to numerical values
        microdata_recode = micro_itl3.copy()
        # recode_mapping = {}
        for c in micro_itl3.columns:
            if c != 'n':
                microdata_recode[c+'_recode'] = pd.factorize(microdata_recode[c])[0]
                recode_mapping[c] = dict(zip( microdata_recode[c], microdata_recode[c+'_recode'] ))
                microdata_recode = microdata_recode.drop(columns = [c])

        survey_recode = survey[survey['itl3'].isin(micro_itl3['itl3'].unique())].copy()
        for c in micro_itl3.columns:
            if c not in ['n', 'country']:
                # try:
                survey_recode[c + '_recode'] = survey_recode[c].map(recode_mapping[c]).astype(np.int64)
                survey_recode = survey_recode.drop(columns = [c])
                # except: 
                #     print(c)
    elif socio_dems_needed and not micro_needed:
        survey_recode = survey.copy()
        print(f'list of sociodems is now: {socio_dems}')
        for c in socio_dems:
            if c not in ['n', 'country']:
                # try:
                survey_recode[c+'_recode'] = pd.factorize(survey_recode[c])[0].astype(np.int64)
                recode_mapping[c] = dict(zip( survey_recode[c], survey_recode[c+'_recode'] ))
                survey_recode = survey_recode.drop(columns = [c])
                # except: 
                #     print(c)

    else:
        survey_recode = survey.copy()

    if set(besd_vars).issubset(set(vars_needed)):
        recode_mapping_update = {
        'besd1': { 'Not at all important' : 0, 'A little important' : 1, 'Moderately important' : 2, 'Very important' : 3},
        'besd2': { 'Not at all safe' : 0, 'A little safe' : 1, 'Moderately safe' : 2, 'Very safe' : 3},
        'besd3': { 'Not at all' : 0, 'A little' : 1, 'Moderately' : 2, 'Very much' : 3},
        'besd14': { 'Not at all easy' : 0, 'A little easy' : 1, 'Moderately easy' : 2, 'Very easy': 3},
        'besd15': { 'Not at all easy' : 0, 'A little easy' : 1, 'Moderately easy' : 2, 'Very easy' : 3},
        'besd17': { 'Not at all satisfied' : 0, 'A little satisfied' : 1, 'Moderately satisfied' : 2, 'Very satisfied' : 3}
        }

        recode_mapping.update(recode_mapping_update)

        for c in besd_ord:
            survey_recode[c + '_recode'] = survey_recode[c].map(recode_mapping[c]).astype(np.int64)
            survey_recode = survey_recode.drop(columns = [c])

        for c in besd_ber:
            recode_mapping[c] = {'No' : 0, 'Yes' : 1}
            survey_recode[c + '_recode'] = survey_recode[c].map(recode_mapping[c]).astype(np.int64)
            survey_recode = survey_recode.drop(columns = [c])

        for c in besd_cat:
            if c in ['besd6','besd7']:
                recode_mapping[c] = {'Not applicable to me' : 0, 'No' : 1, 'Yes' : 2}
                survey_recode[c + '_recode'] = survey_recode[c].map(recode_mapping[c]).astype(np.int64)
                survey_recode = survey_recode.drop(columns = [c])
            else:
                survey_recode[c+'_recode'] = pd.factorize(survey_recode[c])[0]
                recode_mapping[c] = dict(zip( survey_recode[c], survey_recode[c+'_recode'] ))
                survey_recode = survey_recode.drop(columns = [c])



    survey_recode['rsv_intent_hyp_pre_ber'] = survey_recode['rsv_intent_hyp_pre'].map({'No, definitely not':0, 'Unsure, but leaning towards no':0,'Unsure, but leaning towards yes':1, 'Yes, definitely':1})
    survey_recode['rsv_intent_hyp_pre'] = survey_recode['rsv_intent_hyp_pre'].map( recode_mapping['rsv_intent_hyp_pre'] )
    survey_recode['rsv_intent_htp_post_ber'] = survey_recode['rsv_intent_hyp_post'].map({'No, definitely not':0, 'Unsure, but leaning towards no':0,'Unsure, but leaning towards yes':1, 'Yes, definitely':1})
    survey_recode['rsv_intent_hyp_post'] = survey_recode['rsv_intent_hyp_post'].map( recode_mapping['rsv_intent_hyp_post'] )

    survey_recode['RCT'] = survey_recode['RCT'].map( recode_mapping['RCT'] )
    # data dictionary for stan 
    print( recode_mapping['LAN'] )
    for k,v in recode_mapping.items():
        _dict = {j:i for i,j in v.items()}
        recode_mapping_reverse[k] = _dict.copy()


    data = {'N' : survey_recode.shape[0],
            'N_dep' : len(survey_recode['rsv_intent_hyp_pre'].unique()),
            'y' : survey_recode['rsv_intent_hyp_pre_ber'],
            'y_ordinal' : survey_recode['rsv_intent_hyp_pre'],
            'y_pre' : survey_recode['rsv_intent_hyp_pre'],
            'y_post' : survey_recode['rsv_intent_hyp_post'],
            'N_treat' : 7,
            'treat' : survey_recode['RCT'],
            }

    if socio_dems_needed and micro_needed:
        data_add_socio = {
                'N' : survey_recode.shape[0],
                'N_pred' : microdata_recode.shape[0],
                'N_dep' : len(survey_recode['rsv_intent_hyp_pre'].unique()),
                #
                'N_reg': micro_itl3['itl3'].nunique(),
                'N_age': micro_itl3['AGE'].nunique(),
                'N_sex': micro_itl3['SEX'].nunique(),
                'N_edu': micro_itl3['EDU'].nunique(),
                'N_rel': micro_itl3['REL'].nunique(),
                'N_eth': micro_itl3['ETH'].nunique(),
                'N_lan': micro_itl3['LAN'].nunique(),
                'N_hhlan': micro_itl3['HHLAN'].nunique(),
                'N_mig': micro_itl3['MIG'].nunique(),
                'N_buk': micro_itl3['BUK'].nunique(),
                'N_yuk': micro_itl3['YUK'].nunique(),
                'N_enp': micro_itl3['ENP'].nunique(),
                'N_dis': micro_itl3['DIS'].nunique(),
                'N_hea': micro_itl3['HEA'].nunique(),
                'N_emp': micro_itl3['EMP'].nunique(),
                'N_sch': micro_itl3['SCH'].nunique(),
                'y' : survey_recode['rsv_intent_hyp_pre_ber'],
                'y_ordinal' : survey_recode['rsv_intent_hyp_pre'],
                'y_pre' : survey_recode['rsv_intent_hyp_pre'],
                'y_post' : survey_recode['rsv_intent_hyp_post'],
                'N_treat' : 7,
                'treat' : survey_recode['RCT'],
                #
                'age' : survey_recode['AGE_recode'] + 1,
                'reg' : survey_recode['itl3_recode'] + 1,
                'eth' : survey_recode['ETH_recode'] + 1,
                'lan' : survey_recode['LAN_recode'] + 1,
                'emp' : survey_recode['EMP_recode'] + 1,
                'edu' : survey_recode['EDU_recode'] + 1,
                'sex' : survey_recode['SEX_recode'] + 1,
                'rel' : survey_recode['REL_recode'] + 1,
                'hhlan' : survey_recode['HHLAN_recode'] + 1,
                'mig' : survey_recode['MIG_recode'] + 1,
                'yuk' : survey_recode['YUK_recode'] + 1,
                'buk' : survey_recode['BUK_recode'] + 1,
                'enp' : survey_recode['ENP_recode'] + 1,
                'dis' : survey_recode['DIS_recode'] + 1,
                'hea' : survey_recode['HEA_recode'] + 1,
                'sch' : survey_recode['SCH_recode'] + 1,
                'age_pred' : microdata_recode['AGE_recode'] + 1,
                'reg_pred' : microdata_recode['itl3_recode'] + 1,
                'eth_pred' : microdata_recode['ETH_recode'] + 1,
                'lan_pred' : microdata_recode['LAN_recode'] + 1,
                'emp_pred' : microdata_recode['EMP_recode'] + 1,
                'edu_pred' : microdata_recode['EDU_recode'] + 1,
                'sex_pred' : microdata_recode['SEX_recode'] + 1,
                'rel_pred' : microdata_recode['REL_recode'] + 1,
                'hhlan_pred' : microdata_recode['HHLAN_recode'] + 1,
                'mig_pred' : microdata_recode['MIG_recode'] + 1,
                'yuk_pred' : microdata_recode['YUK_recode'] + 1,
                'buk_pred' : microdata_recode['BUK_recode'] + 1,
                'enp_pred' : microdata_recode['ENP_recode'] + 1,
                'dis_pred' : microdata_recode['DIS_recode'] + 1,
                'hea_pred' : microdata_recode['HEA_recode'] + 1,
                'sch_pred' : microdata_recode['SCH_recode'] + 1,
            }
        
        data.update(data_add_socio)

    elif socio_dems_needed and not micro_needed:
        data_add_socio = {
                'N' : survey_recode.shape[0],
                # 'N_pred' : microdata_recode.shape[0],
                'N_dep' : len(survey_recode['rsv_intent_hyp_pre'].unique()),
                #
                'N_reg': survey['itl3'].nunique(),
                'N_age': survey['AGE'].nunique(),
                'N_sex': survey['SEX'].nunique(),
                'N_edu': survey['EDU'].nunique(),
                'N_rel': survey['REL'].nunique(),
                'N_eth': survey['ETH'].nunique(),
                'N_lan': survey['LAN'].nunique(),
                'N_hhlan': survey['HHLAN'].nunique(),
                'N_mig': survey['MIG'].nunique(),
                'N_buk': survey['BUK'].nunique(),
                'N_yuk': survey['YUK'].nunique(),
                'N_enp': survey['ENP'].nunique(),
                'N_dis': survey['DIS'].nunique(),
                'N_hea': survey['HEA'].nunique(),
                'N_emp': survey['EMP'].nunique(),
                'N_sch': survey['SCH'].nunique(),
                'y' : survey_recode['rsv_intent_hyp_pre_ber'],
                'y_ordinal' : survey_recode['rsv_intent_hyp_pre'],
                'y_pre' : survey_recode['rsv_intent_hyp_pre'],
                'y_post' : survey_recode['rsv_intent_hyp_post'],
                'N_treat' : 7,
                'treat' : survey_recode['RCT'],
                #
                'age' : survey_recode['AGE_recode'] + 1,
                'reg' : survey_recode['itl3_recode'] + 1,
                'eth' : survey_recode['ETH_recode'] + 1,
                'lan' : survey_recode['LAN_recode'] + 1,
                'emp' : survey_recode['EMP_recode'] + 1,
                'edu' : survey_recode['EDU_recode'] + 1,
                'sex' : survey_recode['SEX_recode'] + 1,
                'rel' : survey_recode['REL_recode'] + 1,
                'hhlan' : survey_recode['HHLAN_recode'] + 1,
                'mig' : survey_recode['MIG_recode'] + 1,
                'yuk' : survey_recode['YUK_recode'] + 1,
                'buk' : survey_recode['BUK_recode'] + 1,
                'enp' : survey_recode['ENP_recode'] + 1,
                'dis' : survey_recode['DIS_recode'] + 1,
                'hea' : survey_recode['HEA_recode'] + 1,
                'sch' : survey_recode['SCH_recode'] + 1,
                # 'age_pred' : microdata_recode['AGE_recode'] + 1,
                # 'reg_pred' : microdata_recode['itl3_recode'] + 1,
                # 'eth_pred' : microdata_recode['ETH_recode'] + 1,
                # 'lan_pred' : microdata_recode['LAN_recode'] + 1,
                # 'emp_pred' : microdata_recode['EMP_recode'] + 1,
                # 'edu_pred' : microdata_recode['EDU_recode'] + 1,
                # 'sex_pred' : microdata_recode['SEX_recode'] + 1,
                # 'rel_pred' : microdata_recode['REL_recode'] + 1,
                # 'hhlan_pred' : microdata_recode['HHLAN_recode'] + 1,
                # 'mig_pred' : microdata_recode['MIG_recode'] + 1,
                # 'yuk_pred' : microdata_recode['YUK_recode'] + 1,
                # 'buk_pred' : microdata_recode['BUK_recode'] + 1,
                # 'enp_pred' : microdata_recode['ENP_recode'] + 1,
                # 'dis_pred' : microdata_recode['DIS_recode'] + 1,
                # 'hea_pred' : microdata_recode['HEA_recode'] + 1,
                # 'sch_pred' : microdata_recode['SCH_recode'] + 1,
            }
        
        data.update(data_add_socio)

    if set(besd_vars).issubset(set(vars_needed)):
        data_add_besd = {
            'N_besd_ord' : len(besd_ord),
            'num_besd_ord_cat' : 4,
            'N_besd_ber' : len(besd_ber),
            'N_besd6' : len(survey_recode['besd6_recode'].unique()),
            'N_besd7': len(survey_recode['besd7_recode'].unique()),
            # 'N_besd16': len(survey_recode['besd16'].unique()),
            # 'N_besd18': len(survey_recode['besd18'].unique()),
            
            'besd6': survey_recode['besd6_recode'] + 1,
            'besd7' : survey_recode['besd7_recode'] + 1,
            # 'besd16' : survey_recode['besd16_recode'] + 1,
            # 'besd18' : survey_recode['besd18_recode'] + 1,
            
            'besd_ord' : ( survey_recode[ [f'{c}_recode' for c in besd_ord] ] + 1 ).T,
            'besd_ber' : ( survey_recode[ [f'{c}_recode' for c in besd_ber] ] + 1 ).T,
        }
        data.update(data_add_besd)

    if model_variant == 'mrp_ord':
        model = CmdStanModel(stan_file="./stan_codes/mrp_ord_ew.stan",cpp_options={'STAN_THREADS': True})
    elif model_variant == 'ord_wo_poststrat':
        model = CmdStanModel(stan_file="./stan_codes/ord_ew_wo_micro.stan",cpp_options={'STAN_THREADS': True})
    elif model_variant == 'mrp_ber':
        model = CmdStanModel(stan_file="./stan_codes/mrp_ber_ew.stan",cpp_options={'STAN_THREADS': True})
    elif model_variant == 'rsv_causal_wo_soc':
        model = CmdStanModel(stan_file="./stan_codes/ord_causal_wo_socio.stan",cpp_options={'STAN_THREADS': True})
    elif model_variant == 'rsv_causal_w_soc':
        model = CmdStanModel(stan_file="./stan_codes/ord_causal_soc_hier_ew.stan",cpp_options={'STAN_THREADS': True})
    elif model_variant == 'besd_soc':
        model = CmdStanModel(stan_file="./stan_codes/bes_soc_ord_ew.stan",cpp_options={'STAN_THREADS': True})
    fit = model.sample(data = data, threads_per_chain=2, show_console = True, chains = num_chains, iter_warmup = iter_warmup,
                        iter_sampling = iter_sampling * thin, seed = seed, adapt_delta = adapt_delta, thin = thin )


    idata = az.from_cmdstanpy(posterior=fit, save_warmup=False) 

    if fit_path is None:
        if model_variant == 'mrp_ord':
            fit_path = './idata/ew/rsv_intent_ord_soc.nc'
        elif model_variant == 'ord_wo_poststrat':
            fit_path = './idata/ew/rsv_intent_ord_soc_wo_micro.nc'
        elif model_variant == 'mrp_ber':
            fit_path = './idata/ew/rsv_intent_ber_soc.nc'
        elif model_variant == 'rsv_causal_wo_soc':
            fit_path = './idata/ew/rsv_intent_causal_wo_soc.nc'
        elif model_variant == 'rsv_causal_w_soc':
            fit_path = './idata/ew/rsv_intent_causal_w_soc.nc'
        elif model_variant == 'besd_soc':
            fit_path = './idata/ew/rsv_intent_besd_soc.nc'
    if imputed:
        _fit_path_imp = fit_path[:-3]
        fit_path_imp = f'{_fit_path_imp}_imp{imp_i}.nc'
        fit_path_imp_comb = f'{_fit_path_imp}_imp_combined.nc' 
        idata.to_netcdf( fit_path_imp )
        if imp_i == 1:
            idata.to_netcdf( fit_path_imp_comb )
        else:
            idata_combined = az.from_netcdf( fit_path_imp_comb )
            idata_combined = az.concat(idata_combined, idata.copy(), dim = 'chain')
            idata_combined.to_netcdf( fit_path_imp_comb )

    else:
        idata.to_netcdf(fit_path)

    if not imputed:
        survey.to_csv(f'./dat/dat_for_inference/ew/survey_{model_variant}.tsv', sep = '\t')
        survey_recode.to_csv(f'./dat/dat_for_inference/ew/survey_recode_{model_variant}.tsv', sep = '\t')
    if model_variant in ['mrp_ord', 'mrp_ber']:
        micro_itl3.to_csv(f'./dat/dat_for_inference/ew/micro_{model_variant}.tsv', sep = '\t')
        microdata_recode.to_csv(f'./dat/dat_for_inference/ew/micro_recode_{model_variant}.tsv', sep = '\t')
    with open(f'./dat/dat_for_inference/ew/data_dict_{model_variant}.p', 'wb') as f:
        pickle.dump(data, f)
    with open(f'./dat/dat_for_inference/ew/var_mappings_{model_variant}.p', 'wb') as f:
        pickle.dump( {'recode_mapping_reverse' : recode_mapping_reverse, 'reduce_dims_dict' : reduce_dims_dict,}, f)

