import sys
import os
import argparse
import pickle
import warnings

import arviz as az
import numpy as np
import pandas as pd

from scipy.special import expit
from pathlib import Path

default_root = Path(__file__).resolve().parents[1]
work_dir = Path(os.getenv('WORK_DIR', default_root))

os.chdir(work_dir)
cwd = os.getcwd()
print(cwd)

sys.path.insert(0, './src/')



def ordered_logistic_prob_vectorized_v3(beta, c):
    """
    Compute the probabilities P(Y=y) for all categories of an ordinal outcome Y with k categories,
    in a vectorized manner when beta has shape (m_1, m_2) and c has shape (m_1, N_dep - 1).
    
    Parameters:
    beta (np.ndarray): Array of shape (m_1, m_2) containing the linear predictors.
    c (np.ndarray): Array of shape (m_1, N_dep - 1) containing the cutpoints (same cutpoints for each m_2).
    
    Returns:
    np.ndarray: Array of shape (m_1, m_2, N_dep) containing the probabilities P(Y=y) for each category.
    """
    m_1, m_2 = beta.shape
    N_dep_minus_1 = c.shape[1]  # Number of cutpoints (N_dep - 1)
    
    # Compute cumulative probabilities using broadcasting
    cum_probs = expit(c[:, np.newaxis, :] - beta[:, :, np.newaxis])  # Shape (m_1, m_2, N_dep - 1)
    
    # Initialize an array for probabilities of shape (m_1, m_2, N_dep)
    probs = np.zeros((m_1, m_2, N_dep_minus_1 + 1))
    
    # First category probabilities
    probs[:, :, 0] = cum_probs[:, :, 0]
    
    # Middle category probabilities
    probs[:, :, 1:-1] = cum_probs[:, :, 1:] - cum_probs[:, :, :-1]
    
    # Last category probabilities
    probs[:, :, -1] = 1 - cum_probs[:, :, -1]
    
    return probs

def ordered_logistic_prob_for_category(beta, c, p):
    """
    Compute the probability P(Y=p) for a specific category p of an ordinal outcome Y,
    in a vectorized manner when beta has shape (m_1, m_2) and c has shape (m_1, N_dep - 1).
    
    Parameters:
    beta (np.ndarray): Array of shape (m_1, m_2) containing the linear predictors.
    c (np.ndarray): Array of shape (m_1, N_dep - 1) containing the cutpoints (same cutpoints for each m_2).
    p (int): The specific category for which to compute the probability (1 ≤ p ≤ N_dep).
    
    Returns:
    np.ndarray: Array of shape (m_1, m_2) containing the probability P(Y=p).
    """
    m_1, m_2 = beta.shape
    N_dep_minus_1 = c.shape[1]  # Number of cutpoints (N_dep - 1)
    
    # Compute cumulative probabilities using broadcasting
    cum_probs = expit(c[:, np.newaxis, :] - beta[:, :, np.newaxis])  # Shape (m_1, m_2, N_dep - 1)
    
    if p == 1:
        # First category probability P(Y=1)
        return cum_probs[:, :, 0]
    elif p == N_dep_minus_1 + 1:
        # Last category probability P(Y=N_dep)
        return 1 - cum_probs[:, :, -1]
    else:
        # Middle category probability P(Y=p)
        return cum_probs[:, :, p - 1] - cum_probs[:, :, p - 2]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")

    # Define named arguments
    parser.add_argument('--var_post_strat', type = str,  default = 'itl3', help = 'Variable which defines the units for post stratification')
    parser.add_argument('--prob_name_pre', type = str,  default = 'p', help = 'Probability of pre intent for which post stratification is done, must be one of p1,p2,p2,p4')
    # parser.add_argument('--prob_name_post', type = str,  default = 'p', help = 'Probability of post intent for which post stratification is done, must be one of p1,p2,p2,p4')
    parser.add_argument('--post_strat_path', type = str, default = './post_strat/uk/rsv_causal_varying_c', help = 'path to directory for storing post stratified results')
    parser.add_argument('--fit_path', type = str, default = './idata/uk/rsv_intent_causal_w_soc.nc', help = 'path to directory where inference results are stored')
    parser.add_argument('--dat_for_inference_path', type = str, default = './dat/dat_for_inference/uk', help = 'path to directory where data used in inference is stored')

    # parser.add_argument('--pre_or_post', type = str, default = 'pre', help = 'whether pre treatment or treamtent group')
    parser.add_argument('--treat_group', type = int, default = 0, help = 'which treatment group to consider')

    args = parser.parse_args()

    var_post_strat = args.var_post_strat
    prob_name_pre = args.prob_name_pre
    # prob_name_post = args.prob_name_post
    post_strat_path = args.post_strat_path
    fit_path = args.fit_path
    dat_for_inference_path = args.dat_for_inference_path

    # pre_or_post = args.pre_or_post
    treat_group = args.treat_group

    if prob_name_pre == 'p1':
        condition = 1
    elif prob_name_pre == 'p2':
        condition = 2
    elif prob_name_pre == 'p3':
        condition = 3
    elif prob_name_pre == 'p4':
        condition = 4

    os.path.join( dat_for_inference_path, )

    idata = az.from_netcdf(fit_path)

    with open( os.path.join( dat_for_inference_path, 'data_dict_rsv_causal_w_soc.p' ), 'rb') as f:
        data_dict = pickle.load(f)
    micro = pd.read_csv( os.path.join( dat_for_inference_path, 'micro_rsv_causal_w_soc.tsv' ), sep = '\t', index_col = 0)

    #######################################################
    # post stratification

    # parameters

    num_samples = idata.posterior['c'].values.shape[0] * idata.posterior['c'].values.shape[1]
    N_treat = data_dict['N_treat']
    N_pred = data_dict['N_pred']
    N_dep = data_dict['N_dep']
    N_age = data_dict['N_age']
    N_edu = data_dict['N_edu']
    N_eth = data_dict['N_eth']
    N_lan = data_dict['N_lan']
    N_rel = data_dict['N_rel']
    N_emp = data_dict['N_emp']
    N_reg = data_dict['N_reg']
    # N_enp = data_dict['N_enp']
    N_dis = data_dict['N_dis']
    N_hea = data_dict['N_hea']

    reg_pred = data_dict['reg_pred'] - 1

    age_pred = data_dict['age_pred'] - 1
    sex_pred = data_dict['sex_pred'] - 1
    edu_pred = data_dict['edu_pred'] - 1
    eth_pred = data_dict['eth_pred'] - 1
    rel_pred = data_dict['rel_pred'] - 1
    lan_pred = data_dict['lan_pred'] - 1
    emp_pred = data_dict['emp_pred'] - 1
    buk_pred = data_dict['buk_pred'] - 1
    mig_pred = data_dict['mig_pred'] - 1
    # enp_pred = data_dict['enp_pred'] - 1
    dis_pred = data_dict['dis_pred'] - 1
    hea_pred = data_dict['hea_pred'] - 1


    # pre treatment

    N_sex, N_buk, N_mig = 2, 2, 2
    beta_sex_pre = idata.posterior['beta_sex_pre'].values.reshape(num_samples) #, N_sex)
    beta_buk_pre = idata.posterior['beta_buk_pre'].values.reshape(num_samples) # , N_buk)
    beta_mig_pre = idata.posterior['beta_mig_pre'].values.reshape(num_samples) #, N_mig)
    # real beta_hhlan_pre;

    c_pre = idata.posterior['c_pre'].values.reshape(num_samples, N_dep - 1)


    beta_age_pre = idata.posterior['beta_age_pre'].values.reshape(num_samples, N_age)
    beta_edu_pre = idata.posterior['beta_edu_pre'].values.reshape(num_samples, N_edu)
    beta_eth_pre = idata.posterior['beta_eth_pre'].values.reshape(num_samples, N_eth)
    beta_lan_pre = idata.posterior['beta_lan_pre'].values.reshape(num_samples, N_lan)
    beta_rel_pre = idata.posterior['beta_rel_pre'].values.reshape(num_samples, N_rel)
    beta_emp_pre = idata.posterior['beta_emp_pre'].values.reshape(num_samples, N_emp)
    beta_reg_pre = idata.posterior['beta_reg_pre'].values.reshape(num_samples, N_reg)

    # beta_enp_pre = idata.posterior['beta_enp_pre'].values.reshape(num_samples, N_enp)
    beta_dis_pre = idata.posterior['beta_dis_pre'].values.reshape(num_samples, N_dis)
    beta_hea_pre = idata.posterior['beta_hea_pre'].values.reshape(num_samples, N_hea)
    # vector[N_yuk] beta_yuk_pre = sigma_yuk_pre * beta_yuk_raw2_pre;

    # treatment effects

    beta_sex = idata.posterior['beta_sex'].values.reshape(num_samples, N_treat) #, N_sex)
    beta_buk = idata.posterior['beta_buk'].values.reshape(num_samples, N_treat) #, N_buk)
    beta_mig = idata.posterior['beta_mig'].values.reshape(num_samples, N_treat) #, N_mig)

    beta_age = idata.posterior['beta_age'].values.reshape(num_samples, N_treat, N_age)
    beta_edu = idata.posterior['beta_edu'].values.reshape(num_samples, N_treat, N_edu)
    beta_eth = idata.posterior['beta_eth'].values.reshape(num_samples, N_treat, N_eth)
    beta_lan = idata.posterior['beta_lan'].values.reshape(num_samples, N_treat, N_lan)
    beta_rel = idata.posterior['beta_rel'].values.reshape(num_samples, N_treat, N_rel)
    beta_emp = idata.posterior['beta_emp'].values.reshape(num_samples, N_treat, N_emp)
    # beta_reg = idata.posterior['beta_reg'].values.reshape(num_samples, N_treat, N_reg)
    # beta_enp = idata.posterior['beta_enp'].values.reshape(num_samples, N_treat, N_enp)
    beta_dis = idata.posterior['beta_dis'].values.reshape(num_samples, N_treat, N_dis)
    beta_hea = idata.posterior['beta_hea'].values.reshape(num_samples, N_treat, N_hea)

    c  = idata.posterior['c'].values.reshape(num_samples, N_treat, N_dep, N_dep - 1)
    # // array[N_treat] vector[N_yuk] beta_yuk;

    print( 'I am here 1' )

    # arrays for post stratification
    lin_pred_pre = ( beta_age_pre[:, age_pred] + np.stack([beta_sex_pre, -beta_sex_pre], axis = 1)[:, sex_pred] + beta_edu_pre[:, edu_pred] + beta_eth_pre[:, eth_pred]
                    + beta_lan_pre[:, lan_pred] + beta_rel_pre[:, rel_pred] + beta_emp_pre[:, emp_pred] + beta_reg_pre[:, reg_pred]
                    + beta_dis_pre[:, dis_pred] + beta_hea_pre[:, hea_pred] # + beta_enp_pre[:, enp_pred]
                    + np.stack([beta_buk_pre, -beta_buk_pre], axis = 1)[:, buk_pred] + np.stack([beta_mig_pre, -beta_mig_pre], axis = 1)[:, mig_pred] )

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    print( 'I am here 2' )

    p1p1_labels = [f'p1pre_p1post_{s}' for s in range(num_samples)]
    p1p2_labels = [f'p1pre_p2post_{s}' for s in range(num_samples)]
    p1p3_labels = [f'p1pre_p3post_{s}' for s in range(num_samples)]
    p1p4_labels = [f'p1pre_p4post_{s}' for s in range(num_samples)]

    p2p1_labels = [f'p2pre_p1post_{s}' for s in range(num_samples)]
    p2p2_labels = [f'p2pre_p2post_{s}' for s in range(num_samples)]
    p2p3_labels = [f'p2pre_p3post_{s}' for s in range(num_samples)]
    p2p4_labels = [f'p2pre_p4post_{s}' for s in range(num_samples)]

    p3p1_labels = [f'p3pre_p1post_{s}' for s in range(num_samples)]
    p3p2_labels = [f'p3pre_p2post_{s}' for s in range(num_samples)]
    p3p3_labels = [f'p3pre_p3post_{s}' for s in range(num_samples)]
    p3p4_labels = [f'p3pre_p4post_{s}' for s in range(num_samples)]

    p4p1_labels = [f'p4pre_p1post_{s}' for s in range(num_samples)]
    p4p2_labels = [f'p4pre_p2post_{s}' for s in range(num_samples)]
    p4p3_labels = [f'p4pre_p3post_{s}' for s in range(num_samples)]
    p4p4_labels = [f'p4pre_p4post_{s}' for s in range(num_samples)]

    print( 'I am here 3' )

    if var_post_strat == 'no_post_strat_var':
        data_w_weights = micro[['n']].copy() 
    else:
        data_w_weights = micro[[var_post_strat,'n']].copy()

    y_prob_pre = np.transpose( ordered_logistic_prob_vectorized_v3( lin_pred_pre, c_pre ), (1, 0, 2) )[:, :, condition-1]

    lin_pred_post = (lin_pred_pre 
            + beta_age[:, treat_group, age_pred] + np.stack([beta_sex[:,treat_group], -beta_sex[:,treat_group]], axis = 1)[:,  sex_pred] + beta_edu[:, treat_group,  edu_pred] + beta_eth[:, treat_group,  eth_pred]
            + beta_lan[:, treat_group,  lan_pred] + beta_rel[:, treat_group,  rel_pred] + beta_emp[:, treat_group,  emp_pred] # + beta_reg[:, t,  reg_pred]
            + beta_dis[:, treat_group,  dis_pred] + beta_hea[:, treat_group,  hea_pred] # + beta_enp[:, treat_group,  enp_pred]
            + np.stack([beta_buk[:,treat_group], -beta_buk[:,treat_group]], axis = 1)[:,  buk_pred] + np.stack([beta_mig[:,treat_group], -beta_mig[:,treat_group]], axis = 1)[:,  mig_pred] )
    
    print( 'I am here 4' )

    del lin_pred_pre

    print( 'I am here 5' )

    y_prob_post = np.transpose( ordered_logistic_prob_vectorized_v3( lin_pred_post, c[:,treat_group, condition-1, :] ), (1, 0, 2) )
    # y_prob_post = np.transpose(y_prob_post, (1, 0, 2) )

    del lin_pred_post 

    print( 'I am here 6' )

    if prob_name_pre == 'p1':
        data_w_weights[p1p1_labels] = y_prob_pre * y_prob_post[:,:,0]
        data_w_weights[p1p2_labels] = y_prob_pre * y_prob_post[:,:,1]
        data_w_weights[p1p3_labels] = y_prob_pre * y_prob_post[:,:,2]
        data_w_weights[p1p4_labels] = y_prob_pre * y_prob_post[:,:,3]
        _labels = p1p1_labels + p1p2_labels + p1p3_labels + p1p4_labels
    elif prob_name_pre == 'p2':
        data_w_weights[p2p1_labels] = y_prob_pre * y_prob_post[:,:,0]
        data_w_weights[p2p2_labels] = y_prob_pre * y_prob_post[:,:,1]
        data_w_weights[p2p3_labels] = y_prob_pre * y_prob_post[:,:,2]
        data_w_weights[p2p4_labels] = y_prob_pre * y_prob_post[:,:,3]
        _labels = p2p1_labels + p2p2_labels + p2p3_labels + p2p4_labels
    elif prob_name_pre == 'p3':
        data_w_weights[p3p1_labels] = y_prob_pre * y_prob_post[:,:,0]
        data_w_weights[p3p2_labels] = y_prob_pre * y_prob_post[:,:,1]
        data_w_weights[p3p3_labels] = y_prob_pre * y_prob_post[:,:,2]
        data_w_weights[p3p4_labels] = y_prob_pre * y_prob_post[:,:,3]
        _labels = p3p1_labels + p3p2_labels + p3p3_labels + p3p4_labels
    elif prob_name_pre == 'p4':
        data_w_weights[p4p1_labels] = y_prob_pre * y_prob_post[:,:,0]
        data_w_weights[p4p2_labels] = y_prob_pre * y_prob_post[:,:,1]
        data_w_weights[p4p3_labels] = y_prob_pre * y_prob_post[:,:,2]
        data_w_weights[p4p4_labels] = y_prob_pre * y_prob_post[:,:,3]
        _labels = p4p1_labels + p4p2_labels + p4p3_labels + p4p4_labels

    else:
        raise ValueError('non valid prob_name_pre')

    del y_prob_pre 
    del y_prob_post
    del idata

    print( 'I am here 7' )

    if var_post_strat == 'no_post_strat_var':
        weights = data_w_weights['n'].values
        posterior_samples = data_w_weights[ _labels ]
        post_stratified_samples = ( posterior_samples * weights[:,None] ).sum(axis = 0) / weights.sum()
        df_post_strat = pd.Series( post_stratified_samples )
    else:
        results = []
        grouped = data_w_weights.groupby(var_post_strat)
        for _var, group in grouped:
            weights = group['n'].values
            posterior_samples = group[ _labels ]

            post_stratified_samples = ( posterior_samples * weights[:,None] ).sum(axis = 0) / weights.sum()

            _new_row = { var_post_strat : _var }
            _new_row.update( dict( zip( _labels, post_stratified_samples ) ) )
            results.append( _new_row.copy() )
        df_post_strat = pd.DataFrame(results )

    file_path = f'{var_post_strat}_rsv_pre_{prob_name_pre}_post{treat_group}_p1p2p3p4.tsv'
    file_path = os.path.join(post_strat_path, file_path )

    if var_post_strat != 'no_post_strat_var':
        df_post_strat = df_post_strat.set_index(var_post_strat)

    df_post_strat.to_csv(  file_path, sep = '\t')
        

