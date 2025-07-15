import sys
import os
import argparse
import pickle
import warnings

import arviz as az
import numpy as np
import pandas as pd

import geopandas as gpd
from pathlib import Path

default_root = Path(__file__).resolve().parents[1]
work_dir = Path(os.getenv('WORK_DIR', default_root))

os.chdir(work_dir)
cwd = os.getcwd()
print(cwd)

sys.path.insert(0, './src/')

def post_stratify(data_w_weights, percentiles = [2.5, 50, 97.5], num_samples=2000, var_name='p', var_post_strat='itl3'):
    """
    Perform post-stratification for each region/variable and compute percentiles over post-stratified samples.
    
    Args:
    - data_w_weights: Pandas DataFrame containing columns 'var_post_strat', 'n' (weights), and samples from posterior distribution.
    - percentiles: List of percentiles to compute (values between 0 and 100).
    - num_samples: (int) Number of posterior samples, default is 2000.
    - var_name: (str) Base name of the posterior sample variable (e.g., 'p' in 'p_0', 'p_1', ...), default is 'p'.
    - var_post_strat: (str) Column name in `data_w_weights` that contains the variable/region identifier (e.g., 'itl3'), default is 'itl3'.
    
    Returns:
    - DataFrame with post-stratified percentiles and mean for each region/variable.
    """
    def weighted_percentile(data, weights, percentiles):
        """
        Compute weighted percentiles for the given data and weights.
        
        Args:
        - data: 1D array-like of data values.
        - weights: 1D array-like of weights.
        - percentiles: List of percentiles to compute (values between 0 and 100).
        
        Returns:
        - List of computed percentiles corresponding to the input percentiles.
        """
        sorted_idx = np.argsort(data)
        sorted_data = np.array(data)[sorted_idx]
        sorted_weights = np.array(weights)[sorted_idx]
        
        # Compute cumulative sum of weights
        cum_weights = np.cumsum(sorted_weights)
        cum_weights /= cum_weights[-1]  # Normalize to make it go from 0 to 1 (CDF)
        
        # Use numpy interpolation to get the percentiles
        return np.interp(np.array(percentiles) / 100, cum_weights, sorted_data)
    # Group the data by the post-stratification variable (e.g., region)
    grouped = data_w_weights.groupby(var_post_strat)

    # Prepare an empty list to store the results
    results = []

    # Loop over each group (e.g., region or variable)
    for region, group in grouped:
        # Extract weights and posterior samples for the current group
        weights = group['n'].values
        posterior_samples = group[[f'{var_name}_{i}' for i in range(num_samples)]].values
        
        # Post-stratify each sample using the weights
        post_stratified_samples = np.average(posterior_samples, axis=0, weights=weights)
        
        # Compute percentiles over the post-stratified samples
        # pcts = weighted_percentile(post_stratified_samples, np.ones_like(post_stratified_samples), percentiles)
        pcts = np.percentile(post_stratified_samples, percentiles)
        
        # Compute the mean of the post-stratified samples
        post_stratified_mean = np.mean(post_stratified_samples)
        
        # Append the result for this region/variable
        results.append({
            var_post_strat: region,
            f'{percentiles[0]}percentile': pcts[0],
            f'{percentiles[1]}percentile': pcts[1],
            f'{percentiles[2]}percentile': pcts[2],
            'mean': post_stratified_mean
        })
    
    # Convert the results to a DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")

    # Define named arguments
    parser.add_argument('--var_post_strat', type = str,  default = 'itl3', help = 'Variable which defines the units for post stratification')
    parser.add_argument('--var_name', type = str,  default = 'p', help = 'Variable for which post stratification is done, must be one of p,p_1,p_2,p_2,p_4')
    parser.add_argument('--post_strat_path', type = str, default = './post_strat/uk', help = 'path to directory for storing post stratified results')
    parser.add_argument('--fit_path', type = str, default = './idata/uk/rsv_intent_ord_soc.nc', help = 'path to directory where inference results are stored')
    parser.add_argument('--dat_for_inference_path', type = str, default = './dat/dat_for_inference/uk', help = 'path to directory where data used in inference is stored')

    args = parser.parse_args()

    var_post_strat = args.var_post_strat
    var_name = args.var_name
    post_strat_path = args.post_strat_path
    fit_path = args.fit_path
    dat_for_inference_path = args.dat_for_inference_path

    os.path.join( dat_for_inference_path, )

    idata_mrp_ord_uk = az.from_netcdf(fit_path)
    with open( os.path.join( dat_for_inference_path, 'var_mappings_mrp_ord.p' ), 'rb') as f:
        var_mappings = pickle.load(f)
    with open( os.path.join( dat_for_inference_path, 'data_dict_mrp_ord.p' ), 'rb') as f:
        data_dict = pickle.load(f)
    survey = pd.read_csv( os.path.join( dat_for_inference_path, 'survey_mrp_ord.tsv' ), sep = '\t', index_col = 0)
    micro = pd.read_csv( os.path.join( dat_for_inference_path, 'micro_mrp_ord.tsv' ), sep = '\t', index_col = 0)

    #############
    lad21_to_itl21 = pd.read_csv('./dat/region_mappings/Local_Authority_District_(April_2021)_to_LAU1_to_ITL3_to_ITL2_to_ITL1_(January_2021)_Lookup_in_United_Kingdom.csv')
    itl321_cd_to_itl221_cd = dict( zip (lad21_to_itl21['ITL321CD'], lad21_to_itl21['ITL221CD']) )
    itl321_cd_to_itl121_cd = dict( zip (lad21_to_itl21['ITL321CD'], lad21_to_itl21['ITL121CD']) )
    itl321_cd_to_itl221_cd['TLM61_64_65_66'] = 'TLM6'
    itl321_cd_to_itl121_cd['TLM61_64_65_66'] = 'TLM'

    micro['itl2'] = micro['itl3'].map(itl321_cd_to_itl221_cd)
    micro['itl1'] = micro['itl3'].map(itl321_cd_to_itl121_cd)

    survey['itl2'] = survey['itl3'].map(itl321_cd_to_itl221_cd)
    survey['itl1'] = survey['itl3'].map(itl321_cd_to_itl121_cd)

    micro['Marginal'] = 'marginal'
    survey['Marginal'] = 'marginal'
    #############

    data_w_weights = micro[[var_post_strat,'n']].copy()

    y_prob = idata_mrp_ord_uk.posterior['y_log_prob'].values
    y_prob = np.exp( y_prob )

    num_samples = y_prob.shape[0] * y_prob.shape[1] 
    y_prob = y_prob.reshape( ( num_samples, y_prob.shape[2], y_prob.shape[3]) )

    y_prob = np.transpose(y_prob, (1, 0, 2) )


    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    p1_labels = [f'p1_{s}' for s in range(num_samples)]
    p2_labels = [f'p2_{s}' for s in range(num_samples)]
    p3_labels = [f'p3_{s}' for s in range(num_samples)]
    p4_labels = [f'p4_{s}' for s in range(num_samples)]
    p_labels = [f'p_{s}' for s in range(num_samples)]

    if var_name == 'p1':
        data_w_weights[p1_labels] = y_prob[:,:,0]
    elif var_name == 'p2':
        data_w_weights[p2_labels] = y_prob[:,:,1]
    elif var_name == 'p3':
        data_w_weights[p3_labels] = y_prob[:,:,2]
    elif var_name == 'p4':
        data_w_weights[p4_labels] = y_prob[:,:,3]
    elif var_name == 'p':
        data_w_weights[p_labels] = y_prob[:,:,2] + y_prob[:,:,3]
    else:
        raise ValueError('non valid var_name')
    
    df_post_strat = post_stratify(data_w_weights, percentiles = [2.5, 50, 97.5], num_samples = 2000, var_name = var_name, var_post_strat = var_post_strat)

    df_post_strat.set_index(var_post_strat).to_csv(  os.path.join(post_strat_path, f'{var_post_strat}_post_strat_rsv_intent_{var_name}.tsv'), sep = '\t')