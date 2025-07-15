import argparse
import arviz as az
from cmdstanpy import CmdStanModel
from cmdstanpy import set_cmdstan_path
import numpy as np
import pandas as pd
#import argparse
import sys
import os
from pathlib import Path

default_root = Path(__file__).resolve().parents[1]
work_dir = Path(os.getenv('WORK_DIR', default_root))

os.chdir(work_dir)
cwd = os.getcwd()
print(cwd)

sys.path.insert(0, './src/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")

    # Define named arguments
    parser.add_argument('--model_variant', type=str, required=True, help='which model to fit')
    
    parser.add_argument('--country', type=str, default = 'uk', help='Country')
    
    parser.add_argument('--fit_path', type=str, default = None, help='path for storing arviz idata')

    parser.add_argument('--imputed', action='store_true', help='Whether to use imputed survey')

    parser.add_argument('--include_enp', action='store_true', help='Whether to include Variable ENP (English Proficiency)')

    parser.add_argument('--max_num_imp', type=int, default=100, help='Number of imputed surveys to use, only relevant if imputed==True')

    parser.add_argument('--num_chains', type=int, default=4, help='Number of chains for HMC')

    parser.add_argument('--iter_warmup', type=int, default=1000, help='Number of warmup steps for HMC')

    parser.add_argument('--iter_sampling', type=int, default=500, help='Number of posterior samples per chain')

    parser.add_argument('--adapt_delta', type=float, default = None, help = 'adapt_delta parameter for HMC')

    parser.add_argument('--stan_seed', type=int, default=42, help='Random seed for stan sampling')

    parser.add_argument('--thin', type=int, default=5, help='Thinning parameter for MCMC samples')
    # parser.add_argument('--imp_i', type=int, default=None, help='Number of imputed survey, only relevant if imputed==True')

    args = parser.parse_args()

    model_variant = args.model_variant
    country = args.country
    fit_path = args.fit_path
    imputed = args.imputed
    max_num_imp = args.max_num_imp

    num_chains = args.num_chains
    iter_warmup = args.iter_warmup
    iter_sampling = args.iter_sampling
    adapt_delta = args.adapt_delta

    include_enp = args.include_enp

    stan_seed = args.stan_seed
    thin = args.thin

    if country == 'uk':
        from fit.uk import fit_stan
    elif country == 'ew':
        from fit.ew import fit_stan
    elif country == 'scot':
        from fit.scot import fit_stan
    elif country == 'ni':
        from fit.ni import fit_stan
    else:
        raise ValueError('non valid country')

    if imputed:
        imp_i = 1
        while os.path.exists(f'./dat/questionnaire/imputed_socdems/survey_imp_{imp_i}.tsv') and (imp_i <= max_num_imp):
            fit_stan(model_variant = model_variant, imputed = True, imp_i = imp_i, fit_path = fit_path,
                      num_chains = num_chains, iter_warmup = iter_warmup, iter_sampling = iter_sampling, adapt_delta = adapt_delta, seed = stan_seed, thin = thin)
            print(f'fitted to imputed data {imp_i}')
            imp_i += 1

    else:
        fit_stan(model_variant = model_variant, imputed = False, fit_path = fit_path,
                      num_chains = num_chains, iter_warmup = iter_warmup, iter_sampling = iter_sampling, adapt_delta = adapt_delta, seed = stan_seed, include_enp = include_enp, thin = thin)
        

    


