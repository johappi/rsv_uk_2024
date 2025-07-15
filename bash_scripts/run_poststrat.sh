#!/bin/bash

# Define the arrays for var_post_strat and var_name
# var_post_strat_arr=('AGE' 'SEX' 'EDU' 'REL' 'ETH' 'LAN' 'MIG' 'BUK' 'ENP' 'DIS' 'HEA' 'EMP' 'itl3' 'country')
# var_post_strat_arr=('AGE' 'SEX' 'EDU' 'REL' 'ETH' 'LAN' 'MIG' 'BUK' 'DIS' 'HEA' 'EMP' 'itl1' 'itl2' 'itl3' 'country' 'Marginal')
var_post_strat_arr=('Marginal')
var_name_arr=('p' 'p1' 'p2' 'p3' 'p4')

# Define the path to the poststratify.py script
poststratify_script_path="../scripts/poststratify.py"

# Loop over all combinations of var_post_strat and var_name
for var_post_strat in "${var_post_strat_arr[@]}"; do
  for var_name in "${var_name_arr[@]}"; do
    echo "Running poststratify.py for var_post_strat=$var_post_strat and var_name=$var_name"
    
    # Run the Python script with the current combination of arguments
    python "$poststratify_script_path" \
      --var_post_strat "$var_post_strat" \
      --var_name "$var_name" \
      --post_strat_path "./post_strat/uk" \
      --fit_path "./idata/uk/rsv_intent_ord_soc.nc" \
      --dat_for_inference_path "./dat/dat_for_inference/uk"
  done
done
