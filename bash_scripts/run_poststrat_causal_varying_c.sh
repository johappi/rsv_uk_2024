#!/bin/bash

# Define the arrays for var_post_strat and prob_name_pre
# var_post_strat_arr=('no_post_strat_var' 'AGE' 'SEX' 'EDU' 'REL' 'ETH' 'LAN' 'MIG' 'BUK' 'ENP' 'DIS' 'HEA' 'EMP' 'itl3' 'country')
var_post_strat_arr=('no_post_strat_var' 'AGE' 'SEX' 'EDU' 'REL' 'ETH' 'LAN' 'MIG' 'BUK' 'DIS' 'HEA' 'EMP' 'itl3' 'country')
# var_post_strat_arr=('no_post_strat_var')
prob_name_pre_arr=('p1' 'p2' 'p3' 'p4')
# pre_or_post_arr=('pre' 'post')
treat_group_arr=(0 1 2 3 4 5 6)

# Define the path to the poststratify.py script
poststratify_script_path="../scripts/poststratify_causal_varying_c.py"

# Loop over all combinations of var_post_strat and prob_name_pre
for var_post_strat in "${var_post_strat_arr[@]}"; do
  for prob_name_pre in "${prob_name_pre_arr[@]}"; do
    for treat_group in "${treat_group_arr[@]}"; do
      echo "Running $poststratify_script_path for var_post_strat=$var_post_strat, prob_name_pre=$prob_name_pre, treat_group=$treat_group"
      # Run the Python script with the current combination of arguments
      python "$poststratify_script_path" \
        --var_post_strat "$var_post_strat" \
        --prob_name_pre "$prob_name_pre" \
        --treat_group $treat_group \
        --post_strat_path "./post_strat/uk/rsv_causal_varying_c" \
        --fit_path "./idata/uk/rsv_intent_causal_w_soc.nc" \
        --dat_for_inference_path "./dat/dat_for_inference/uk"
    done
  done
done

