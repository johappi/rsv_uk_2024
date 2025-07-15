import pandas as pd
import numpy as np
    
def apply_rename_dict(df, rename_dict, delete_vars=[]):
    _df = df.copy()
    # Apply the renaming dictionaries to each column
    for var, mapping in rename_dict.items():
        if isinstance(mapping,dict):
            if var in _df.columns:
                _df[var] = _df[var].map(mapping).fillna(_df[var])
                print(f'transforming variable {var}')
            else:
                print(f'!!!Warning!!!: variable {var} is not in input data')
        else:
            print(f'{mapping} for variable {var} is not a dictionary')
    
    # Drop the specified variables
    _df = _df.drop(columns=delete_vars) #, errors='ignore')
    
    # Group by all remaining sociodemographic variables and sum the counts
    sociodemographic_vars = [var for var in _df.columns if var != 'n']
    _df = _df.groupby(sociodemographic_vars, as_index=False).agg({'n': 'sum'})
    return _df