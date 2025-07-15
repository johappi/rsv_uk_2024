import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path

default_root = Path(__file__).resolve().parents[1]
work_dir = Path(os.getenv('WORK_DIR', default_root))

os.chdir(work_dir)
cwd = os.getcwd()
print(cwd)

sys.path.insert(0, './src/')


### create map between msoas and itl3 from geo boundary files

# from https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/f602e96224994ee5b37d3131cd60803b/shapefile?layers=0
itl3_2021 = gpd.read_file('./dat/geo_boundaries/International_Territorial_Level_3_January_2021_UK_BGC_V3_2022_8770952818551609998/ITL3_JAN_2021_UK_BGC_V3.shp')

# from https://statistics.ukdataservice.ac.uk/dataset/2011-census-geography-boundaries-middle-layer-super-output-areas-and-intermediate-zones
msoa_2011 = gpd.read_file('./dat/geo_boundaries/infuse_msoa_lyr_2011_clipped/infuse_msoa_lyr_2011_clipped.shp')

msoa_2011 = msoa_2011.to_crs(itl3_2021.crs)

# Perform a spatial join to find the intersection
joined = gpd.overlay(msoa_2011, itl3_2021, how='intersection', keep_geom_type=False)

# Calculate the area of each intersection
joined['intersection_area'] = joined.geometry.area

# Find the ITL3 region with the largest intersection area for each MSOA
idx = joined.groupby('geo_code')['intersection_area'].idxmax()
largest_intersections = joined.loc[idx]

# Create the lookup table
lookup_table = largest_intersections[['geo_code', 'ITL321CD']]

# Save the lookup table to a CSV file
lookup_table.to_csv('./dat/region_mappings/msoa2011_to_itl32021_lookup.tsv', index=False, sep = '\t')


### create two mappings from postcode to itl3: one via lads that uniquely map to itl3, 
### and one via msoa for the remainder. The reason for this complicated procedure is that for unique lads not all msoa data is available

lad21_to_itl21 = pd.read_csv('./dat/region_mappings/Local_Authority_District_(April_2021)_to_LAU1_to_ITL3_to_ITL2_to_ITL1_(January_2021)_Lookup_in_United_Kingdom.csv')

msoa11_to_itl321 = pd.read_csv('./dat/region_mappings/msoa2011_to_itl32021_lookup.tsv', sep = '\t')

# from https://geoportal.statistics.gov.uk/datasets/e7824b1475604212a2325cd373946235/about
postcode_to_lad22 = pd.read_csv('./dat/region_mappings/PCD_OA_LSOA_MSOA_LAD_MAY22_UK_LU.csv', encoding='latin-1', low_memory = False)

msoa11_to_itl321_dict = dict( zip(msoa11_to_itl321['geo_code'], msoa11_to_itl321['ITL321CD']))

# lads in postcode lookup whic are not in lad to itl lookup
not_found21 = [r for r in np.unique( postcode_to_lad22['ladcd'].astype(str) )  if r not in  np.unique(lad21_to_itl21['LAD21CD'])]
print( not_found21 )

itl321_cd_to_nm = dict( zip (lad21_to_itl21['ITL321CD'], lad21_to_itl21['ITL321NM']) )
itl221_cd_to_nm = dict( zip (lad21_to_itl21['ITL221CD'], lad21_to_itl21['ITL221NM']) )
itl121_cd_to_nm = dict( zip (lad21_to_itl21['ITL121CD'], lad21_to_itl21['ITL121NM']) )

itl321_to_itl221 = dict( zip (lad21_to_itl21['ITL321CD'], lad21_to_itl21['ITL221CD']) )
itl321_to_itl121 = dict( zip (lad21_to_itl21['ITL321CD'], lad21_to_itl21['ITL121CD']) )

# lads with unique mapping from lad to itl321
lads = (lad21_to_itl21['LAD21CD'].unique() )

lad21_to_itl321_unique = {}

nonunique_lads = []
unique_lads = []

for la in lads:
    _df = lad21_to_itl21[lad21_to_itl21['LAD21CD'] == la]
    if _df.shape[0] == 1:
        lad21_to_itl321_unique[la] = _df['ITL321CD'].values[0]
        unique_lads.append(la)
    else:
        nonunique_lads.append(la)
        print(_df)
for la in not_found21:
    lad21_to_itl321_unique[la] = 'nan' 

print( f'num unique lads: {len(unique_lads)}, num non unique lads: {len(nonunique_lads)}, nu, lads total: {len(lad21_to_itl21['LAD21CD'].unique())}, num itl3 with unique lad: {len(set(lad21_to_itl321_unique.values()))}' )

postcode_nonunique_lads = postcode_to_lad22[postcode_to_lad22['ladcd'].isin(nonunique_lads)]

print(f'For remainder: num msoa in postcode lookup which are not in msoa to itl3 lookup: {postcode_nonunique_lads[~postcode_nonunique_lads['msoa11cd'].isin( msoa11_to_itl321['geo_code'].unique() )]}')

# map postcode to itl3 via unique lads
postcode_to_lad22.loc[postcode_to_lad22['ladcd'].isin(unique_lads), 'ITL321CD'] = postcode_to_lad22[postcode_to_lad22['ladcd'].isin(unique_lads)]['ladcd'].map(lad21_to_itl321_unique)

# map postcode to itl3 via msoa for remainder
postcode_to_lad22.loc[postcode_to_lad22['ladcd'].isin(nonunique_lads), 'ITL321CD'] = postcode_to_lad22[postcode_to_lad22['ladcd'].isin(nonunique_lads)]['msoa11cd'].map(msoa11_to_itl321_dict)

print(f'num itl3 which are not in postcode to itl3 lookup: {len([r for r in lad21_to_itl21['ITL321CD'].unique() if r not in postcode_to_lad22['ITL321CD'].unique()])}')

postcode_to_lad22['ITL221CD'] = postcode_to_lad22['ITL321CD'].map(itl321_to_itl221)

postcode_to_lad22['ITL121CD'] = postcode_to_lad22['ITL321CD'].map(itl321_to_itl121)

postcode_to_lad22['ITL321NM'] = postcode_to_lad22['ITL321CD'].map(itl321_cd_to_nm)
postcode_to_lad22['ITL221NM'] = postcode_to_lad22['ITL221CD'].map(itl221_cd_to_nm)
postcode_to_lad22['ITL121NM'] = postcode_to_lad22['ITL121CD'].map(itl121_cd_to_nm)

postcode_to_lad22.to_csv('./dat/region_mappings/postcode_to_lad22_w_itl.tsv', sep = '\t')