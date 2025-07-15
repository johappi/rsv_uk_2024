import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

default_root = Path(__file__).resolve().parents[1]
work_dir = Path(os.getenv('WORK_DIR', default_root))

os.chdir(work_dir)
cwd = os.getcwd()
print(cwd)

sys.path.insert(0, './src/')

# data
postcode_to_lad22 = pd.read_csv('./dat/region_mappings/postcode_to_lad22_w_itl.tsv', sep = '\t', low_memory = False, index_col = 0)

survey = pd.read_csv('./dat/questionnaire/RSV-September14-final.csv', low_memory = False, index_col = 0)


micro_ew = pd.read_csv('./dat/trs_microdata/england_wales/itl3.tsv', sep = '\t', index_col = 0)
micro_scot = pd.read_csv('./dat/trs_microdata/scotland/itl3.tsv', sep = '\t', index_col = 0)
micro_ni = pd.read_csv('./dat/trs_microdata/ni/itl3.tsv', sep = '\t', index_col = 0)

#



survey['OPC'] = survey['OPC'].str.upper()

postcode_to_lad22['OPC'] = postcode_to_lad22['pcds'].apply(lambda x: x.split(' ')[0] )

opc_to_ilt1 = dict(zip(postcode_to_lad22[~postcode_to_lad22['ITL121CD'].isna()].OPC, postcode_to_lad22[~postcode_to_lad22['ITL121CD'].isna()].ITL121CD))
opc_to_ilt2 = dict(zip(postcode_to_lad22[~postcode_to_lad22['ITL221CD'].isna()].OPC, postcode_to_lad22[~postcode_to_lad22['ITL221CD'].isna()].ITL221CD))
opc_to_ilt3 = dict(zip(postcode_to_lad22[~postcode_to_lad22['ITL321CD'].isna()].OPC, postcode_to_lad22[~postcode_to_lad22['ITL321CD'].isna()].ITL321CD))

itl3cd_to_itl1cd = dict(zip(postcode_to_lad22[~postcode_to_lad22['ITL321CD'].isna()].ITL321CD, postcode_to_lad22[~postcode_to_lad22['ITL321CD'].isna()].ITL121CD))
itl3cd_to_itl2cd = dict(zip(postcode_to_lad22[~postcode_to_lad22['ITL321CD'].isna()].ITL321CD, postcode_to_lad22[~postcode_to_lad22['ITL321CD'].isna()].ITL221CD))

itl1_cd_to_ilt1_nm = dict(zip(postcode_to_lad22.ITL121CD, postcode_to_lad22.ITL121NM))

survey['itl1'] = survey['OPC'].map(opc_to_ilt1)
survey['itl2'] = survey['OPC'].map(opc_to_ilt2)
survey['itl3'] = survey['OPC'].map(opc_to_ilt3)

##############################################################
### deal with outer posctocde responses not mapped to itl3 ###
##############################################################

missing_opcs = list( survey[survey['itl3'].isna()]['OPC'].unique() )
print(f'number of outer postcodes not mapped to itl3: {len(missing_opcs)}')
print(f'number of respondents with outer postcodes not mapped to itl3: {survey[survey['itl3'].isna()].shape[0]}')

bad_opc_to_itl3 = { k:np.nan for k in missing_opcs}

bad_opc_to_multiple_itl3 = {}

for _miss_opc in missing_opcs:
    _a = [ k for k in opc_to_ilt3 if _miss_opc in k ]
    _b = [ opc_to_ilt3[k] for k in opc_to_ilt3 if _miss_opc in k ]
    _c = list(set(_b))
    print(f'{_miss_opc}: {_b}')
    if len(_c) == 1:
        bad_opc_to_itl3[_miss_opc] = _b[0]
    if len(_c) >1:
        bad_opc_to_multiple_itl3[_miss_opc] = _c
    if len(_c) == 0:
        _miss_opc_2 = _miss_opc[:-1]
        print(f'Shortend: {_miss_opc_2}')
        _a = [ k for k in opc_to_ilt3 if _miss_opc_2 in k ]
        _b = [ opc_to_ilt3[k] for k in opc_to_ilt3 if _miss_opc_2 in k ]
        _c = list(set(_b))
        print(f'Shortened: {_miss_opc_2}: {_b}')
        if len(_c) == 1:
            bad_opc_to_itl3[_miss_opc] = _b[0]
        
opc_to_ilt3_new = opc_to_ilt3.copy()
opc_to_ilt3_new.update(bad_opc_to_itl3)

survey['itl3_new'] = survey['OPC'].map(opc_to_ilt3_new)

missing_opcs_new = list( survey[survey['itl3_new'].isna()]['OPC'].unique() )
print(f'number of outer postcodes not mapped to itl3: {len(missing_opcs_new)}')
print(f'number of respondents with outer postcodes not mapped to itl3: {survey[survey['itl3_new'].isna()].shape[0]}')

survey = survey.drop('itl3', axis = 1)
survey = survey.rename(columns={'itl3_new' : 'itl3'})

survey['itl2'] = survey['itl3'].map(itl3cd_to_itl2cd)
survey['itl1'] = survey['itl3'].map(itl3cd_to_itl1cd)

##################
##################
##################


REG_to_itl1_nm = {
'South East England' : 'South East (England)', 
'West Midlands' : 'West Midlands (England)', 
'Wales' : 'Wales',
'North West England' : 'North West (England)', 
'Yorkshire and the Humber' : 'Yorkshire and The Humber',
'Northern Ireland' : 'Northern Ireland', 
'East of England' : 'East', 
'Scotland' : 'Scotland', 
'East Midlands' : 'East Midlands (England)',
'South West England' : 'South West (England)', 
'London' :  'London', 
'North East England' : 'North East (England)',
'Other (for example, Jersey, Guernsey, Isle of Man)' : 'Other', 
np.nan : np.nan,
}

survey['REG'] = survey['REG'].map(REG_to_itl1_nm)

survey['itl1_nm'] = survey['itl1'].map(itl1_cd_to_ilt1_nm)

original_socio_dem_vars = ['AGE', 'SEX', 'EDU', 'REL', 'ETH', 'LAN', 'HHLAN', 'MIG', 'BUK',
                           'YUK', 'ENP', 'DIS1','DIS2','Age', 'HEA', 'EMP', 'SCH'] 
non_socio_dem_vars = [k for k in survey.columns if k not in original_socio_dem_vars]

# England and Wales dictionary mapping variables from questionnaire to microdata

_var_list_ew = ['resident_age_18m',
 'sex',
 'highest_qualification',
 'religion_tb',
 'ethnic_group_tb_20b',
 'main_language_detailed_10m',
 'hh_language',
 'migrant_ind',
 'country_of_birth_10a',
 'year_arrival_uk',
 'english_proficiency_5a',
 'disability_4a',
 'health_in_general',
 'economic_activity_status_15m',
 'economic_activity_status_15m_copy']

var_names_quest_to_micro_ew = {
    'AGE' : 'resident_age_18m',
    'SEX' : 'sex',
    # 'REG' : 'region', # is not in microdata
    'EDU' : 'highest_qualification',
    'REL' : 'religion_tb',
    'ETH' : 'ethnic_group_tb_20b',
    'LAN' : 'main_language_detailed_10m',
    'HHLAN' : 'hh_language', 
    'MIG' : 'migrant_ind',
    'BUK' : 'country_of_birth_10a',# 'country_of_birth_3a',
    'YUK' : 'year_arrival_uk',
    'ENP_v2' : 'english_proficiency_5a', #'english_proficiency',
    # 'DIS1' : 'disability_4a', #'disability' # yes ->1,2,3, no -> 4
    'DIS_v2' : 'disability_4a', #'disability'
    'HEA' : 'health_in_general',
    'EMP_v2' : 'economic_activity_status_15m',
    'SCH' : 'economic_activity_status_15m_copy',
}


# Scotland dictionary mapping variables from questionnaire to microdata

_var_list_scot = ['AGE','SEX','HLQPS11','RELPS11','ETHNIC','LANGPS11','MOVEFROM', 'COB',
                  'YR_ARRIVALPUK11','LANGPRF','DISABILITY','HEALTH','ECOPUK11','STUDENT',
                  'COUNCIL_AREA_GROUP', 'DPCFAMUK11']



var_names_quest_to_micro_scot = {
    'AGE' : 'AGE',
    'SEX' : 'SEX',
    # 'REG' : 'region', # is not in microdata
    'EDU' : 'HLQPS11',
    'REL' : 'RELPS11',
    'ETH' : 'ETHNIC',
    'LAN' : 'LANGPS11', # not really the same
    # 'HHLAN' : 'hh_language', # does not exist in Scotland microdata
    'MIG' : 'MOVEFROM',
    'BUK' : 'COB',# 'country_of_birth_3a',
    'YUK' : 'YR_ARRIVALPUK11',
    'ENP_v2' : 'LANGPRF', #'english_proficiency',
    'ENP' : 'LANGPRF', #'english_proficiency',
    # 'DIS1' : 'disability_4a', #'disability' # yes ->1,2,3, no -> 4
    'DIS_v2' : 'DISABILITY', #'disability'
    'HEA' : 'HEALTH',
    'EMP_v2' : 'ECOPUK11',
    'SCH' : 'STUDENT',
}


# Northern Ireland dictionary mapping variables from questionnaire to microdata

_var_list_ni = ['AGEh','SEX','HLQUPUK11', 'RELIGIONNI', 'ETHNICITYNI_G', 
                 'MAINLANGgNI','HHLDLANG11','MOVEFROMg_v2','COBgNI','LANGPRF_v2',
                 'DISABILITY','HEALTH','ECOPUK11','STUDENT',
                 'DPCFAMUK11','LA_CODE_2014']

var_names_quest_to_micro_ni = {
    'AGE' : 'AGEh',
    'SEX' : 'SEX',
    # 'REG' : 'region', # is not in microdata
    'EDU' : 'HLQUPUK11',
    'REL' : 'RELIGIONNI',
    'ETH' : 'ETHNICITYNI_G',
    'LAN' : 'MAINLANGgNI',
    'HHLAN' : 'HHLDLANG11', 
    'MIG' : 'MOVEFROMg_v2',
    'BUK' : 'COBgNI',# 'country_of_birth_3a',
    'YUK' : 'year_arrival_uk',
    'ENP' : 'LANGPRF_v2', #'english_proficiency',
    # 'DIS1' : 'disability_4a', #'disability' # yes ->1,2,3, no -> 4
    'DIS_v2' : 'DISABILITY', #'disability'
    'HEA' : 'HEALTH',
    'EMP_v2' : 'ECOPUK11',
    'SCH' : 'STUDENT',
}


#####
#####
#####

# England Wales

def eng_welsh_ind(lan):
    if lan in ['English', 'Welsh']:
        return 'Yes'
    elif lan in ['Spanish', 'Punjabi', 'Portuguese', 'Urdu', 'Arabic', 'Romanian', 'Polish', 'Other']:
        return 'No'
    else:
        return 'nan'

survey['ENG_OR_WELSH'] = survey['LAN'].apply(eng_welsh_ind)
survey['ENP'] = survey.apply(lambda row: 'Main language is English (English or Welsh in Wales)' if row['ENG_OR_WELSH']=='Yes' else row['ENP'], axis = 1)

ENP_v2_recode_dict = {
    'Main language is English (English or Welsh in Wales)' : 'Main language is English (English or Welsh in Wales)',
    'Very well' : 'Very well or well',
    'Well' : 'Very well or well',
    'Not well' : 'Not well',
    'Not at all' : 'Not at all',
} 

survey['ENP_v2'] = survey['ENP'].map(ENP_v2_recode_dict)

survey['LAN'] = survey['LAN'].apply(lambda x: 'English (English or Welsh in Wales)' if x in ['English','Welsh'] else x )

survey['AGE'] = survey['Age'].map(lambda x: '65+' if x == 'over 65' else x)

survey['_EMP_v2'] = survey.apply(lambda row: f"{row['SCH']}_and_{row['EMP']}", axis=1)

EMP_v2_recode_dict = {
'No_and_In full-time or part-time employment (including self-employed)' : 'Working part or full-time (including self-employed, excluding students)',
'Yes_and_In full-time or part-time employment (including self-employed)' : 'Economically active and full-time student: In employment', 
'No_and_Unemployed and looking after the home or family' : 'Economically inactive: Looking after home or family',
'No_and_Unemployed and long-term sick or disabled' : 'Economically inactive: Long-term sick or disabled',    
'No_and_Unemployed but seeking work or waiting to start a job (and available to work within two weeks)' : 'Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
'Yes_and_Unemployed but seeking work or waiting to start a job (and available to work within two weeks)' : 'Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
'No_and_Unemployed, other' : 'Economically inactive: Other',
'No_and_Unemployed and not looking for work' : 'Economically inactive: Other',
'No_and_Unemployed and retired'            : 'Economically inactive: Retired', 
'Yes_and_Unemployed and looking after the home or family' : 'Economically inactive: Student',
'Yes_and_Unemployed, other' : 'Economically inactive: Student',
'Yes_and_Unemployed and retired' : 'Economically inactive: Student',
'No_and_nan' : np.nan,
'Yes_and_Unemployed and not looking for work' : 'Economically inactive: Student',
'Yes_and_nan' : np.nan,
'Yes_and_Unemployed and long-term sick or disabled' : 'Economically inactive: Student',            
'nan_and_nan' : np.nan,    
}

survey['EMP_v2'] = survey['_EMP_v2'].map(EMP_v2_recode_dict)

survey['_DIS_v2'] = survey.apply(lambda row: f"{row['DIS1']}_and_{row['DIS2']}", axis=1)
DIS_v2_recode_dict = {
    'No_and_nan' : 'Not disabled under the Equality Act',
    'Yes_and_Not at all' : 'Not disabled under the Equality Act',
    'Yes_and_Yes, a little' : 'Disabled under the Equality Act: Day-to-day activities limited a little',
    'Yes_and_Yes, a lot' : 'Disabled under the Equality Act: Day-to-day activities limited a lot',
    'nan_and_nan' : np.nan,
}

survey['DIS_v2'] = survey['_DIS_v2'].map(DIS_v2_recode_dict)

survey['YUK'] = survey.apply(lambda row: 'Born in the UK' if row['BUK'] == 'Yes' else row['YUK'], axis=1)

###
###
###

# Scotland

quest_recode_to_scot = {}

quest_recode_to_scot['AGE'] = 'no_changes_needed'

quest_recode_to_scot['SEX'] = 'no_changes_needed'

quest_recode_to_scot['EDU'] = {
    'Does not apply': 'Does not apply',
    'No qualifications' : 'level-0',
    '1 to 4 GCSEs grade A* to C, Any GCSEs at other grades, O levels or CSEs (any grades), 1 AS level, NVQ level 1, Foundation GNVQ, Basic or Essential Skills' : 'level-1',
    '5 or more GCSEs (A* to C or 9 to 4), O levels (passes), CSEs (grade 1), School Certification, 1 A level, 2 to 3 AS levels, VCEs, Intermediate or Higher Diploma, Welsh Baccalaureate Intermediate Diploma, NVQ level 2, Intermediate GNVQ, City and Guilds Craft, BTEC First or General Diploma, RSA Diploma' : 'level-2',
    'Other: apprenticeships, vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales' : 'level-2',
    '2 or more A levels or VCEs, 4 or more AS levels, Higher School Certificate, Progression or Advanced Diploma, Welsh Baccalaureate Advance Diploma, NVQ level 3; Advanced GNVQ, City and Guilds Advanced Craft, ONC, OND, BTEC National, RSA Advanced Diploma' : 'level-3',
    'Degree (BA, BSc), higher degree (MA, PhD, PGCE), NVQ level 4 to 5, HNC, HND, RSA Higher Diploma, BTEC Higher level, professional qualifications (for example, teaching, nursing, accountancy)' : 'level-4',
}

quest_recode_to_scot['REL'] = 'no_changes_needed'
# REL no changes needed

quest_recode_to_scot['ETH']  = {
'Does not apply' : 'Does not apply',
'Arab' : 'Other',
'Asian, Asian British or Asian Welsh: Bangladeshi' : 'Asian, Asian British or Asian Welsh: Other Asian',
'Asian, Asian British or Asian Welsh: Chinese' : 'Asian, Asian British or Asian Welsh: Other Asian',
'Asian, Asian British or Asian Welsh: Indian' : 'Asian, Asian British or Asian Welsh: Indian',
'Asian, Asian British or Asian Welsh: Pakistani' : 'Asian, Asian British or Asian Welsh: Pakistani',
'Asian, Asian British or Asian Welsh: Other Asian' : 'Asian, Asian British or Asian Welsh: Other Asian',
'Black, Black British, Black Welsh, Caribbean or African: African' : 'Black, Black British, Black Welsh, Caribbean or African: African',
'Black, Black British, Black Welsh, Caribbean or African: Caribbean' : 'Caribbean or Black',
'Black, Black British, Black Welsh, Caribbean or African: Other Black' : 'Caribbean or Black',
'Mixed or Multiple ethnic groups: White and Asian' : 'Mixed or multiple ethnic groups',
'Mixed or Multiple ethnic groups: White and Black African' : 'Mixed or multiple ethnic groups',
'Mixed or Multiple ethnic groups: White and Black Caribbean' : 'Mixed or multiple ethnic groups',
'Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups' : 'Mixed or multiple ethnic groups',
'White: English, Welsh, Scottish, Northern Irish or British' : 'White: English, Welsh, Scottish, Northern Irish or British',
'White: Irish' : 'White: Irish',
'White: Gypsy or Irish Traveller' : 'White: Gypsy or Irish Traveller',
'White: Roma' : 'White: Other White',
'White: Other White' : 'White: Other White' ,
'Other ethnic group' : 'Other',
'Prefer not to say' : 'Prefer not to say',
}

quest_recode_to_scot['LAN']  = {
'Does not apply' : 'Does not apply',
'English (English or Welsh in Wales)' : 'English (English or Welsh in Wales)',
'Polish' : 'Polish' ,
'Romanian' : 'Other',
'Punjabi' : 'Other',
'Urdu' : 'Other',
'Portuguese' : 'Portuguese',
'Spanish' : 'Other',
'Arabic' : 'Other',
'Other' : 'Other',
}

quest_recode_to_scot['HHLAN'] = 'variable_does_not_exist'

quest_recode_to_scot['MIG'] = 'no_changes_needed'

quest_recode_to_scot['BUK'] = 'no_changes_needed'

quest_recode_to_scot['YUK'] = 'no_changes_needed' 

quest_recode_to_scot['ENP_v2'] = {
'Does not apply' : 'Does not apply',
'Main language is English (English or Welsh in Wales)' : 'Very well or well',
'Very well or well' : 'Very well or well',
'Not well' : 'Not well',
'Not at all' : 'Not at all',    
}

# Alternatively keep Very well and well distinct (does not combine with England and Wales)

# micro_scot_recode_dict['_LANGPRF_recode'] = {
# 'Does not apply' : 'Does not apply',
# 'Very well' : 'Very well',
# 'Well' : 'Well',
# 'Not well' : 'Not well',
# 'Not at all' : 'Not at all',
# 'Schoolchildren and full-time students living away from home during term time.' : 'Does not apply',
# }

quest_recode_to_scot['ENP'] = {
'Does not apply' : 'Does not apply',
'Main language is English (English or Welsh in Wales)' : 'Very well',
'Very well' : 'Very well',
'Well' : 'Well',
'Not well' : 'Not well',
'Not at all' : 'Not at all',    
}

quest_recode_to_scot['DIS_v2'] = 'no_changes_needed'

quest_recode_to_scot['HEA'] = 'no_changes_needed'

quest_recode_to_scot['EMP_v2'] = {
'Working part or full-time (including self-employed, excluding students)' : 'Working part or full-time (including self-employed, excluding students)',
'Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks' : 'Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
'Economically active and full-time student: In employment' : 'Economically Active Full-time students',
'Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks' : 'Economically Active Full-time students',
'Economically inactive: Retired' : 'Economically inactive: Retired',    
'Economically inactive: Student' : 'Economically inactive: Student',
'Economically inactive: Looking after home or family' : 'Economically inactive: Looking after home or family',
'Economically inactive: Long-term sick or disabled' : 'Economically inactive: Long-term sick or disabled', #?
'Economically inactive: Other' : 'Economically inactive: Other',
'Does not apply' : 'Does not apply',
}

quest_recode_to_scot['SCH'] = 'no_changes_needed'


trs_micro_to_quest_scot = {}
for k, v in var_names_quest_to_micro_scot.items():
    _recode = quest_recode_to_scot[k]
    if _recode == 'needs_to_be_dropped':
        pass
    elif _recode == 'no_changes_needed':
        print(f'no changes needed for variable {k}')
        trs_micro_to_quest_scot[v] = k 
    else:
        _k = f'{k}_scot'
        survey[_k] = survey[k].map(_recode)
        print(f'created variable _k')
        trs_micro_to_quest_scot[v] = _k

###
###
###

quest_recode_to_ni = {}

# recode Northern Ireland

quest_recode_to_ni['SEX'] = 'no_changes_needed'

quest_recode_to_ni['AGE'] = 'no_changes_needed'

quest_recode_to_ni['EDU'] = 'no_changes_needed'

quest_recode_to_ni['REL'] = {
    'Does not apply': 'Does not apply',
    'No religion': 'No religion',
    'Christian': 'Christian',
    'Buddhist': 'Other religion',
    'Hindu': 'Other religion',
    'Jewish': 'Other religion',
    'Muslim': 'Other religion',
    'Sikh': 'Other religion',
    'Other religion': 'Other religion',
    'Do not wish to answer': 'Do not wish to answer'
}

quest_recode_to_ni['ETH'] = {
'Does not apply': 'Does not apply',
'Asian, Asian British or Asian Welsh: Bangladeshi': 'Other',
'Asian, Asian British or Asian Welsh: Chinese': 'Other',
'Asian, Asian British or Asian Welsh: Indian': 'Other',
'Asian, Asian British or Asian Welsh: Pakistani': 'Other',
'Asian, Asian British or Asian Welsh: Other Asian': 'Other',
'Black, Black British, Black Welsh, Caribbean or African: African': 'Other',
'Black, Black British, Black Welsh, Caribbean or African: Caribbean': 'Other',
'Black, Black British, Black Welsh, Caribbean or African: Other Black': 'Other',
'Mixed or Multiple ethnic groups: White and Asian': 'Other',
'Mixed or Multiple ethnic groups: White and Black African': 'Other',
'Mixed or Multiple ethnic groups: White and Black Caribbean': 'Other',
'Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups': 'Other',
'White: English, Welsh, Scottish, Northern Irish or British': 'White',
'White: Irish': 'White',
'White: Gypsy or Irish Traveller': 'White',
'White: Roma': 'White',
'White: Other White': 'White',
'Arab': 'Other',
'Other ethnic group': 'Other',
'Prefer not to say' : np.nan,
}


quest_recode_to_ni['LAN'] = {
'Does not apply' : 'Does not apply',
'English (English or Welsh in Wales)' : 'English (English or Welsh in Wales)',
'Polish' : 'Polish',
'Romanian': 'Other',
'Punjabi': 'Other',
'Urdu': 'Other',
'Portuguese' : 'Portuguese',
'Spanish': 'Other',
'Arabic': 'Other',
'Other': 'Other',
}


quest_recode_to_ni['HHLAN'] = 'no_changes_needed'

# MIG no changes

quest_recode_to_ni['MIG'] = 'no_changes_needed'

# BUK no changes

quest_recode_to_ni['BUK'] = 'no_changes_needed'

# YUK has to be dropped (only year of arrival in Northern Ireland)

quest_recode_to_ni['YUK'] = 'needs_to_be_dropped'

# ENP no changes

quest_recode_to_ni['ENP'] = 'no_changes_needed'

# HEA no changes needed

quest_recode_to_ni['HEA'] = 'no_changes_needed'

# DIS_v2 no changes needed

quest_recode_to_ni['DIS_v2'] = 'no_changes_needed'

# emp_v2 no changes needed

quest_recode_to_ni['EMP_v2'] = 'no_changes_needed'

quest_recode_to_ni['SCH'] = 'no_changes_needed'

trs_micro_to_quest_ni = {}
for k, v in var_names_quest_to_micro_ni.items():
    _recode = quest_recode_to_ni[k]
    print(f'now deealing with variable {k}')
    if _recode == 'needs_to_be_dropped':
        pass
    elif _recode == 'no_changes_needed':
        print(f'no changes needed for variable {k}')
        trs_micro_to_quest_ni[v] = k 
    else:
        _k = f'{k}_ni'
        survey[_k] = survey[k].map(_recode)
        print(f'created variable _k')
        trs_micro_to_quest_ni[v] = _k


###
###
###

scot_to_uk_recode = {}
ew_to_uk_recode = {}
ni_to_uk_recode = {}

# Education

ew_to_uk_recode['EDU'] = quest_recode_to_scot['EDU']
ni_to_uk_recode['EDU'] = quest_recode_to_scot['EDU']

# Ethnicity
scot_to_uk_recode['ETH_scot'] = {
        'White: English, Welsh, Scottish, Northern Irish or British' : 'White',
        'White: Other White' : 'White',
        'Asian, Asian British or Asian Welsh: Pakistani' : 'Other',
        'Black, Black British, Black Welsh, Caribbean or African: African' : 'Other',
        'Asian, Asian British or Asian Welsh: Other Asian' : 'Other',
        'Other' : 'Other',
        'Asian, Asian British or Asian Welsh: Indian' : 'Other',
        'Mixed or multiple ethnic groups' : 'Other',
        'White: Irish' : 'White',
        'White: Gypsy or Irish Traveller' : 'White',
        'Caribbean or Black' : 'Other',
}

ew_to_uk_recode['ETH'] = quest_recode_to_ni['ETH']

# Religion

scot_to_uk_recode['REL'] = quest_recode_to_ni['REL']
ew_to_uk_recode['REL'] = quest_recode_to_ni['REL']

# Language

ni_to_uk_recode['LAN_ni'] = {
    'Other' : 'Other',
    'English (English or Welsh in Wales)' : 'English (English or Welsh in Wales)',
    'Polish' : 'Polish',
    'Portuguese' : 'Portuguese', # 'Portuguese' : 'Other',
}

ew_to_uk_recode['LAN'] = {
 'Does not apply': 'Does not apply',
 'English (English or Welsh in Wales)': 'English (English or Welsh in Wales)',
 'Polish': 'Polish',
 'Romanian': 'Other',
 'Punjabi': 'Other',
 'Urdu': 'Other',
 'Portuguese': 'Portuguese', # 'Portuguese' : 'Other',
 'Spanish': 'Other',
 'Arabic': 'Other',
 'Other': 'Other'}

scot_to_uk_recode['LAN_scot'] = {
    'Other' : 'Other',
    'English (English or Welsh in Wales)' : 'English (English or Welsh in Wales)',
    'Polish' : 'Polish',
    'Portuguese' : 'Portuguese', # 'Portuguese' : 'Other',
}

# ENP

scot_to_uk_recode['ENP_scot'] = {
    'Very well' : 'Very well or well',
    'Well' : 'Very well or well',
    'Not well' : 'Not well',
    'Not at all' :'Not at all',
}

ni_to_uk_recode['ENP'] = {
'Not well' : 'Not well',
'Main language is English (English or Welsh in Wales)' : 'Very well or well',
'Very well' : 'Very well or well',
'Well' : 'Very well or well',
'Not at all' : 'Not at all'
}

ew_to_uk_recode['ENP_v2'] = {
'Main language is English (English or Welsh in Wales)' : 'Very well or well',
'Very well or well' : 'Very well or well',
'Not well' : 'Not well',
'Not at all' : 'Not at all',
}

# EMP

ni_to_uk_recode['EMP_v2'] = quest_recode_to_scot['EMP_v2']
ew_to_uk_recode['EMP_v2'] = quest_recode_to_scot['EMP_v2']




micro_scot = micro_scot.rename(trs_micro_to_quest_scot, axis = 1)
micro_ni = micro_ni.rename(trs_micro_to_quest_ni, axis = 1)

##########################################################
##########################################################
##########################################################

# separate data for EW, SCOT, NI
socio_dems_ew = ['AGE', 'SEX', 'EDU', 'SCH', 'REL', 'ETH', 'LAN', 'HHLAN', 'MIG', 'BUK', 'YUK', 'HEA', 'ENP_v2', 'EMP_v2', 'DIS_v2']


# EW
survey_ew = survey[ non_socio_dem_vars + list(var_names_quest_to_micro_ew)  ]

survey_ew.to_csv('./dat/trs_questionnaire/england_wales/survey_ew.tsv', sep = '\t')

micro_ew.to_csv('./dat/trs_microdata/england_wales/itl3_renamed.tsv', sep = '\t')

# Scotland

survey_scot = survey[ non_socio_dem_vars + list( trs_micro_to_quest_scot.values() ) ]

survey_scot.to_csv('./dat/trs_questionnaire/scotland/survey_scot.tsv', sep = '\t')

micro_scot.to_csv('./dat/trs_microdata/scotland/itl3_renamed.tsv', sep = '\t')

# NI
survey_ni = survey[ non_socio_dem_vars + list( trs_micro_to_quest_ni.values() ) ]

survey_ni.to_csv('./dat/trs_questionnaire/ni/survey_ni.tsv', sep = '\t')

micro_ni.to_csv('./dat/trs_microdata/ni/itl3_renamed.tsv', sep = '\t')

##########################################################
##########################################################
##########################################################
### coarsened micro and survey shared across EW, SCOT, NI

def trs_to_uk(survey, micro_ew, micro_scot, micro_ni, ew_to_uk_recode, scot_to_uk_recode, ni_to_uk_recode, itl = 'itl3', var_names_to_region = None, var_exclude = None, var_include = None):

    assert itl in micro_ew.columns
    assert itl in micro_scot.columns
    assert itl in micro_ni.columns
    
    _micro_ew = micro_ew.copy()
    _micro_scot = micro_scot.copy()
    _micro_ni = micro_ni.copy()
    _survey = survey.copy()

    if var_exclude is None:
        var_exclude = []

    if var_include is None:
        var_include = ['AGE', 'SEX', 'EDU', 'REL', 'ETH', 'LAN', 'HHLAN', 'MIG', 'BUK',
       'YUK', 'ENP', 'DIS', 'HEA', 'EMP', 'SCH']        
    
    if var_names_to_region is None:
        var_names_to_region = {}
        
        var_names_to_region['AGE'] = {'ew' : 'AGE', 'scot' : 'AGE', 'ni' : 'AGE'}
        
        var_names_to_region['SEX'] = {'ew' : 'SEX', 'scot' : 'SEX', 'ni' : 'SEX'}
        
        var_names_to_region['EDU'] = {'ew' : 'EDU', 'scot' : 'EDU_scot', 'ni' : 'EDU'}
        
        var_names_to_region['REL'] = {'ew' : 'REL', 'scot' : 'REL', 'ni' : 'REL_ni'}
        
        var_names_to_region['ETH'] = {'ew' : 'ETH', 'scot' : 'ETH_scot', 'ni' : 'ETH_ni'}
        
        var_names_to_region['LAN'] = {'ew' : 'LAN', 'scot' : 'LAN_scot', 'ni' : 'LAN_ni'}
        
        var_names_to_region['HHLAN'] = {'ew' : 'HHLAN', 'scot' : 'does not exist', 'ni' : 'HHLAN'}
        
        var_names_to_region['MIG'] = {'ew' : 'MIG', 'scot' : 'MIG', 'ni' : 'MIG'}
        
        var_names_to_region['BUK'] = {'ew' : 'BUK', 'scot' : 'BUK', 'ni' : 'BUK'}
        
        var_names_to_region['YUK'] = {'ew' : 'YUK', 'scot' : 'YUK', 'ni' : 'does not exist'}
        
        var_names_to_region['ENP'] = {'ew' : 'ENP_v2', 'scot' : 'ENP_scot', 'ni' : 'ENP'}
        
        var_names_to_region['DIS'] = {'ew' : 'DIS_v2', 'scot' : 'DIS_v2', 'ni' : 'DIS_v2'}
        
        var_names_to_region['HEA'] = {'ew' : 'HEA', 'scot' : 'HEA', 'ni' : 'HEA'}
        
        var_names_to_region['EMP'] = {'ew' : 'EMP_v2', 'scot' : 'EMP_v2_scot', 'ni' : 'EMP_v2'}
        
        var_names_to_region['SCH'] = {'ew' : 'SCH', 'scot' : 'SCH', 'ni' : 'SCH'}


    for var in var_include:
        _dict = var_names_to_region[var]
        _var_ew, _var_scot, _var_ni = _dict['ew'], _dict['scot'], _dict['ni']
        if 'does not exist' in [_var_ew, _var_scot, _var_ni]:
            var_exclude.append(var)
    
    new_socio_dems = []
    
    for var in var_include:
        if var not in var_exclude:
            _dict = var_names_to_region[var]
            _var_ew, _var_scot, _var_ni = _dict['ew'], _dict['scot'], _dict['ni']
            
            if _var_ew in ew_to_uk_recode:
                _micro_ew[f'{var}_uk'] = _micro_ew[_var_ew].map( ew_to_uk_recode[_var_ew] )
                _survey[f'{var}_uk'] = _survey[_var_ew].map( ew_to_uk_recode[_var_ew] )
            else:
                _micro_ew[f'{var}_uk'] = _micro_ew[_var_ew]
                _survey[f'{var}_uk'] = _survey[_var_ew]
                
            if _var_scot in scot_to_uk_recode:
                _micro_scot[f'{var}_uk'] = _micro_scot[_var_scot].map( scot_to_uk_recode[_var_scot] )
            else:
                _micro_scot[f'{var}_uk'] = _micro_scot[_var_scot]
                
            if _var_ni in ni_to_uk_recode:
                _micro_ni[f'{var}_uk'] = _micro_ni[_var_ni].map( ni_to_uk_recode[_var_ni] )
            else:
                _micro_ni[f'{var}_uk'] = _micro_ni[_var_ni]
            
            
            new_socio_dems.append(f'{var}_uk')

    print( f'created variables {new_socio_dems}' )

    new_socio_dems.append(itl)
    
    _micro_ew = _micro_ew[new_socio_dems + ['n']]
    _micro_scot = _micro_scot[new_socio_dems + ['n']]
    _micro_ni = _micro_ni[new_socio_dems + ['n']]

    
    _micro_ew['country'] = 'ew'
    _micro_scot['country'] = 'scot'
    _micro_ni['country'] = 'ni'

    new_socio_dems.append('country')
    
    _micro_uk = pd.concat([_micro_ew, _micro_scot,_micro_ni], ignore_index=True)

    _micro_uk = _micro_uk.groupby(new_socio_dems, as_index = False).agg({'n': 'sum'})
            
    return {'micro_counts' : _micro_uk, 'survey' : _survey, 'var_names' : new_socio_dems}


_res = trs_to_uk(survey = survey, micro_ew = micro_ew, micro_scot = micro_scot, micro_ni = micro_ni, ew_to_uk_recode = ew_to_uk_recode,
                  scot_to_uk_recode = scot_to_uk_recode, ni_to_uk_recode = ni_to_uk_recode,
                  var_names_to_region = None, var_exclude = None, var_include = None)

survey_combined = _res['survey']

micro_uk = _res['micro_counts']

socio_dems_uk = _res['var_names']

socio_dems_uk = [k for k in socio_dems_uk if k not in ['country', 'itl3','itl2','itl1']]

survey_combined = survey_combined[ non_socio_dem_vars + socio_dems_uk ]

survey_combined.to_csv('./dat/trs_questionnaire/uk/survey_coarsened.tsv', sep = '\t')

micro_uk.to_csv('./dat/trs_microdata/uk/micro_coarsened.tsv', sep = '\t')