import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

default_root = Path(__file__).resolve().parents[1]
work_dir = Path(os.getenv('WORK_DIR', default_root))

os.chdir(work_dir)
cwd = os.getcwd()
print(cwd)

sys.path.insert(0, './src/')

def compose_dict(dict1, dict2):
    return {k: dict2[v] for k, v in dict1.items()}

def get_label_to_numeric_dict(d, default_vals = {'Does not apply':-8}, start = 0):
    # labels = list( np.unique( list(d.values() ) ) )
    labels = list(d.values() )
    n = start
    new_dict = {}

    def update_dict(d, key, value, n):
        if key not in d:
            d[key] = value
            n += 1
        return n

    
    for label in labels:
        if label in default_vals:
            new_dict[label] = default_vals[label]
        else:
            n = update_dict(new_dict, label, n, n)
            #new_dict.setdefault(label, n)
            #n += 1
    # return( new_dict )
    return {'recode_dict':compose_dict(d, new_dict),'new_labels_to_code': new_dict}

def expand_by_itl(df, mapping, itl_name, print_every_n = 50000):
    expanded_rows = []
    num_cases = df.shape[0]
    k = 0
    for _, row in df.iterrows():
        gltla = row['gltla22cd']
        if gltla in mapping:
            for itl, proportion in mapping[gltla].items():
                new_row = row.copy()
                new_row[itl_name] = itl
                new_row['n'] = row['n'] * proportion
                expanded_rows.append(new_row)
        if k%print_every_n == 0:
            print(k/num_cases)
        k +=1 
    return pd.DataFrame(expanded_rows)

def map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl, itl_name, var_poststrat):
    new_microdata_counts = expand_by_itl(microdata_counts, gltla_to_itl, itl_name = itl_name)
    new_microdata_counts = new_microdata_counts.drop(columns=['gltla22cd'])
    var_poststrat_wo_gltla = [var for var in var_poststrat if var not in 'gltla22cd']
    new_microdata_counts = new_microdata_counts.groupby([itl_name] + var_poststrat_wo_gltla, as_index=False).agg({'n': 'sum'})
    return new_microdata_counts

### read data

# from https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2022-names-and-codes-in-the-uk/about
ltla22 = pd.read_csv('./dat/region_mappings//Local_Authority_Districts_(December_2022)_Names_and_Codes_in_the_United_Kingdom.csv')

postcode_to_lad22 = pd.read_csv('./dat/region_mappings/postcode_to_lad22_w_itl.tsv', sep = '\t', index_col = 0, low_memory = False)

# from https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland
pop_2021 = pd.read_excel('./dat/census_2021/ukpopestimatesmid2021on2021geographyfinal.xls', sheet_name = 'MYE2 - Persons', header = 7)

lad21_to_itl21 = pd.read_csv('./dat/region_mappings/Local_Authority_District_(April_2021)_to_LAU1_to_ITL3_to_ITL2_to_ITL1_(January_2021)_Lookup_in_United_Kingdom.csv')

micro_ni = pd.read_csv('./dat/safeguarded_microdata_2011/7770tab_422E929A7D4EC04D4349A8D7C3E9B1651C8808B12809F4A317935990C1E27D83_V1/UKDA-7770-tab/tab/ni_safeguarded_la.tab', sep = '\t')
micro_ni = micro_ni.apply(pd.to_numeric)

codebook_ni = pd.read_excel('./dat/safeguarded_microdata_2011/7770tab_422E929A7D4EC04D4349A8D7C3E9B1651C8808B12809F4A317935990C1E27D83_V1/UKDA-7770-tab/mrdoc/excel/7770_ni_safeguarded_la_codebook.xlsx', sheet_name = 'FULL_VARIABLE_CLASSIFICATIONS', )# header = 0 )

### prepare data
itl3cd_to_itl1cd = dict(zip(lad21_to_itl21['ITL321CD'], lad21_to_itl21['ITL121CD']) )
itl3cd_to_itl1nm = dict(zip(lad21_to_itl21['ITL321CD'], lad21_to_itl21['ITL121NM']) )

itl2cd_to_itl1cd = dict(zip(lad21_to_itl21['ITL221CD'], lad21_to_itl21['ITL121CD']) )
itl2cd_to_itl1nm = dict(zip(lad21_to_itl21['ITL221CD'], lad21_to_itl21['ITL121NM']) )

pop_2021['Code'] = pop_2021['Code'].astype(str)
pop_2021['Code'] = pop_2021['Code'].str.strip()
pop_2021['All ages'] = pop_2021['All ages'].astype(np.float64)

pop_2021_ltla22 = pop_2021[pop_2021['Code'].isin(ltla22['LAD22CD'])]
ltla22_to_pop = dict( zip( pop_2021_ltla22['Code'], pop_2021_ltla22['All ages'] ) )
ltla22['pop'] = ltla22['LAD22CD'].map(ltla22_to_pop)

ltla_nm_to_cd = ltla22.set_index('LAD22NM')['LAD22CD'].to_dict()

lad22_to_itl321 = dict( zip( postcode_to_lad22['ladcd'], postcode_to_lad22['ITL321CD'] ) )
lad22_to_itl221 = dict( zip( postcode_to_lad22['ladcd'], postcode_to_lad22['ITL221CD'] ) )
lad22_to_itl121 = dict( zip( postcode_to_lad22['ladcd'], postcode_to_lad22['ITL121CD'] ) )

pop_2021['Code'] = pop_2021['Code'].astype(str)
pop_2021['Code'] = pop_2021['Code'].str.strip()
pop_2021['All ages'] = pop_2021['All ages'].astype(np.float64)

pop_2021_ltla22 = pop_2021[pop_2021['Code'].isin(ltla22['LAD22CD'])]
ltla22_to_pop = dict( zip( pop_2021_ltla22['Code'], pop_2021_ltla22['All ages'] ) )
ltla22['pop'] = ltla22['LAD22CD'].map(ltla22_to_pop)

ltla_nm_to_cd = ltla22.set_index('LAD22NM')['LAD22CD'].to_dict()

lad22_to_itl321 = dict( zip( postcode_to_lad22['ladcd'], postcode_to_lad22['ITL321CD'] ) )
lad22_to_itl221 = dict( zip( postcode_to_lad22['ladcd'], postcode_to_lad22['ITL221CD'] ) )
lad22_to_itl121 = dict( zip( postcode_to_lad22['ladcd'], postcode_to_lad22['ITL121CD'] ) )

###
codebook_ni_cd_to_nm_dict = {}
for v in codebook_ni['VARIABLE'].unique():
    df_sel = codebook_ni[codebook_ni['VARIABLE'] == v]
    if v == 'AGEh':
        new_dict = dict(zip( pd.to_numeric(df_sel['SORT']), df_sel['NAME'] ))
    elif v == 'LA_CODE_2014':
        new_dict = dict(zip( pd.to_numeric(df_sel['SORT']), df_sel['CODE'] ))
    else:
        try:
            new_dict = dict(zip( pd.to_numeric(df_sel['CODE']), df_sel['NAME'] ))
        except:
            print(f'non numeric code values for {v}')
            new_dict = dict( zip(df_sel['CODE'], df_sel['NAME'] ))
    codebook_ni_cd_to_nm_dict[v] = new_dict.copy()

### change main language variable

micro_ni['LANGPRF_v2'] = micro_ni.apply(lambda row : 0 if row['MAINLANGgNI'] == 1 else row['LANGPRF'], axis = 1).astype(np.int64)

codebook_ni_cd_to_nm_dict['LANGPRF_v2'] = codebook_ni_cd_to_nm_dict['LANGPRF'].copy()
codebook_ni_cd_to_nm_dict['LANGPRF_v2'][0] = 'Main language is English (English or Welsh in Wales)'
codebook_ni_cd_to_nm_dict['LANGPRF_v2']

# change moved from somewhere variable

micro_ni['MOVEFROMg_v2'] = micro_ni.apply(lambda row: 0 if row['POPBASESEC'] == 1 else row['MOVEFROMg'], axis=1).astype(np.int64)

codebook_ni_cd_to_nm_dict['MOVEFROMg_v2'] = codebook_ni_cd_to_nm_dict['MOVEFROMg'].copy()
codebook_ni_cd_to_nm_dict['MOVEFROMg_v2'][0] = 'Non-migrant'
codebook_ni_cd_to_nm_dict['MOVEFROMg_v2']

#

_var_list_irl = ['AGEh','SEX','HLQUPUK11', 'RELIGIONNI', 'ETHNICITYNI_G', 
                 'MAINLANGgNI','HHLDLANG11','MOVEFROMg_v2','COBgNI','LANGPRF_v2',
                 'DISABILITY','HEALTH','ECOPUK11','STUDENT',
                 'DPCFAMUK11','LA_CODE_2014'] # 'YRARRNIg'

# recode Northern Ireland

micro_ni_recode_dict = {}

quest_recode_to_ni = {}

micro_ni_recode_dict['DPCFAMUK11_recode'] = {
'No dependent children' : 'No dependent children',
'One dependent child aged 0-9' : 'One or more dependent children',
'One dependent child aged 10-18' : 'One or more dependent children',
'Two dependent children, youngest aged 0-9' : 'One or more dependent children',
'Two dependent children, youngest aged 10-18' : 'One or more dependent children',
'Three or more dependent children, youngest aged 0-9' : 'One or more dependent children',
'Three or more dependent children, youngest aged 10-18' : 'One or more dependent children',
'Persons not in a family and persons in other related families ' : 'Does not apply'
}

micro_ni_recode_dict['AGEh_recode'] = {
    'Aged 0 to 4 years' : '0-15',
    'Aged 5 to 9 years' : '0-15',
    'Aged 10 to 15 years' : '0-15',
    # 'Aged 16 to 19 years' : '16-19',
    'Aged 16 to 19 years' : '16-19',
    'Aged 20 to 24 years' : '20-24',
    'Aged 25 to 29 years' : '25-34',
    'Aged 30 to 34 years' : '25-34',
    'Aged 35 to 39 years' : '35-44',
    'Aged 40 to 44 years' : '35-44',
    'Aged 45 to 49 years' : '45-54',
    'Aged 50 to 54 years' : '45-54',
    'Aged 55 to 59 years' : '55-64',
    'Aged 60 to 64 years' : '55-64',
    'Aged 65 to 69 years' : '65+',
    'Aged 70 to 74 years' : '65+',
    'Aged 75 to 79 years' : '65+',
    'Aged 80 to 84 years' : '65+',
    'Aged 85 to 89 years' : '65+',
    'Aged 90 years and over' : '65+'
}    

quest_recode_to_ni['AGE'] = 'no_changes_needed'

micro_ni_recode_dict['HLQUPUK11_recode'] = {
    'people aged under 16 and students at their non term time address': 'Does not apply',
    'No academic or professional qualifications (England & Wales & Northern Ireland)' : 'No qualifications',
    'Level 1: 1-4 O Levels/CSE/GCSEs (any grades), Entry Level, Foundation Diploma, NVQ level 1, Foundation GNVQ, Basic/Essential Skills (England & Wales & Northern Ireland)': '1 to 4 GCSEs grade A* to C, Any GCSEs at other grades, O levels or CSEs (any grades), 1 AS level, NVQ level 1, Foundation GNVQ, Basic or Essential Skills',
    'Level 2: 5+ O Level (Passes)/CSEs (Grade 1)/GCSEs (Grades A*-C), School Certificate, 1 A Level/ 2-3 AS Levels/VCEs, Intermediate/Higher Diploma, Welsh Baccalaureate Intermediate Diploma, NVQ level 2, Intermediate GNVQ, City and Guilds)': '5 or more GCSEs (A* to C or 9 to 4), O levels (passes), CSEs (grade 1), School Certification, 1 A level, 2 to 3 AS levels, VCEs, Intermediate or Higher Diploma, Welsh Baccalaureate Intermediate Diploma, NVQ level 2, Intermediate GNVQ, City and Guilds Craft, BTEC First or General Diploma, RSA Diploma',
    'Apprenticeship (England & Wales & Northern Ireland)': 'Other: apprenticeships, vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales',
    'Level 3: 2+ A Levels/VCEs, 4+ AS Levels, Higher School Certificate, Progression/Advanced Diploma, NVQ Level 3; Advanced GNVQ, City and Guilds Advanced Craft, ONC, OND, BTEC National, RSA Advanced Diploma (England & Wales & Northern Ireland)': '2 or more A levels or VCEs, 4 or more AS levels, Higher School Certificate, Progression or Advanced Diploma, Welsh Baccalaureate Advance Diploma, NVQ level 3; Advanced GNVQ, City and Guilds Advanced Craft, ONC, OND, BTEC National, RSA Advanced Diploma',
    'Level 4+: Degree (BA, BSc), Higher Degree (MA, PhD, PGCE), NVQ Level 4-5, HNC, HND, RSA Higher Diploma, BTEC Higher level, Foundation degree (NI), Professional Qualifications (Teaching, Nursing, Accountancy) (England & Wales & Northern Ireland)': 'Degree (BA, BSc), higher degree (MA, PhD, PGCE), NVQ level 4 to 5, HNC, HND, RSA Higher Diploma, BTEC Higher level, professional qualifications (for example, teaching, nursing, accountancy)',
    'Other: Vocational/Work-related Qualifications, Foreign Qualifications/ Qualifications gained outside the UK (NI) (Not stated/ level unknown) (England & Wales & Northern Ireland)': 'Other: apprenticeships, vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales'
}

quest_recode_to_ni['EDU'] = 'no_changes_needed'

micro_ni_recode_dict['RELIGIONNI_recode'] = {
    'Students at their non term-time address, no valid response (unimputed variable)' : 'Does not apply',
    'No religion' : 'No religion',
    'Catholic' : 'Christian',
    'Presbyterian Church of Ireland' : 'Christian',
    'Church of Ireland' : 'Christian',
    'Methodist Church of Ireland' : 'Christian',
    'Other Christian (including Christian related)' : 'Christian',
    # 'Buddhist': 'Buddhist',
    # 'Hindu': 'Hindu',
    # 'Jewish': 'Jewish',
    # 'Muslim': 'Muslim',
    # 'Sikh': 'Sikh',
    'Other religions': 'Other religion',
    'Not stated': 'Do not wish to answer'
}

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

REL_quest_recode_to_ni = {
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

micro_ni_recode_dict['ETHNICITYNI_G_recode'] = {
    'Not in Northern Ireland, student/schoolchild at non term-time address': 'Does not apply',
    'White' : 'White',
    'Other' : 'Other',
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
'Prefer not to say' : np.nan, # None
}

micro_ni_recode_dict['MAINLANGgNI_recode'] = {
    'English' : 'English (English or Welsh in Wales)',
    'Polish' : 'Polish',
    'Lithuanian' :  'Other',
    'Irish (Gaelic)' :  'Other',
    'Portuguese' :  'Portuguese',
    'Slovak' :  'Other',
    'Chinese' :  'Other',
    'Tagalog/Filipino' :  'Other',
    'Latvian' :  'Other',
    'Russian' :  'Other',
    'Malayalam' :  'Other',
    'Hungarian' :  'Other',
    'Other' :  'Other',
    'Resident in England or Wales or Scotland, or not aged 3 and over, or student/schoolchild living away during term-time' : 'Does not apply',
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


micro_ni_recode_dict['HHLDLANG11_recode'] = {
    'All adults speak English as a main language / English or Welsh (Wales)' : 'Yes, at least one person in my household has English or Welsh as their main language',
    'At least one (but not all) adults speak/s English as a main language/ English or Welsh (Wales)' : 'Yes, at least one person in my household has English or Welsh as their main language',
    'No adults speak English as a main language/ English or Welsh (Wales) but at least one child does' : 'Yes, at least one person in my household has English or Welsh as their main language',
    'No one speaks English as a main language/ English or Welsh (Wales)' : 'No people in my household have English or Welsh as their main language',
    'Unoccupied household spaces' : 'Does not apply'
}




quest_recode_to_ni['HHLAN'] = 'no_changes_needed'

micro_ni_recode_dict['MOVEFROMg_v2_recode'] = {
'0-2 km': 'No',
'3-4 km': 'No',
'5-6 km': 'No',
'7-9 km': 'No',
'10-14 km': 'No',
'15-19 km': 'No',
'20-29 km': 'No',
'30-39 km': 'No',
'40-49 km': 'No',
'50-59 km': 'No',
'60-79 km': 'No',
'80-99 km': 'No',
'100-119 km': 'No',
'120-149 km': 'No',
'150-199 km': 'No',
'200-249 km': 'No',
'250 km and over': 'No',
'From other UK country (except between England and Wales)': 'No',
'From outside UK' : 'Yes',
'Non-migrant, or aged under one, or student/schoolchild living away during term-time' : 'Does not apply', # this does not include non migrants anymore
'Non-migrant' : 'No',
}

quest_recode_to_ni['MIG'] = 'no_changes_needed'

micro_ni_recode_dict['COBgNI_recode']= {
    'England': 'Yes',
    'Scotland' : 'Yes',
    'Northern Ireland': 'Yes',
    'Wales': 'Yes',
    'UK part not specified': 'Yes',
    'Republic of Ireland': 'No',
    'Channel Islands and the Isle of Man': 'No',
    'EU: Member countries': 'No',
    'EU: Accession countries': 'No',
    'Other Europe': 'No',
    'Africa': 'No',
    'Middle East and Asia': 'No',
    'Americas and Caribbean': 'No',
    'Antartica and Oceania (including Australasia) and other': 'No',
    'Student/schoolchild living away during term-time' : 'Does not apply',
}

quest_recode_to_ni['BUK'] = 'no_changes_needed'

# YUK has to be dropped (only year of arrival in Northern Ireland)

quest_recode_to_ni['YUK'] = 'needs_to_be_dropped'

# micro_ni_recode_dict['LANGPRF_v2_recode'] = {
# 'Very well' : 'Very well or well',
# 'Well' : 'Very well or well',
# 'Not well' : 'Not well',
# 'Not at all' : 'Not at all',
# 'Not applicable' : 'Does not apply',
# 'Main language is English (English or Welsh in Wales)' : 'Main language is English (English or Welsh in Wales)'
# }

micro_ni_recode_dict['LANGPRF_v2_recode'] = {
'Very well' : 'Very well',
'Well' : 'Well',
'Not well' : 'Not well',
'Not at all' : 'Not at all',
'Not applicable' : 'Does not apply',
'Main language is English (English or Welsh in Wales)' : 'Main language is English (English or Welsh in Wales)'
}


# ENP no changes
quest_recode_to_ni['ENP'] = 'no_changes_needed'

micro_ni_recode_dict['HEALTH_recode'] = {
'Very good health': 'Very good',
'Good health': 'Good',
'Fair health': 'Fair',
'Bad health': 'Bad',
'Very bad health': 'Very bad',
'Not applicable' : 'Does not apply'
}

# HEA no changes needed
quest_recode_to_ni['HEA'] = 'no_changes_needed'

micro_ni_recode_dict['DISABILITY_recode'] = {
'Day-to-day activities limited a lot' : 'Disabled under the Equality Act: Day-to-day activities limited a lot',
'Day-to-day activities limited a little' : 'Disabled under the Equality Act: Day-to-day activities limited a little',
'Day-to-day activities not limited' : 'Not disabled under the Equality Act',
'Not usual resident' : 'Does not apply'
}

# DIS_v2 no changes needed

quest_recode_to_ni['DIS_v2'] = 'no_changes_needed'

micro_ni_recode_dict['ECOPUK11_recode'] = {
'Economically Active (excluding Full-time Students), In Employment, Employee, Part-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active( excluding Full-time Students), In Employment, Employee, Full-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time Students), In Employment, Self-employed with employees, Part-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time Students), In Employment, Self-employed with employees, Full-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time Students), In Employment, Self-employed without employees, Part-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time Students), In Employment, Self-employed without employees, Full-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time Students), Unemployed, Seeking work and ready to start in 2 weeks, and Waiting to start a job already obtained and available to start within 2 weeks' : 'Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
'Economically Active Full-time Students, In Employment, Employee, Part-time' : 'Economically active and full-time student: In employment',
'Economically Active Full-time Students, In Employment, Employee, Full-time'  : 'Economically active and full-time student: In employment',
'Economically Active Full-time Students, In Employment, Self-employed' : 'Economically active and full-time student: In employment',
'Economically Active Full-time Students, Unemployed, Seeking work and ready to start in 2 weeks, and Waiting to start a job already obtained and available to start within 2 weeks' : 'Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
'Economically Inactive, Retired' : 'Economically inactive: Retired',
'Economically Inactive, Student' : 'Economically inactive: Student',
'Economically Inactive, Looking after home/family' : 'Economically inactive: Looking after home or family',
'Economically Inactive, Permanently sick/disabled' : 'Economically inactive: Long-term sick or disabled',
'Economically Inactive, Other' : 'Economically inactive: Other',
'People aged under 16 and students living away during term-time' :  'Does not apply'
}

# emp_v2 no changes needed
quest_recode_to_ni['EMP_v2'] = 'no_changes_needed'

quest_recode_to_ni['SEX'] = 'no_changes_needed'

quest_recode_to_ni['SCH'] = 'no_changes_needed'

################
################

var_cd_to_cat_red_dict = {}
for var_name in _var_list_irl:
    _var_name = f'{var_name}_recode'
    if _var_name in micro_ni_recode_dict:
        var_cd_to_cat_red_dict[var_name] = compose_dict( codebook_ni_cd_to_nm_dict[var_name], micro_ni_recode_dict[_var_name] )
    else:
        print( f'{var_name} not in recode dictionary')

var_cd_to_cat_red_dict['SEX'] = codebook_ni_cd_to_nm_dict['SEX']
var_cd_to_cat_red_dict['STUDENT'] = codebook_ni_cd_to_nm_dict['STUDENT']
var_cd_to_cat_red_dict['LA_CODE_2014'] = codebook_ni_cd_to_nm_dict['LA_CODE_2014']

var_poststrat = [var for var in _var_list_irl if var not in ['REGION', 'DPCFAMUK11'] ] # with emplopyment # [var for var in microdata_sel.columns if (var != 'gltla22cd') & (var != 'n')]
_var_poststrat = [var for var in _var_list_irl if var not in ['REGION'] ]


microdata_sel = micro_ni[_var_poststrat].copy()
for var in _var_poststrat:
    if var in var_cd_to_cat_red_dict:
        #recode_dict = get_label_to_numeric_dict(var_cd_to_cat_red_dict[var])['recode_dict']
        recode_dict = var_cd_to_cat_red_dict[var]
        microdata_sel[var] = microdata_sel[var].map(recode_dict)
    else:
        print(f'{var} not in var_cd_to_cat_red_dict')


microdata_sel = microdata_sel[microdata_sel['AGEh'] != '0-15'].copy()

microdata_sel = microdata_sel[ microdata_sel['DPCFAMUK11'].isin(['One or more dependent children'])] 
microdata_sel.drop(columns=['DPCFAMUK11'], inplace = True)

microdata_sel = microdata_sel[~microdata_sel.apply(lambda row: row.astype(str).str.contains('Does not apply')).any(axis=1)]

microdata_counts = microdata_sel.groupby(var_poststrat).size().reset_index(name='n')

microdata_counts.rename(columns={'LA_CODE_2014': 'gltla22cd'}, inplace=True)

# add half the poulattio in age group 16-19 to new age group 18-24

df = microdata_counts.copy()
# Convert the 'n' column to float
df['n'] = df['n'].astype(float)

# Step 1: Update '20-24' to '18-24'
df['AGEh'] = df['AGEh'].replace('20-24', '18-24')

# Step 2: Create a DataFrame for '16-19' rows and remove them from the original DataFrame
df_16_19 = df[df['AGEh'] == '16-19']
df = df[df['AGEh'] != '16-19']


# Step 3: Combine counts for '18-24' and '16-19'
for idx, row in df[df['AGEh'] == '18-24'].iterrows():
    matching_16_19 = df_16_19[(df_16_19.drop(columns=['AGEh', 'n']) == row.drop(['AGEh', 'n'])).all(axis=1)]
    if not matching_16_19.empty:
        df.at[idx, 'n'] += 0.5 * matching_16_19['n'].values[0]
        df_16_19 = df_16_19.drop(matching_16_19.index)

# Step 4: Create new rows for remaining '16-19'
df_16_19_remaining = df_16_19.copy()
df_16_19_remaining['AGEh'] = '18-24'
df_16_19_remaining['n'] *= 0.5

# Append the new rows to the original DataFrame
df = pd.concat([df, df_16_19_remaining], ignore_index=True)

_old_num_rows = microdata_counts.shape[0]
_new_num_rows = df.shape[0]
print( (f'after adding half the population from age group 16-19 to new age group 18-24, \nthe number of rows has changed from'
      f'{_old_num_rows} to {_new_num_rows}') )

microdata_counts = df.copy()

###



gltla_to_itl3 = { la : { lad22_to_itl321[la] : 1.0} for la in list(codebook_ni_cd_to_nm_dict['LA_CODE_2014'].values()) }
gltla_to_itl2 = { la : { lad22_to_itl221[la] : 1.0} for la in list(codebook_ni_cd_to_nm_dict['LA_CODE_2014'].values()) }
gltla_to_itl1 = { la : { lad22_to_itl121[la] : 1.0} for la in list(codebook_ni_cd_to_nm_dict['LA_CODE_2014'].values()) }

with open('./dat/region_mappings/ni_gltla_to_itl1.json', 'w') as f:
    json.dump(gltla_to_itl1, f)

with open('./dat/region_mappings/ni_gltla_to_itl2.json', 'w') as f:
    json.dump(gltla_to_itl2, f)

with open('./dat/region_mappings/ni_gltla_to_itl3.json', 'w') as f:
    json.dump(gltla_to_itl3, f)
###



var_poststrat = [k for k in var_poststrat if k != 'LA_CODE_2014'] + ['gltla22cd']

microdata_itl3_counts = map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl = gltla_to_itl3, itl_name = 'itl3', var_poststrat = var_poststrat)
microdata_itl2_counts = map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl = gltla_to_itl2, itl_name = 'itl2', var_poststrat = var_poststrat)
microdata_itl1_counts = map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl = gltla_to_itl1, itl_name = 'itl1', var_poststrat = var_poststrat)


microdata_counts.to_csv('./dat/trs_microdata/ni/microdata_counts.tsv', sep = '\t')
microdata_itl3_counts.to_csv('./dat/trs_microdata/ni/itl3.tsv', sep = '\t')
microdata_itl2_counts.to_csv('./dat/trs_microdata/ni/itl2.tsv', sep = '\t')
microdata_itl1_counts.to_csv('./dat/trs_microdata/ni/itl1.tsv', sep = '\t')