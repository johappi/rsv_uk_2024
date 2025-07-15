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


### for transforming the codebook file
def trs_codebook(df):
    import re
    
    def extract_code_category(label):
        match = re.match(r'(\d+)\.?\s*(.*)', label)
        if match:
            return match.groups()
        return None, None
        
    # Initialize new columns for variable names and descriptions
    df.dropna(how='all', inplace = True)
    df.index = np.arange(len(df))
    
    df['Variable Description'] = None

    df['_Variable name'] = df['Variable name'].astype(str)
    
    # Loop through the DataFrame to assign variable names and descriptions
    new_var_indices = []
    for i in range(len(df)):
        if df.loc[i, '_Variable name'].isupper():
            new_var_indices.append(i)
    
    for i in new_var_indices:      
        if i < len(df):
            df.loc[i, 'Variable Description'] = df.loc[i + 1, 'Variable name']
            df.loc[i + 1, 'Variable name'] = df.loc[i, 'Variable name']

    # Fill down the 'Variable name' and 'Variable Description'
    df['Variable name'] = df['Variable name'].ffill()
    df['Variable Description'] = df['Variable Description'].ffill()
    
    # Fill down other columns
    df['Variable applicable to'] = df['Variable applicable to'].ffill()
    df['Number of categories'] = df['Number of categories'].ffill()
    df['Code -9 percentage'] = df['Code -9 percentage'].ffill()
    df['What is covered by -9 (No code required)?'] = df['What is covered by -9 (No code required)?'].ffill()

    df[['Code', 'Category']] = df[' Codes and Categories labels'].apply(lambda x: pd.Series(extract_code_category(x)))

    # df['Category'] = df['Category'].fillna('Does not apply')
    df['What is covered by -9 (No code required)?'] = df['What is covered by -9 (No code required)?'].fillna('Does not apply') 
    
    df['Category'] = df['Category'].replace([None, np.nan], 'Does not apply')
    
    vars_keep = ['Variable name', 'Variable Description', 'Variable applicable to', 'Number of categories', 'Percentage of Sample', 'Code -9 percentage', 'What is covered by -9 (No code required)?', 'Code', 'Category'] #'' Codes and Categories labels'] # 
    df = df[vars_keep].copy()
    df.loc[pd.isna(df['Code -9 percentage']),'Code -9 percentage'] = 0.0
    return df
### read data

# from https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2022-names-and-codes-in-the-uk/about
ltla22 = pd.read_csv('./dat/region_mappings//Local_Authority_Districts_(December_2022)_Names_and_Codes_in_the_United_Kingdom.csv')

postcode_to_lad22 = pd.read_csv('./dat/region_mappings/postcode_to_lad22_w_itl.tsv', sep = '\t', index_col = 0, low_memory = False)

# from https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland
pop_2021 = pd.read_excel('./dat/census_2021/ukpopestimatesmid2021on2021geographyfinal.xls', sheet_name = 'MYE2 - Persons', header = 7)

lad21_to_itl21 = pd.read_csv('./dat/region_mappings/Local_Authority_District_(April_2021)_to_LAU1_to_ITL3_to_ITL2_to_ITL1_(January_2021)_Lookup_in_United_Kingdom.csv')

micro_scot = pd.read_csv('./dat/safeguarded_microdata_2011/7835tab_8F25D4CE8EF6FEE0DBECC4CE7A8089AB2EE06F7228FD41DE58B0DC5733DEB052_V1/UKDA-7835-tab/tab/safeguarded_grouped_la.tab', sep = '\t')
micro_scot = micro_scot.apply(pd.to_numeric)

codebook_scot = pd.read_excel('./dat/safeguarded_microdata_2011/7835tab_8F25D4CE8EF6FEE0DBECC4CE7A8089AB2EE06F7228FD41DE58B0DC5733DEB052_V1/UKDA-7835-tab/mrdoc/excel/7835_safeguarded_grouped_local_authority_codebook.xls', sheet_name = 'Codebook', header = 1)

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

lad22_to_itl321 = {}
lad22_to_itl221 = {}
lad22_to_itl121 = {}

for la in postcode_to_lad22['ladcd'].unique():
    _df = postcode_to_lad22[postcode_to_lad22['ladcd'] == la].copy()
    _itl3 = _df['ITL321CD'].unique()
    _itl2 = _df['ITL221CD'].unique()
    _itl1 = _df['ITL121CD'].unique()
    lad22_to_itl321[la] = _itl3
    lad22_to_itl221[la] = _itl2
    lad22_to_itl121[la] = _itl1

###

_df = codebook_scot.copy()
_df = trs_codebook(_df)

codebook_scot_cd_to_nm_dict = {}
for v in _df['Variable name'].unique():
    df_sel = _df[_df['Variable name'] == v]
    try:
        new_dict = dict(zip( df_sel['Code'].astype(np.int64), df_sel['Category'] ))
    except:
        new_dict = dict(zip( df_sel['Code'], df_sel['Category'] ))
        print(f'non numeric codes for variable {v}')
    new_dict[-9] = df_sel['What is covered by -9 (No code required)?'].iloc[0]
    codebook_scot_cd_to_nm_dict[v] = new_dict.copy()

###

_var_list_scot = ['AGE','SEX','HLQPS11','RELPS11','ETHNIC','LANGPS11','MOVEFROM', 'COB',
                  'YR_ARRIVALPUK11','LANGPRF','DISABILITY','HEALTH','ECOPUK11','STUDENT',
                  'COUNCIL_AREA_GROUP', 'DPCFAMUK11']

# recode Scotland
micro_scot_recode_dict = {}
quest_recode_to_scot = {}

micro_scot_recode_dict['DPCFAMUK11_recode'] = {
'Does not apply' : 'Does not apply',
'No dependent children' : 'No dependent children',
'One dependent child aged 0-11' : 'One or more dependent children',
'One dependent child aged 12-18' : 'One or more dependent children',
'Two dependent children, youngest aged 0-11' : 'One or more dependent children',
'Two dependent children, youngest aged 12-18' : 'One or more dependent children',
'Three or more dependent children, youngest aged 0-11' : 'One or more dependent children',
'Three or more dependent children, youngest aged 12-18' : 'One or more dependent children',
'Persons not in a family OR persons in other related families ' : 'Does not apply',
}

micro_scot_recode_dict['AGE_recode'] = {
'Does not apply' : 'Does not apply',
'0-4' : '0-15',
'5-9' : '0-15',
'10-15' : '0-15',
'16-18' : '16-18',
'19-24' : '19-24',
'25-29' : '25-34',
'30-34' : '25-34',
'35-39' : '35-44',
'40-44' : '35-44',
'45-49' : '45-54',
'50-54' : '45-54',
'55-59' : '55-64',
'60-64' : '55-64',
'65-69' : '65+',
'70-74' : '65+',
'75-79' : '65+',
'80-84 ' : '65+',
'85-89': '65+',
'90+' : '65+',   
}

quest_recode_to_scot['AGE'] = 'no_changes_needed'

micro_scot_recode_dict['SEX_recode'] = {
'Male' : 'Male',
'Female' : 'Female',
'Persons resident in a communal establishment OR  student/schoolchild living away during term-time' : 'Does not apply',
}

quest_recode_to_scot['SEX'] = 'no_changes_needed'


micro_scot_recode_dict['HLQPS11_recode'] = {
'Does not apply' : 'Does not apply',
'Schoolchildren and full-time students living away from home during term time and all those under the age of 16.' : 'Does not apply',
'No qualifications' : 'level-0',
'Level 1: 0 Grade, Standard Grade, Access 3 Cluster, Intermediate 1 or 2, GCSE, CSE, Senior Certification or equivalent; GSVQ Foundation or Intermediate, SVQ level 1 or 2, SCOTVEC Module, City and Guilds Craft or equivalent; Other school qualifications not already mentioned (including foreign qualifications).' : 'level-1',
'Level 2: SCE Higher Grade, Higher, Advanced Higher, CSYS, A Level, AS Level, Advanced Senior Certificate or equivalent; GSVQ Advanced, SVQ level 3, ONC, OND, SCOTVEC National Diploma, City and Guilds Advanced Craft or equivalent.' : 'level-2',
'Level 3: HNC, HND, SVQ level 4 or equivalent; Other post-school but pre-Higher Education qualifications not already mentioned (including foreign qualifications).' : 'level-3',
'Level 4 and above: Degree, Postgraduate qualifications, Masters, PhD, SVQ level 5 or equivalent; Professional qualifications (for example, teaching, nursing, accountancy); Other Higher Education qualifications not already mentioned (including foreign qualifications).' : 'level-4',
}    

quest_recode_to_scot['EDU'] = {
    'Does not apply': 'Does not apply',
    'No qualifications' : 'level-0',
    '1 to 4 GCSEs grade A* to C, Any GCSEs at other grades, O levels or CSEs (any grades), 1 AS level, NVQ level 1, Foundation GNVQ, Basic or Essential Skills' : 'level-1',
    '5 or more GCSEs (A* to C or 9 to 4), O levels (passes), CSEs (grade 1), School Certification, 1 A level, 2 to 3 AS levels, VCEs, Intermediate or Higher Diploma, Welsh Baccalaureate Intermediate Diploma, NVQ level 2, Intermediate GNVQ, City and Guilds Craft, BTEC First or General Diploma, RSA Diploma' : 'level-2',
    'Other: apprenticeships, vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales' : 'level-2',
    '2 or more A levels or VCEs, 4 or more AS levels, Higher School Certificate, Progression or Advanced Diploma, Welsh Baccalaureate Advance Diploma, NVQ level 3; Advanced GNVQ, City and Guilds Advanced Craft, ONC, OND, BTEC National, RSA Advanced Diploma' : 'level-3',
    'Degree (BA, BSc), higher degree (MA, PhD, PGCE), NVQ level 4 to 5, HNC, HND, RSA Higher Diploma, BTEC Higher level, professional qualifications (for example, teaching, nursing, accountancy)' : 'level-4',
}

micro_scot_recode_dict['RELPS11_recode'] = {
'Does not apply' : 'Does not apply',
'Student/schoolchild living away during term-time' : 'Does not apply', 
'No religion' : 'No religion',
'Christian ' : 'Christian',
'Buddhist' : 'Buddhist',
'Hindu' : 'Hindu',
'Muslim' : 'Muslim',
'Sikh' : 'Sikh',
'Jewish' : 'Jewish',
'Other religion' : 'Other religion',
'Not stated' : 'Do not wish to answer',
}

quest_recode_to_scot['REL'] = 'no_changes_needed'
# REL no changes needed

micro_scot_recode_dict['ETHNIC_recode'] = {
'Does not apply' : 'Does not apply',
'Schoolchildren and full-time students living away from home during term time.' : 'Does not apply',
'White: Scottish' : 'White: English, Welsh, Scottish, Northern Irish or British',
'White: Other British' : 'White: English, Welsh, Scottish, Northern Irish or British',
'White: Irish' : 'White: Irish',
'White: Gypsy/ Traveller' : 'White: Gypsy or Irish Traveller',
'White: Polish' : 'White: Other White',
'White: Other White' : 'White: Other White',
'Mixed or multiple ethnic groups' : 'Mixed or multiple ethnic groups',
'Asian, Asian Scottish or Asian British: Pakistani, Pakistani Scottish or Pakistani British' : 'Asian, Asian British or Asian Welsh: Pakistani',
'Asian, Asian Scottish or Asian British: Indian, Indian Scottish or Indian British' : 'Asian, Asian British or Asian Welsh: Indian',
'Asian, Asian Scottish or Asian British: Other Asian' : 'Asian, Asian British or Asian Welsh: Other Asian',
'African' : 'Black, Black British, Black Welsh, Caribbean or African: African',
'Caribbean or Black' : 'Caribbean or Black',
'Other ethnic groups: Other ethnic group' : 'Other',
}

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
'Other ethnic group' : 'Other ethnic group',
'Prefer not to say' : 'Prefer not to say',
}

micro_scot_recode_dict['LANGPS11_recode'] = {
'Does not apply' : 'Does not apply',
'Schoolchildren and full-time students living away from home during term time.' : 'Does not apply',
'English Only' : 'English (English or Welsh in Wales)',
'Scots' : 'English (English or Welsh in Wales)',
'Polish' : 'Polish',
'Gaelic' : 'Other',
'British Sign Language' : 'English (English or Welsh in Wales)',
'Other' : 'Other',
}

quest_recode_to_scot['LAN']  = {
'Does not apply' : 'Does not apply',
'English (English or Welsh in Wales)' : 'English (English or Welsh in Wales)',
'Polish' : 'Polish' ,
'Romanian' : 'Other',
'Punjabi' : 'Other',
'Urdu' : 'Other',
'Portuguese' : 'Other',
'Spanish' : 'Other',
'Arabic' : 'Other',
'Other' : 'Other',
}

quest_recode_to_scot['HHLAN'] = 'variable_does_not_exist'

micro_scot_recode_dict['MOVEFROM_recode'] = {
'Does not apply' : 'Does not apply',
'Persons aged under 1 OR student/schoolchild living away during term-time OR persons who were living at the address on the front of the questionnaire 1 year before Census day' : 'Does not apply',
'Did not move' : 'No',
'greater than 0 -9km' : 'No',
'greater than 9 - 39km' : 'No',
'greater than 39 - 99km' : 'No',
'greater than 99-119 km' : 'No',
'greater than 119-149 km' : 'No',
'greater than 149-199 km' : 'No',
'greater than 199-249 km' : 'No',
'greater than 249 km and over' : 'No',
'From other UK country ' : 'No',
'From outside UK' : 'Yes'}

quest_recode_to_scot['MIG'] = 'no_changes_needed'

micro_scot_recode_dict['COB_recode'] = {
'Does not apply' : 'Does not apply',
'Schoolchildren and full-time students living away from home during term time.' : 'Does not apply',
'Europe: United Kingdom :England': 'Yes',
'Europe: United Kingdom :Scotland': 'Yes',
'Europe: United Kingdom: Northern Ireland': 'Yes',
'Europe: United Kingdom: Wales': 'Yes',
'Europe: United Kingdom: United Kingdom not otherwise specified': 'Yes',
'Europe: Republic of Ireland': 'No',
'Europe: Other Europe: EU countries: member countries in March 2001': 'No',
'Europe: Other Europe: EU countries: Accession countries April 2001 to March 2011': 'No',
'Europe: Other Europe: Rest of Europe': 'No',
'Africa': 'No',
'Middle East and Asia': 'No',
'The Americas and the Caribbean ': 'No',
'Antartica, Oceania (including Australasia) and Other': 'No'
}

quest_recode_to_scot['BUK'] = 'no_changes_needed'

micro_scot_recode_dict['YR_ARRIVALPUK11_recode'] = {
'Does not apply' : 'Does not apply',
'Student/schoolchild living away during term-time OR person born in the UK ' : 'Does not apply',
'Born in the UK' : 'Born in the UK',
'Before 1941': 'Before 1951',
'1941-1950': 'Before 1951',
'1951-1960': 'Arrived 1951 to 1960',
'1961-1970': 'Arrived 1961 to 1970',
'1971-1980': 'Arrived 1971 to 1980',
'1981-1990': 'Arrived 1981 to 1990',
'1991-2000': 'Arrived 1991 to 2000',
'2001-2003': 'Arrived 2001 to 2010',
'2004-2006': 'Arrived 2001 to 2010',
'2007-2009': 'Arrived 2001 to 2010',
'2010-2011': 'Arrived after 2010',
}


quest_recode_to_scot['YUK'] = 'no_changes_needed' 


# micro_scot_recode_dict['LANGPRF_recode'] = {
# 'Does not apply' : 'Does not apply',
# 'Very well' : 'Very well or well',
# 'Well' : 'Very well or well',
# 'Not well' : 'Not well',
# 'Not at all' : 'Not at all',
# 'Schoolchildren and full-time students living away from home during term time.' : 'Does not apply',
# }

# quest_recode_to_scot['ENP_v2'] = {
# 'Does not apply' : 'Does not apply',
# 'Main language is English (English or Welsh in Wales)' : 'Very well or well',
# 'Very well or well' : 'Very well or well',
# 'Not well' : 'Not well',
# 'Not at all' : 'Not at all',    
# }

# Alternatively keep Very well and well distinct (does not combine with England and Wales)

micro_scot_recode_dict['LANGPRF_recode'] = {
'Does not apply' : 'Does not apply',
'Very well' : 'Very well',
'Well' : 'Well',
'Not well' : 'Not well',
'Not at all' : 'Not at all',
'Schoolchildren and full-time students living away from home during term time.' : 'Does not apply',
}

quest_recode_to_scot['ENP'] = {
'Does not apply' : 'Does not apply',
'Main language is English (English or Welsh in Wales)' : 'Very well',
'Very well' : 'Very well',
'Well' : 'Well',
'Not well' : 'Not well',
'Not at all' : 'Not at all',    
}



micro_scot_recode_dict['DISABILITY_recode'] = {
'Does not apply' : 'Does not apply',
'Day-to-day activities limited a lot' : 'Disabled under the Equality Act: Day-to-day activities limited a lot',
'Day-to-day activities limited a little' : 'Disabled under the Equality Act: Day-to-day activities limited a little',
'Day-to-day activities not limited' : 'Not disabled under the Equality Act',
'Schoolchildren and full-time students living away from home during term time.' : 'Does not apply',
}


quest_recode_to_scot['DIS_v2'] = 'no_changes_needed'

micro_scot_recode_dict['HEALTH_recode'] = {
'Does not apply' : 'Does not apply',
'Very good health': 'Very good',
'Good health': 'Good',
'Fair health': 'Fair',
'Bad health': 'Bad',
'Very bad health': 'Very bad',
'Schoolchildren and full-time students living away from home during term time.' : 'Does not apply',
}

quest_recode_to_scot['HEA'] = 'no_changes_needed'

micro_scot_recode_dict['ECOPUK11_recode'] = {
'Does not apply' : 'Does not apply',
'Economically Active (excluding Full-time students), In Employment, Employee, Part-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Activie (excluding Full-time students), In Employment, Employee, Full-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time students), In employment, Self employed with employees, Part-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time students), In employment, Self employed with employees, Full-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time students), In employment, Self employed without employees, Part-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time students), In employment, Self employed without employees, Full-time' : 'Working part or full-time (including self-employed, excluding students)',
'Economically Active (excluding Full-time students), Seeking work and ready to start within 2 weeks, and Waiting to start a job already obtained and available to start within 2 weeks' : 'Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
'Economically Active Full-time students' : 'Economically Active Full-time students',
'Economically Inactive, Retired' : 'Economically inactive: Retired',
'Economically Inactive, Student' : 'Economically inactive: Student',
'Economically Inactive, Looking after home/family' : 'Economically inactive: Looking after home or family',
'Economically Inactive, Permanently sick/disabled' : 'Economically inactive: Long-term sick or disabled',
'Economically Inactive, Other' : 'Economically inactive: Other',
'Schoolchildren and full-time students living away from home during term time and all those under the age of 16.' : 'Does not apply',
}

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


micro_scot_recode_dict['STUDENT_recode'] = {
'Does not apply' : 'Does not apply',
'Yes' : 'Yes',
'No' : 'No',
'Full-time student/schoolchild aged under 4 years, persons who are not full-time students/schoolchildren ' : 'Does not apply',
}

quest_recode_to_scot['SCH'] = 'no_changes_needed'

###

postcode_ladcd_to_ladnm = dict( zip( postcode_to_lad22[postcode_to_lad22['ITL121NM'] == 'Scotland']['ladcd'], postcode_to_lad22[postcode_to_lad22['ITL121NM'] == 'Scotland']['ladnm'] ) )
postcode_ladnm_to_ladcd = {v:k for k,v in postcode_ladcd_to_ladnm.items() }
codebook_scot_nm_to_cd_gltla =  {v : k for k,v in codebook_scot_cd_to_nm_dict['COUNCIL_AREA_GROUP'].items() }

gltla_nm_to_ltla_nm = {
'Aberdeen City' : ['Aberdeen City'],
'Aberdeenshire & Moray' : ['Aberdeenshire', 'Moray'],
'Dundee City' : ['Dundee City'],
'East Ayrshire' : ['East Ayrshire'],
'East Dunbartonshire & East Renfrewshire' : ['East Dunbartonshire', 'East Renfrewshire'],
'East Lothian & Midlothian' : ['East Lothian', 'Midlothian'],
'Edinburgh, City of' : ['City of Edinburgh'],
'Falkirk' : ['Falkirk'],
'Fife' : ['Fife'],
'Glasgow City' : ['Glasgow City'],
'Highland, Eilean Siar, Orkney Islands and Shetland Islands' : ['Highland', 'Na h-Eileanan Siar', 'Orkney Islands', 'Shetland Islands'],
'Inverclyde, Argyll & Bute' : ['Inverclyde', 'Argyll and Bute'],
'North Ayrshire' : ['North Ayrshire'],
'North Lanarkshire' : ['North Lanarkshire'],
'Perth, Kinross & Angus' : ['Perth and Kinross', 'Angus'],
'Renfrewshire & West Dunbartonshire' : ['Renfrewshire', 'West Dunbartonshire'],
'Scottish Borders, South Ayrshire, Dumfries & Galloway' : ['Scottish Borders', 'South Ayrshire',  'Dumfries and Galloway'],
'South Lanarkshire' : ['South Lanarkshire'],
'Stirling & Clackmannanshire' : ['Stirling', 'Clackmannanshire'],
'West Lothian' : ['West Lothian'],
'All Persons not in a family' : ['Does not apply'],
}

gltla_to_ltla = {}
for k,v in gltla_nm_to_ltla_nm.items():
    print(v)
    if v[0] != 'Does not apply':
        _new_list = [ postcode_ladnm_to_ladcd[i] for i in v  ]
        gltla_to_ltla[ codebook_scot_nm_to_cd_gltla[k] ] = _new_list.copy()
    else:
        pass
        # _new_list = [ 'Does not apply'  ]
        # gltla_to_ltla[ -9 ] = _new_list.copy()

ltla_to_gltla = {}
for k,v in gltla_to_ltla.items():
    for la in v:
        ltla_to_gltla[la] = k


ltla22_w_pop = ltla22.set_index('LAD22CD')

gltla_to_itl3 = {}
gltla_to_itl2 = {}
gltla_to_itl1 = {}

for gla, las in gltla_to_ltla.items():

    tot_pop = np.sum( ltla22_w_pop.loc[las, 'pop'] )
    
    itl3_list = [i for la in las for i in lad22_to_itl321[la]]
    itl3_list = list( set( itl3_list ) )
    gltla_to_itl3[gla] = {itl3 : 0.0 for itl3 in itl3_list}
    for la in las:
        itl3_list = lad22_to_itl321[la]
        num_itl3s = len(itl3_list)
        for itl3 in itl3_list:
            gltla_to_itl3[gla][itl3] += ltla22_w_pop.loc[la, 'pop'] / ( tot_pop * num_itl3s)
    ###
    itl2_list = [i for la in las for i in lad22_to_itl221[la]]
    itl2_list = list( set( itl2_list ) )
    gltla_to_itl2[gla] = {itl2 : 0.0 for itl2 in itl2_list}
    for la in las:
        itl2_list = lad22_to_itl221[la]
        num_itl2s = len(itl2_list)
        for itl2 in itl2_list:
            gltla_to_itl2[gla][itl2] += ltla22_w_pop.loc[la, 'pop'] / ( tot_pop * num_itl2s)
    ###
    itl1_list = [i for la in las for i in lad22_to_itl121[la]]
    itl1_list = list( set( itl1_list ) )
    gltla_to_itl1[gla] = {itl1 : 0.0 for itl1 in itl1_list}
    for la in las:
        itl1_list = lad22_to_itl121[la]
        num_itl1s = len(itl1_list)
        for itl1 in itl1_list:
            gltla_to_itl1[gla][itl1] += ltla22_w_pop.loc[la, 'pop'] / ( tot_pop * num_itl1s)

_gltla_to_itl1 = {codebook_scot_cd_to_nm_dict['COUNCIL_AREA_GROUP'][k]:v for k,v in gltla_to_itl1.items()}
_gltla_to_itl2 = {codebook_scot_cd_to_nm_dict['COUNCIL_AREA_GROUP'][k]:v for k,v in gltla_to_itl2.items()}
_gltla_to_itl3 = {codebook_scot_cd_to_nm_dict['COUNCIL_AREA_GROUP'][k]:v for k,v in gltla_to_itl3.items()}


with open('./dat/region_mappings/scot_gltla_to_itl1.json', 'w') as f:
    json.dump(_gltla_to_itl1, f)

with open('./dat/region_mappings/scot_gltla_to_itl2.json', 'w') as f:
    json.dump(_gltla_to_itl2, f)

with open('./dat/region_mappings/scot_gltla_to_itl3.json', 'w') as f:
    json.dump(_gltla_to_itl3, f)
###

_var_list_scot = ['AGE','SEX','HLQPS11','RELPS11','ETHNIC','LANGPS11','MOVEFROM', 'COB','YR_ARRIVALPUK11','LANGPRF','DISABILITY','HEALTH','ECOPUK11','STUDENT','COUNCIL_AREA_GROUP', 'DPCFAMUK11']


var_cd_to_cat_red_dict = {}
for var_name in _var_list_scot:
    _var_name = f'{var_name}_recode'
    if _var_name in micro_scot_recode_dict:
        # print(_var_name)
        var_cd_to_cat_red_dict[var_name] = compose_dict( codebook_scot_cd_to_nm_dict[var_name], micro_scot_recode_dict[_var_name] )
    else:
        print( f'{var_name} not in recode dictionary' )

def _fn(x):
    if x == -9:
        return 'Does not apply'
    return x
var_cd_to_cat_red_dict['COUNCIL_AREA_GROUP'] = { x: _fn(x) for x in codebook_scot_cd_to_nm_dict['COUNCIL_AREA_GROUP'].keys() }

var_poststrat = [var for var in _var_list_scot if var not in ['REGION', 'DPCFAMUK11'] ] # with emplopyment 
_var_poststrat = [var for var in _var_list_scot if var not in ['REGION'] ]


microdata_sel = micro_scot[_var_poststrat].copy()
for var in _var_poststrat:
    if var in var_cd_to_cat_red_dict:
        #recode_dict = get_label_to_numeric_dict(var_cd_to_cat_red_dict[var])['recode_dict']
        recode_dict = var_cd_to_cat_red_dict[var]
        microdata_sel[var] = microdata_sel[var].map(recode_dict)
    else:
        print(f'{var} not in var_cd_to_cat_red_dict')


microdata_sel = microdata_sel[microdata_sel['AGE'] != '0-15'].copy()

microdata_sel = microdata_sel[ microdata_sel['DPCFAMUK11'].isin(['One or more dependent children'])] 
microdata_sel.drop(columns=['DPCFAMUK11'], inplace = True)

microdata_sel = microdata_sel[~microdata_sel.apply(lambda row: row.astype(str).str.contains('Does not apply')).any(axis=1)]

microdata_counts = microdata_sel.groupby(var_poststrat).size().reset_index(name='n')

microdata_counts.rename(columns={'COUNCIL_AREA_GROUP': 'gltla22cd'}, inplace=True)

###
###

# add 1/3 the poulattio in age group 16-18 to new age group 18-24

df = microdata_counts.copy()
# Convert the 'n' column to float
df['n'] = df['n'].astype(float)

# Step 1: Update '20-24' to '18-24'
df['AGE'] = df['AGE'].replace('19-24', '18-24')

# Step 2: Create a DataFrame for '16-8' rows and remove them from the original DataFrame
df_16_18 = df[df['AGE'] == '16-18']
df = df[df['AGE'] != '16-18']


# Step 3: Combine counts for '18-24' and '16-18'
for idx, row in df[df['AGE'] == '18-24'].iterrows():
    matching_16_18 = df_16_18[(df_16_18.drop(columns=['AGE', 'n']) == row.drop(['AGE', 'n'])).all(axis=1)]
    if not matching_16_18.empty:
        df.at[idx, 'n'] += (1./3.) * matching_16_18['n'].values[0]
        df_16_18 = df_16_18.drop(matching_16_18.index)

# Step 4: Create new rows for remaining '16-18'
df_16_18_remaining = df_16_18.copy()
df_16_18_remaining['AGE'] = '18-24'
df_16_18_remaining['n'] *= (1./3.)

# Append the new rows to the original DataFrame
df = pd.concat([df, df_16_18_remaining], ignore_index=True)

_old_num_rows = microdata_counts.shape[0]
_new_num_rows = df.shape[0]
print( (f'after adding half the population from age group 16-18 to new age group 18-24, \nthe number of rows has changed from'
      f'{_old_num_rows} to {_new_num_rows}') )

microdata_counts = df.copy()

###
###

var_poststrat = [k for k in var_poststrat if k != 'COUNCIL_AREA_GROUP'] + ['gltla22cd']

microdata_itl3_counts = map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl = gltla_to_itl3, itl_name = 'itl3', var_poststrat = var_poststrat)
microdata_itl2_counts = map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl = gltla_to_itl2, itl_name = 'itl2', var_poststrat = var_poststrat)
microdata_itl1_counts = map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl = gltla_to_itl1, itl_name = 'itl1', var_poststrat = var_poststrat)

# display(microdata_itl1_counts)

microdata_counts.to_csv('./dat/trs_microdata/scotland/microdata_counts.tsv', sep = '\t')
microdata_itl3_counts.to_csv('./dat/trs_microdata/scotland/itl3.tsv', sep = '\t')
microdata_itl2_counts.to_csv('./dat/trs_microdata/scotland/itl2.tsv', sep = '\t')
microdata_itl1_counts.to_csv('./dat/trs_microdata/scotland/itl1.tsv', sep = '\t')