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


def get_cd_to_cat_dict(microdata_codes,var_name):
    microdata_codes_sel = microdata_codes[microdata_codes['Microdata_variable_name'] == var_name]
    return dict( zip ( microdata_codes_sel['Category_code'], microdata_codes_sel['Category_label']) )



# read data
# from https://geoportal.statistics.gov.uk/datasets/ons::local-authority-districts-december-2022-names-and-codes-in-the-uk/about
ltla22 = pd.read_csv('./dat/region_mappings//Local_Authority_Districts_(December_2022)_Names_and_Codes_in_the_United_Kingdom.csv')

postcode_to_lad22 = pd.read_csv('./dat/region_mappings/postcode_to_lad22_w_itl.tsv', sep = '\t', index_col = 0, low_memory = False)

# from https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland
pop_2021 = pd.read_excel('./dat/census_2021/ukpopestimatesmid2021on2021geographyfinal.xls', sheet_name = 'MYE2 - Persons', header = 7)

lad21_to_itl21 = pd.read_csv('./dat/region_mappings/Local_Authority_District_(April_2021)_to_LAU1_to_ITL3_to_ITL2_to_ITL1_(January_2021)_Lookup_in_United_Kingdom.csv')

# from https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=9155
micro_ew = pd.read_csv('./dat/safeguarded_microdata_2021/9155tab_33C4483534BD7B54C0DB97884C900C91375E78B2150F98868EC4C357E716931F_V1/UKDA-9155-tab/tab/safeguarded_la_final_csv2023_07_12.tab', sep = '\t')
codebook_ew = pd.read_excel('./dat/safeguarded_microdata_2021/9155tab_33C4483534BD7B54C0DB97884C900C91375E78B2150F98868EC4C357E716931F_V1/UKDA-9155-tab/mrdoc/excel/9155_updated_microdata_sample_codes.ods',
                                sheet_name = 'Microdata_sample_codes', header = 5)


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

gltla_codes = codebook_ew[codebook_ew['Microdata_variable_name'] == 'gltla22cd']

ltla_names_w_comma = [l for l in ltla22['LAD22NM'] if ',' in l]
# ltla_names_w_comma_wo_addon = [l[0] for l in ltla_names_w_comma]
ltla_names_w_comma_addon = [l[1] for l in ltla_names_w_comma]


ltla_names_w_comma_wo_addon_to_full_name = {l[0]:l for l in ltla_names_w_comma}

# Initialize the dictionary
gltla_to_ltla = {cd:[] for cd in gltla_codes['Category_code']}

# Iterate over each row in the grouped dataframe
for index, row in gltla_codes.iterrows():
    cds = row['Category_code']
    label = row['Category_label']
    
    if label in ltla_nm_to_cd:
        gltla_to_ltla[cds].append( ltla_nm_to_cd[label] )
    else:
        ltla_names = label.split(', ')
        for name in ltla_names:
            if name in ltla_nm_to_cd:
                gltla_to_ltla[cds].append( ltla_nm_to_cd[name] )
            elif name in ltla_names_w_comma_wo_addon_to_full_name:
                full_name = ltla_names_w_comma_wo_addon_to_full_name[name]
                gltla_to_ltla[cds].append(  ltla_nm_to_cd[full_name] )
            elif name in ltla_names_w_comma_addon:
                pass
            else:
                print(label, name)


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
    
    itl3_list = [lad22_to_itl321[la] for la in las]
    itl3_list = list( set( itl3_list ) )
    gltla_to_itl3[gla] = {itl3 : 0.0 for itl3 in itl3_list}
    for la in las:
        itl3 = lad22_to_itl321[la]
        gltla_to_itl3[gla][itl3] += ltla22_w_pop.loc[la, 'pop'] / tot_pop
    ###
    itl2_list = [lad22_to_itl221[la] for la in las]
    itl2_list = list( set( itl2_list ) )
    gltla_to_itl2[gla] = {itl2 : 0.0 for itl2 in itl2_list}
    for la in las:
        itl2 = lad22_to_itl221[la]
        gltla_to_itl2[gla][itl2] += ltla22_w_pop.loc[la, 'pop'] / tot_pop
    ###
    itl1_list = [lad22_to_itl121[la] for la in las]
    itl1_list = list( set( itl1_list ) )
    gltla_to_itl1[gla] = {itl1 : 0.0 for itl1 in itl1_list}
    for la in las:
        itl1 = lad22_to_itl121[la]
        gltla_to_itl1[gla][itl1] += ltla22_w_pop.loc[la, 'pop'] / tot_pop

###
with open('./dat/region_mappings/ew_gltla_to_itl1.json', 'w') as f:
    json.dump(gltla_to_itl1, f)

with open('./dat/region_mappings/ew_gltla_to_itl2.json', 'w') as f:
    json.dump(gltla_to_itl2, f)

with open('./dat/region_mappings/ew_gltla_to_itl3.json', 'w') as f:
    json.dump(gltla_to_itl3, f)
###


micro_ew['economic_activity_status_15m_copy'] = micro_ew['economic_activity_status_15m'].copy()

var_names_quest_to_micro = {
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

var_list_ew = list(var_names_quest_to_micro.values()) + ['gltla22cd','family_dependent_children_8m'] #,'dependent_child_ind'
codebook_ew_cd_to_nm_dict = {}
for var_name in var_list_ew:
    codebook_ew_cd_to_nm_dict[var_name] = get_cd_to_cat_dict( codebook_ew, var_name )

codebook_ew_cd_to_nm_dict['economic_activity_status_15m_copy'] = codebook_ew_cd_to_nm_dict['economic_activity_status_15m'].copy()

###

# recode England Wales to match survey responses

micro_ew_recode_dict = {}

micro_ew_recode_dict['family_dependent_children_8m'] = {'Does not apply':'Does not apply',
                   'Family with no dependent children':'No dependent children',
                    'One dependent child: Aged 0 to 9 years':'One or more dependent children',
                     'One dependent child: Aged 10 to 18 years':'One or more dependent children',
                    'Two dependent children: Youngest aged 0 to 9 years':'One or more dependent children',
                    'Two dependent children: Youngest aged 10 to 18 years':'One or more dependent children',
                     'Three or more dependent children: Youngest aged 0 to 9 years':'One or more dependent children',
                     'Three or more dependent children: Youngest aged 10 to 18 years':'One or more dependent children'
                   }


micro_ew_recode_dict['resident_age_18m'] = {
'Does not apply' : 'Does not apply',
'Aged 4 years and under' : '0-15',
'Aged 5 to 9 years' : '0-15',
'Aged 10 to 15 years' : '0-15',
'Aged 16 to 18 years' : '16-18',
'Aged 19 to 24 years' : '19-24', # not exact but should be ok
'Aged 30 to 34 years' : '25-34',
'Aged 35 to 39 years' : '35-44',
'Aged 50 to 54 years' : '45-54',
'Aged 40 to 44 years' : '35-44',
'Aged 25 to 29 years' : '25-34',
'Aged 45 to 49 years' : '45-54',
'Aged 55 to 59 years' : '55-64',
'Aged 60 to 64 years' : '55-64',
'Aged 65 to 69 years' : '65+',
'Aged 70 to 74 years' : '65+',
'Aged 75 to 79 years' : '65+',
'Aged 80 to 84 years' : '65+',
'Aged 85 years and over' : '65+'}


# micro_ew_recode_dict['region'] = {
#     'North East': 'North East England',
#     'North West': 'North West England',
#     'Yorkshire and The Humber': 'Yorkshire and the Humber',
#     'East Midlands': 'East Midlands',
#     'West Midlands': 'West Midlands',
#     'East of England': 'East of England',
#     'London': 'London',
#     'South East': 'South East England',
#     'South West': 'South West England',
#     'Does not apply: Northern Ireland': 'Northern Ireland',
#     'Does not apply: Scotland': 'Scotland',
#     'Wales': 'Wales'
# }

micro_ew_recode_dict['highest_qualification'] = {
    'Does not apply': 'Does not apply',
    'No qualifications': 'No qualifications',
    'Level 1 and entry level qualifications: 1 to 4 GCSEs grade A* to C, Any GCSEs at other grades, O levels or CSEs (any grades), 1 AS level, NVQ level 1, Foundation GNVQ, Basic or Essential Skills': '1 to 4 GCSEs grade A* to C, Any GCSEs at other grades, O levels or CSEs (any grades), 1 AS level, NVQ level 1, Foundation GNVQ, Basic or Essential Skills',
    'Level 2 qualifications: 5 or more GCSEs (A* to C or 9 to 4), O levels (passes), CSEs (grade 1), School Certification, 1 A level, 2 to 3 AS levels, VCEs, Intermediate or Higher Diploma, Welsh Baccalaureate Intermediate Diploma, NVQ level 2, Intermediate GNVQ, City and Guilds Craft, BTEC First or General Diploma, RSA Diploma': '5 or more GCSEs (A* to C or 9 to 4), O levels (passes), CSEs (grade 1), School Certification, 1 A level, 2 to 3 AS levels, VCEs, Intermediate or Higher Diploma, Welsh Baccalaureate Intermediate Diploma, NVQ level 2, Intermediate GNVQ, City and Guilds Craft, BTEC First or General Diploma, RSA Diploma',
    'Apprenticeship': 'Other: apprenticeships, vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales',
    'Level 3 qualifications: 2 or more A levels or VCEs, 4 or more AS levels, Higher School Certificate, Progression or Advanced Diploma, Welsh Baccalaureate Advance Diploma, NVQ level 3; Advanced GNVQ, City and Guilds Advanced Craft, ONC, OND, BTEC National, RSA Advanced Diploma': '2 or more A levels or VCEs, 4 or more AS levels, Higher School Certificate, Progression or Advanced Diploma, Welsh Baccalaureate Advance Diploma, NVQ level 3; Advanced GNVQ, City and Guilds Advanced Craft, ONC, OND, BTEC National, RSA Advanced Diploma',
    'Level 4 qualifications or above: degree (BA, BSc), higher degree (MA, PhD, PGCE), NVQ level 4 to 5, HNC, HND, RSA Higher Diploma, BTEC Higher level, professional qualifications (for example, teaching, nursing, accountancy)': 'Degree (BA, BSc), higher degree (MA, PhD, PGCE), NVQ level 4 to 5, HNC, HND, RSA Higher Diploma, BTEC Higher level, professional qualifications (for example, teaching, nursing, accountancy)',
    'Other: vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales (equivalent not stated or unknown)': 'Other: apprenticeships, vocational or work-related qualifications, other qualifications achieved in England or Wales, qualifications achieved outside England or Wales'
}

micro_ew_recode_dict['religion_tb'] = {
    'Does not apply': 'Does not apply',
    'No religion': 'No religion',
    'Christian': 'Christian',
    'Buddhist': 'Buddhist',
    'Hindu': 'Hindu',
    'Jewish': 'Jewish',
    'Muslim': 'Muslim',
    'Sikh': 'Sikh',
    'Other religion': 'Other religion',
    'Not answered': 'Do not wish to answer'
}

micro_ew_recode_dict['ethnic_group_tb_20b'] = {
    'Does not apply': 'Does not apply',
    'Asian, Asian British or Asian Welsh: Bangladeshi': 'Asian, Asian British or Asian Welsh: Bangladeshi',
    'Asian, Asian British or Asian Welsh: Chinese': 'Asian, Asian British or Asian Welsh: Chinese',
    'Asian, Asian British or Asian Welsh: Indian': 'Asian, Asian British or Asian Welsh: Indian',
    'Asian, Asian British or Asian Welsh: Pakistani': 'Asian, Asian British or Asian Welsh: Pakistani',
    'Asian, Asian British or Asian Welsh: Other Asian': 'Asian, Asian British or Asian Welsh: Other Asian',
    'Black, Black British, Black Welsh, Caribbean or African: African': 'Black, Black British, Black Welsh, Caribbean or African: African',
    'Black, Black British, Black Welsh, Caribbean or African: Caribbean': 'Black, Black British, Black Welsh, Caribbean or African: Caribbean',
    'Black, Black British, Black Welsh, Caribbean or African: Other Black': 'Black, Black British, Black Welsh, Caribbean or African: Other Black',
    'Mixed or Multiple ethnic groups: White and Asian': 'Mixed or Multiple ethnic groups: White and Asian',
    'Mixed or Multiple ethnic groups: White and Black African': 'Mixed or Multiple ethnic groups: White and Black African',
    'Mixed or Multiple ethnic groups: White and Black Caribbean': 'Mixed or Multiple ethnic groups: White and Black Caribbean',
    'Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups': 'Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups',
    'White: English, Welsh, Scottish, Northern Irish or British': 'White: English, Welsh, Scottish, Northern Irish or British',
    'White: Irish': 'White: Irish',
    'White: Gypsy or Irish Traveller': 'White: Gypsy or Irish Traveller',
    'White: Roma': 'White: Roma',
    'White: Other White': 'White: Other White',
    'Other ethnic group: Arab': 'Arab',
    'Other ethnic group: Any other ethnic group': 'Other ethnic group',
    'Not answered': 'Prefer not to say'
}

micro_ew_recode_dict['main_language_detailed_10m'] = {
    'Does not apply': 'Does not apply',
    'English (English or Welsh in Wales)': 'English (English or Welsh in Wales)',
    'Other European language (EU): Polish': 'Polish',
    'Other European language (EU): Romanian': 'Romanian',
    'South Asian language: Panjabi': 'Punjabi',
    'South Asian language: Urdu': 'Urdu',
    'Portuguese': 'Portuguese',
    'Spanish': 'Spanish',
    'Arabic': 'Arabic',
    'Other language': 'Other'
}

micro_ew_recode_dict['hh_language'] = {
    'Does not apply': 'Does not apply',
    'All adults in household have English in England, or English or Welsh in Wales as a main language': 'Yes, at least one person in my household has English or Welsh as their main language',
    'At least one but not all adults in household have English in England, or English or Welsh in Wales as a main language': 'Yes, at least one person in my household has English or Welsh as their main language',
    'No adults in household, but at least one person aged 3 to 15 years, has English in England or English or Welsh in Wales as a main language': 'Yes, at least one person in my household has English or Welsh as their main language',
    'No people in household have English in England, or English or Welsh in Wales as a main language': 'No people in my household have English or Welsh as their main language'
}

micro_ew_recode_dict['migrant_ind'] = {
    'Does not apply': 'Does not apply',
    'Address one year ago is the same as the address of enumeration': 'No',
    'Address one year ago is student term-time or boarding school address in the UK': 'No',
    'Migrant from within the UK: Address one year ago was in the UK': 'No',
    'Migrant from outside the UK: Address one year ago was outside the UK': 'Yes'
}


micro_ew_recode_dict['country_of_birth_10a'] = {
    'Does not apply': 'Does not apply',
    'Europe: United Kingdom': 'Yes',
    'Europe: EU countries: Member countries in March 2001: Ireland': 'No',
    'Europe: EU countries: Member countries in March 2001: All other': 'No',
    'Europe: EU countries: Countries that joined the EU between April 2001 and March 2021': 'No',
    'Europe: Rest of Europe': 'No',
    'Africa': 'No',
    'Middle East and Asia': 'No',
    'The Americas and the Caribbean': 'No',
    'Antarctica, Oceania (including Australasia) and other': 'No'
}


micro_ew_recode_dict['year_arrival_uk'] = {
    'Does not apply': 'Does not apply',
    'Born in the UK': 'Born in the UK',
    'Arrived before 1951': 'Before 1951',
    'Arrived 1951 to 1960': 'Arrived 1951 to 1960',
    'Arrived 1961 to 1970': 'Arrived 1961 to 1970',
    'Arrived 1971 to 1980': 'Arrived 1971 to 1980',
    'Arrived 1981 to 1990': 'Arrived 1981 to 1990',
    'Arrived 1991 to 2000': 'Arrived 1991 to 2000',
    'Arrived 2001 to 2010': 'Arrived 2001 to 2010',
    'Arrived 2011 to 2013': 'Arrived after 2010',
    'Arrived 2014 to 2016': 'Arrived after 2010',
    'Arrived 2017 to 2019': 'Arrived after 2010',
    'Arrived 2020 to 2021': 'Arrived after 2010'
}


micro_ew_recode_dict['english_proficiency_5a'] = {
    'Does not apply': 'Does not apply',
    'Main language is English (English or Welsh in Wales)': 'Main language is English (English or Welsh in Wales)',
    'Main language is not English (English or Welsh in Wales): Can speak English very well or well': 'Very well or well',
    'Main language is not English (English or Welsh in Wales): Cannot speak English well': 'Not well',
    'Main language is not English (English or Welsh in Wales): Cannot speak English': 'Not at all'
}


micro_ew_recode_dict['health_in_general'] = {
    'Does not apply': 'Does not apply',
    'Very good health': 'Very good',
    'Good health': 'Good',
    'Fair health': 'Fair',
    'Bad health': 'Bad',
    'Very bad health': 'Very bad'
}

# create variable SCH in microdata whether full time student
micro_ew_recode_dict['economic_activity_status_15m_copy'] = {
    'Economically active (excluding full-time students): In employment: Employee: Part-time' : 'No',
    'Economically active (excluding full-time students): In employment: Employee: Full-time' : 'No',
    'Economically active (excluding full-time students): In employment: Self-employed with employees: Part-time' : 'No',
    'Economically active (excluding full-time students): In employment: Self-employed with employees: Full-time': 'No',
    'Economically active (excluding full-time students): In employment: Self-employed without employees: Part-time' : 'No',
    'Economically active (excluding full-time students): In employment: Self-employed without employees: Full-time' : 'No',
    'Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks' : 'No',
    'Economically active and full-time student: In employment' : 'Yes',
    'Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks' : 'Yes',
    'Economically inactive: Retired' : 'No', ## in questionnaire also Retired and student
    'Economically inactive: Student' : 'Yes',
    'Economically inactive: Looking after home or family' : 'No',
    'Economically inactive: Long-term sick or disabled' : 'No', #?
    'Economically inactive: Other' : 'No',
    'Does not apply' : 'Does not apply'
}


micro_ew_recode_dict['economic_activity_status_15m'] = {
    'Economically active (excluding full-time students): In employment: Employee: Part-time' : 'Working part or full-time (including self-employed, excluding students)',
    'Economically active (excluding full-time students): In employment: Employee: Full-time' : 'Working part or full-time (including self-employed, excluding students)',
    'Economically active (excluding full-time students): In employment: Self-employed with employees: Part-time' : 'Working part or full-time (including self-employed, excluding students)',
    'Economically active (excluding full-time students): In employment: Self-employed with employees: Full-time': 'Working part or full-time (including self-employed, excluding students)',
    'Economically active (excluding full-time students): In employment: Self-employed without employees: Part-time' : 'Working part or full-time (including self-employed, excluding students)',
    'Economically active (excluding full-time students): In employment: Self-employed without employees: Full-time' : 'Working part or full-time (including self-employed, excluding students)',
    'Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks' : 'Economically active (excluding full-time students): Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
    'Economically active and full-time student: In employment' : 'Economically active and full-time student: In employment',
    'Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks' : 'Economically active and a full-time student: Unemployed: Seeking work or waiting to start a job already obtained: Available to start working within 2 weeks',
    'Economically inactive: Retired' : 'Economically inactive: Retired',    
    'Economically inactive: Student' : 'Economically inactive: Student',
    'Economically inactive: Looking after home or family' : 'Economically inactive: Looking after home or family',
    'Economically inactive: Long-term sick or disabled' : 'Economically inactive: Long-term sick or disabled', #?
    'Economically inactive: Other' : 'Economically inactive: Other',
    'Does not apply' : 'Does not apply'
}

micro_ew_recode_dict['disability_4a'] = {
    'Does not apply': 'Does not apply',
    'Disabled under the Equality Act: Day-to-day activities limited a lot': 'Disabled under the Equality Act: Day-to-day activities limited a lot',#'Yes, a lot',
    'Disabled under the Equality Act: Day-to-day activities limited a little': 'Disabled under the Equality Act: Day-to-day activities limited a little',#'Yes, a little',    
    'Not disabled under the Equality Act' : 'Not disabled under the Equality Act'#,'Not disabled under the Equality Act',    
}

micro_ew_recode_dict['sex'] = {
    'Does not apply' : 'Does not apply',
    'Female' : 'Female',
    'Male' : 'Male'
}

###

var_cd_to_cat_red_dict = {}
for var_name in var_list_ew:
    # _var_name = f'{var_name}_recode'
    if var_name in micro_ew_recode_dict:
        # print(_var_name)
        var_cd_to_cat_red_dict[var_name] = compose_dict( codebook_ew_cd_to_nm_dict[var_name], micro_ew_recode_dict[var_name] )
    else:
        print( f'{var_name} not in recode dictionary' )

###

var_poststrat = [var for var in var_list_ew if var not in ['region', 'family_dependent_children_8m'] ] # with emplopyment # [var for var in microdata_sel.columns if (var != 'gltla22cd') & (var != 'n')]
_var_poststrat = [var for var in var_list_ew if var not in ['region'] ]


microdata_sel = micro_ew[_var_poststrat].copy()
for var in _var_poststrat:
    try:
        #recode_dict = get_label_to_numeric_dict(var_cd_to_cat_red_dict[var])['recode_dict']
        recode_dict = var_cd_to_cat_red_dict[var]
        microdata_sel[var] = microdata_sel[var].map(recode_dict)
    except:
        print(var)


microdata_sel = microdata_sel[microdata_sel['resident_age_18m'] != '0-15'].copy()

microdata_sel = microdata_sel[ microdata_sel['family_dependent_children_8m'].isin(['One or more dependent children'])] 
microdata_sel.drop(columns=['family_dependent_children_8m'], inplace = True)

microdata_sel = microdata_sel[~microdata_sel.apply(lambda row: row.astype(str).str.contains('Does not apply')).any(axis=1)]

microdata_counts = microdata_sel.groupby(var_poststrat).size().reset_index(name='n')

###

###

# add 1/3 the poulattio in age group 16-18 to new age group 18-24

df = microdata_counts.copy()
# Convert the 'n' column to float
df['n'] = df['n'].astype(float)

# Step 1: Update '20-24' to '18-24'
df['resident_age_18m'] = df['resident_age_18m'].replace('19-24', '18-24')

# Step 2: Create a DataFrame for '16-8' rows and remove them from the original DataFrame
df_16_18 = df[df['resident_age_18m'] == '16-18']
df = df[df['resident_age_18m'] != '16-18']


# Step 3: Combine counts for '18-24' and '16-18'
for idx, row in df[df['resident_age_18m'] == '18-24'].iterrows():
    matching_16_18 = df_16_18[(df_16_18.drop(columns=['resident_age_18m', 'n']) == row.drop(['resident_age_18m', 'n'])).all(axis=1)]
    if not matching_16_18.empty:
        df.at[idx, 'n'] += (1./3.) * matching_16_18['n'].values[0]
        df_16_18 = df_16_18.drop(matching_16_18.index)

# Step 4: Create new rows for remaining '16-18'
df_16_18_remaining = df_16_18.copy()
df_16_18_remaining['resident_age_18m'] = '18-24'
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


###################################
#####################################
#####################################

microdata_itl3_counts = map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl = gltla_to_itl3, itl_name = 'itl3', var_poststrat = var_poststrat)
microdata_itl2_counts = map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl = gltla_to_itl2, itl_name = 'itl2', var_poststrat = var_poststrat)
microdata_itl1_counts = map_counts_from_gltla_to_itl(microdata_counts, gltla_to_itl = gltla_to_itl1, itl_name = 'itl1', var_poststrat = var_poststrat)


microdata_counts.rename( {v:k for k,v in var_names_quest_to_micro.items()}, axis = 1, inplace = True)
microdata_itl3_counts.rename( {v:k for k,v in var_names_quest_to_micro.items()}, axis = 1, inplace = True)
microdata_itl2_counts.rename( {v:k for k,v in var_names_quest_to_micro.items()}, axis = 1, inplace = True)
microdata_itl1_counts.rename( {v:k for k,v in var_names_quest_to_micro.items()}, axis = 1, inplace = True)

microdata_itl3_counts.to_csv('./dat/trs_microdata/england_wales/itl3.tsv', sep = '\t')
microdata_itl2_counts.to_csv('./dat/trs_microdata/england_wales/itl2.tsv', sep = '\t')
microdata_itl1_counts.to_csv('./dat/trs_microdata/england_wales/itl1.tsv', sep = '\t')

microdata_counts.to_csv('./dat/trs_microdata/england_wales/microdata_counts.tsv', sep = '\t')


































